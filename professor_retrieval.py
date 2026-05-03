"""
professor_retrieval.py  —  Phase 5 Step 3
Hybrid retrieval engine for the professor publication corpus.

Pipeline per query:
  1. Embed query with Cohere (input_type="search_query")
  2. Semantic search via Pinecone (top-20)
  3. BM25 keyword search over local corpus (top-20)
  4. Reciprocal Rank Fusion → top-k results with provenance

Designed to be imported directly by professor_api.py (FastAPI).
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cohere
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

# ── configuration ──────────────────────────────────────────────────────────────

CORPUS_PATH  = Path('rag_corpus.json')
COHERE_MODEL = 'embed-english-v3.0'

VECTOR_TOP_K = 20   # candidates from Pinecone
BM25_TOP_K   = 20   # candidates from BM25
RRF_K        = 60   # RRF constant (higher = less rank-position sensitivity)
DEFAULT_TOP_K = 8   # final results returned to caller


# ── result type ────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    chunk_id:       str
    publication_id: int
    title:          str
    authors:        str
    year:           int
    doi:            str
    pmid:           str
    pmcid:          str
    source_file:    str
    file_type:      str
    chunk_text:     str
    rrf_score:      float
    vector_rank:    Optional[int] = None   # 1-based rank from Pinecone (None if absent)
    bm25_rank:      Optional[int] = None   # 1-based rank from BM25 (None if absent)


# ── BM25 tokeniser ─────────────────────────────────────────────────────────────

_STOP = frozenset(
    'a an the of in on at to for and or by with from is are was were its'
    ' this that these those be been being have has had do does did will would'
    ' could should may might shall can'.split()
)


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop words."""
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if t not in _STOP and len(t) > 1]


# ── Retriever class ────────────────────────────────────────────────────────────

class Retriever:
    """
    Stateful retriever: initialise once, call retrieve() per query.
    Holds Pinecone + Cohere connections and BM25 index in memory.
    """

    def __init__(self) -> None:
        load_dotenv()
        self._check_env()

        self._co    = cohere.ClientV2(api_key=os.environ['COHERE_API_KEY'])
        pc          = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        self._index = pc.Index(os.environ['PINECONE_INDEX'])

        self._corpus = self._load_corpus()
        self._bm25, self._bm25_idx = self._build_bm25()

        print(f'Retriever ready — {len(self._corpus)} chunks, '
              f'{len({c["publication_id"] for c in self._corpus})} publications')

    # ── initialisation helpers ────────────────────────────────────────────────

    @staticmethod
    def _check_env() -> None:
        required = ('PINECONE_API_KEY', 'PINECONE_INDEX', 'COHERE_API_KEY')
        missing  = [k for k in required if not os.environ.get(k)]
        if missing:
            print(f'ERROR: missing env vars: {missing}', file=sys.stderr)
            sys.exit(1)

    @staticmethod
    def _load_corpus() -> list[dict]:
        if not CORPUS_PATH.exists():
            print(f'ERROR: {CORPUS_PATH} not found. Run build_rag_corpus.py.',
                  file=sys.stderr)
            sys.exit(1)
        return json.loads(CORPUS_PATH.read_text(encoding='utf-8'))

    def _build_bm25(self) -> tuple[BM25Okapi, list[dict]]:
        """
        Build a BM25 index over all corpus chunks.
        Returns (bm25_model, ordered_chunk_list) — the list order matches
        the BM25 internal document ordering, so scores[i] → bm25_idx[i].
        """
        tokenised = [_tokenise(c['chunk_text']) for c in self._corpus]
        return BM25Okapi(tokenised), list(self._corpus)

    # ── filter helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _pinecone_filter(
        publication_id: Optional[int],
        year_from:      Optional[int],
        year_to:        Optional[int],
        doi:            Optional[str],
    ) -> Optional[dict]:
        """
        Build a Pinecone metadata filter dict, or None if no filters set.
        Note: author_contains is NOT applied here — Pinecone has no substring
        operator. It is applied as post-processing on vector results instead.
        """
        clauses: list[dict] = []

        if publication_id is not None:
            clauses.append({'publication_id': {'$eq': publication_id}})
        if doi:
            clauses.append({'doi': {'$eq': doi.strip()}})
        if year_from is not None and year_to is not None:
            clauses.append({'year': {'$gte': year_from, '$lte': year_to}})
        elif year_from is not None:
            clauses.append({'year': {'$gte': year_from}})
        elif year_to is not None:
            clauses.append({'year': {'$lte': year_to}})

        if not clauses:
            return None
        return {'$and': clauses} if len(clauses) > 1 else clauses[0]

    @staticmethod
    def _passes_local_filter(
        chunk: dict,
        publication_id: Optional[int],
        year_from:      Optional[int],
        year_to:        Optional[int],
        doi:            Optional[str],
        author_contains: Optional[str],
    ) -> bool:
        """Apply the same filters locally for the BM25 pass."""
        if publication_id is not None and chunk['publication_id'] != publication_id:
            return False
        if doi and chunk.get('doi', '').strip() != doi.strip():
            return False
        if year_from is not None and (chunk.get('year') or 0) < year_from:
            return False
        if year_to is not None and (chunk.get('year') or 9999) > year_to:
            return False
        if author_contains:
            if author_contains.lower() not in chunk.get('authors', '').lower():
                return False
        return True

    # ── embedding ─────────────────────────────────────────────────────────────

    def _embed_query(self, text: str) -> list[float]:
        resp = self._co.embed(
            texts=[text],
            model=COHERE_MODEL,
            input_type='search_query',   # asymmetric: query != document
            embedding_types=['float'],
        )
        return resp.embeddings.float_[0]

    # ── vector search ─────────────────────────────────────────────────────────

    def _vector_search(
        self,
        embedding: list[float],
        top_k: int,
        pinecone_filter: Optional[dict],
    ) -> list[dict]:
        """
        Query Pinecone. Returns list of dicts with chunk metadata
        (chunk_text included — no corpus re-read needed).
        """
        kwargs: dict = {
            'vector':          embedding,
            'top_k':           top_k,
            'include_metadata': True,
        }
        if pinecone_filter:
            kwargs['filter'] = pinecone_filter

        resp    = self._index.query(**kwargs)
        results = []
        for match in resp.matches:
            meta = match.metadata or {}
            results.append({
                'chunk_id':       match.id,
                'publication_id': int(meta.get('publication_id', 0)),
                'title':          meta.get('title', ''),
                'authors':        meta.get('authors', ''),
                'year':           int(meta.get('year', 0)),
                'doi':            meta.get('doi', ''),
                'pmid':           meta.get('pmid', ''),
                'pmcid':          meta.get('pmcid', ''),
                'source_file':    meta.get('source_file', ''),
                'file_type':      meta.get('file_type', ''),
                'chunk_text':     meta.get('chunk_text', ''),
                'vector_score':   match.score,
            })
        return results

    # ── BM25 search ───────────────────────────────────────────────────────────

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        publication_id: Optional[int],
        year_from:      Optional[int],
        year_to:        Optional[int],
        doi:            Optional[str],
        author_contains: Optional[str],
    ) -> list[dict]:
        """
        Score all corpus chunks with BM25, apply filters, return top-k.
        """
        tokens = _tokenise(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: list[dict] = []
        for idx, score in ranked:
            if len(results) >= top_k:
                break
            chunk = self._bm25_idx[idx]
            if not self._passes_local_filter(
                chunk, publication_id, year_from, year_to, doi, author_contains
            ):
                continue
            results.append({**chunk, 'bm25_score': float(score)})

        return results

    # ── RRF fusion ────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_fuse(
        vector_results: list[dict],
        bm25_results:   list[dict],
        top_k:          int,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion.
        score(d) = Σ  1 / (RRF_K + rank_i(d))
        Merges both ranked lists by chunk_id.
        """
        scores:      dict[str, float] = {}
        vector_rank: dict[str, int]   = {}
        bm25_rank:   dict[str, int]   = {}
        chunk_meta:  dict[str, dict]  = {}

        for rank, item in enumerate(vector_results, 1):
            cid = item['chunk_id']
            scores[cid]      = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            vector_rank[cid] = rank
            chunk_meta[cid]  = item

        for rank, item in enumerate(bm25_results, 1):
            cid = item['chunk_id']
            scores[cid]    = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            bm25_rank[cid] = rank
            if cid not in chunk_meta:
                chunk_meta[cid] = item

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        fused  = []
        for cid, rrf_score in ranked:
            meta = chunk_meta[cid]
            fused.append({
                **meta,
                'rrf_score':    rrf_score,
                'vector_rank':  vector_rank.get(cid),
                'bm25_rank':    bm25_rank.get(cid),
            })
        return fused

    # ── public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        publication_id: Optional[int] = None,
        year_from:      Optional[int] = None,
        year_to:        Optional[int] = None,
        doi:            Optional[str] = None,
        author_contains: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """
        Full hybrid retrieval pipeline.

        Args:
            question:        Natural-language query.
            top_k:           Number of final results to return.
            publication_id:  Restrict to a single publication by ID.
            year_from:       Restrict to publications >= this year.
            year_to:         Restrict to publications <= this year.
            doi:             Restrict to exact DOI match.
            author_contains: Restrict to publications whose author string
                             contains this substring (case-insensitive).

        Returns:
            List of RetrievalResult, ordered by RRF score descending.
        """
        if not question.strip():
            return []

        # 1. Embed query
        embedding = self._embed_query(question)

        # 2. Pinecone filter (server-side: publication_id, year, doi only)
        pf = self._pinecone_filter(publication_id, year_from, year_to, doi)

        # 3. Vector search
        vector_results = self._vector_search(embedding, VECTOR_TOP_K, pf)

        # Apply author_contains post-hoc on vector results (Pinecone has no
        # substring operator — this must be done locally)
        if author_contains:
            needle = author_contains.lower()
            vector_results = [
                r for r in vector_results
                if needle in r.get('authors', '').lower()
            ]

        # 4. BM25 search (with same filters applied locally)
        bm25_results = self._bm25_search(
            question, BM25_TOP_K,
            publication_id, year_from, year_to, doi, author_contains,
        )

        # 5. RRF fusion
        fused = self._rrf_fuse(vector_results, bm25_results, top_k)

        # 6. Package into typed results
        return [
            RetrievalResult(
                chunk_id       = r['chunk_id'],
                publication_id = r['publication_id'],
                title          = r['title'],
                authors        = r['authors'],
                year           = r['year'],
                doi            = r['doi'],
                pmid           = r.get('pmid', ''),
                pmcid          = r.get('pmcid', ''),
                source_file    = r['source_file'],
                file_type      = r['file_type'],
                chunk_text     = r['chunk_text'],
                rrf_score      = r['rrf_score'],
                vector_rank    = r.get('vector_rank'),
                bm25_rank      = r.get('bm25_rank'),
            )
            for r in fused
        ]

    def corpus_stats(self) -> dict:
        """Return corpus summary — used by /corpus-status endpoint."""
        from collections import Counter
        years  = Counter(c['year'] for c in self._corpus)
        ftypes = Counter(c['file_type'] for c in self._corpus)
        pubs   = len({c['publication_id'] for c in self._corpus})
        return {
            'total_chunks':        len(self._corpus),
            'total_publications':  pubs,
            'file_type_counts':    dict(sorted(ftypes.items())),
            'year_distribution':   dict(sorted(years.items())),
            'index_name':          os.environ.get('PINECONE_INDEX', ''),
            'embedding_model':     COHERE_MODEL,
        }


# ── CLI test harness ───────────────────────────────────────────────────────────

def _print_result(i: int, r: RetrievalResult) -> None:
    vr = f'v={r.vector_rank}' if r.vector_rank else 'v=-'
    br = f'b={r.bm25_rank}'  if r.bm25_rank  else 'b=-'
    print(f'  [{i}] rrf={r.rrf_score:.4f}  {vr}  {br}  pub={r.publication_id}  {r.year}')
    print(f'       {r.title[:75]}')
    print(f'       {r.chunk_text[:160].strip()}')
    print()


def _run_tests(retriever: Retriever) -> None:
    TEST_QUERIES = [
        # (label, question, filters)
        ('HIV mother-to-child transmission breastfeeding',
         'What is the risk of HIV transmission through breastfeeding and how can it be reduced?',
         {}),
        ('Nevirapine resistance ART',
         'What resistance mutations develop after single-dose nevirapine prophylaxis?',
         {}),
        ('COVID-19 HIV-positive women outcomes',
         'How does COVID-19 affect outcomes in women living with HIV?',
         {}),
        ('ART effect on infant neurodevelopment',
         'Do antiretrovirals taken during pregnancy affect neurodevelopment in HIV-exposed uninfected children?',
         {}),
        ('Publication year filter 2020+',
         'HIV viral load suppression and antiretroviral adherence',
         {'year_from': 2020}),
        ('Author filter: Kumwenda',
         'nevirapine prophylaxis infant outcomes',
         {'author_contains': 'Kumwenda'}),
    ]

    W = 70
    for label, question, filters in TEST_QUERIES:
        print(f'\n{"="*W}')
        print(f'TEST: {label}')
        print(f'Q   : {question}')
        if filters:
            print(f'FLT : {filters}')
        print(f'{"="*W}')
        results = retriever.retrieve(question, top_k=4, **filters)
        if not results:
            print('  (no results)')
        for i, r in enumerate(results, 1):
            _print_result(i, r)


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    r = Retriever()
    _run_tests(r)
