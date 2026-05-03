"""
vectorize_corpus.py  —  Phase 5 Step 2
Embed rag_corpus.json with Cohere and upsert vectors to Pinecone.

Input:   rag_corpus.json
Output:  Pinecone index (professor-publication-assistant)

Duplicate-safe: fetches existing vector IDs before upserting.
Re-runnable: skips already-indexed chunks, upserts only new ones.
"""

import json
import os
import sys
import time
from pathlib import Path

import cohere
from dotenv import load_dotenv
from pinecone import Pinecone

# ── configuration ──────────────────────────────────────────────────────────────

CORPUS_PATH    = Path('rag_corpus.json')

COHERE_MODEL   = 'embed-english-v3.0'
EMBED_BATCH    = 96    # Cohere max texts per embed call
UPSERT_BATCH   = 100   # Pinecone max vectors per upsert call
FETCH_BATCH    = 100   # Pinecone max IDs per fetch call


# ── helpers ────────────────────────────────────────────────────────────────────

def _batches(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _get_existing_ids(index, all_ids: list[str]) -> set[str]:
    """
    Batch-fetch chunk_ids from Pinecone.
    Returns the set of IDs that are already indexed.
    """
    existing: set[str] = set()
    for batch in _batches(all_ids, FETCH_BATCH):
        result = index.fetch(ids=batch)
        existing.update(result.vectors.keys())
    return existing


def _embed_batch(co: cohere.ClientV2, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, retrying on 429 rate-limit responses."""
    wait = 65  # seconds; Cohere trial limit resets every minute
    for attempt in range(1, 5):
        try:
            resp = co.embed(
                texts=texts,
                model=COHERE_MODEL,
                input_type='search_document',
                embedding_types=['float'],
            )
            return resp.embeddings.float_
        except Exception as exc:
            msg = str(exc)
            if '429' in msg or 'rate limit' in msg.lower():
                print(f'    [429] Rate limit hit — waiting {wait}s (attempt {attempt})...', flush=True)
                time.sleep(wait)
                wait = min(wait * 2, 300)   # cap at 5 minutes
            else:
                raise
    raise RuntimeError(f'Embed failed after retries for batch of {len(texts)} texts')


def _build_record(chunk: dict, vector: list[float]) -> dict:
    """Build a Pinecone upsert record from a corpus chunk + embedding."""
    return {
        'id': chunk['chunk_id'],
        'values': vector,
        'metadata': {
            'publication_id': chunk['publication_id'],
            'chunk_id':       chunk['chunk_id'],
            'title':          chunk['title'],
            'authors':        chunk['authors'],
            'year':           chunk['year'],
            'doi':            chunk['doi'],
            'pmid':           chunk['pmid'],
            'pmcid':          chunk['pmcid'],
            'source_file':    chunk['source_file'],
            'file_type':      chunk['file_type'],
            'chunk_text':     chunk['chunk_text'],
        },
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()

    pinecone_key = os.environ.get('PINECONE_API_KEY', '')
    cohere_key   = os.environ.get('COHERE_API_KEY', '')
    index_name   = os.environ.get('PINECONE_INDEX', 'professor-publication-assistant')

    for name, val in [('PINECONE_API_KEY', pinecone_key),
                      ('COHERE_API_KEY',   cohere_key),
                      ('PINECONE_INDEX',   index_name)]:
        if not val:
            print(f'ERROR: {name} not set in .env', file=sys.stderr)
            sys.exit(1)

    if not CORPUS_PATH.exists():
        print(f'ERROR: {CORPUS_PATH} not found. Run build_rag_corpus.py first.',
              file=sys.stderr)
        sys.exit(1)

    # ── load corpus ───────────────────────────────────────────────────────────
    corpus: list[dict] = json.loads(CORPUS_PATH.read_text(encoding='utf-8'))
    print(f'Corpus loaded: {len(corpus)} chunks from {CORPUS_PATH}')

    # ── connect ───────────────────────────────────────────────────────────────
    pc    = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)
    co    = cohere.ClientV2(api_key=cohere_key)

    stats = index.describe_index_stats()
    print(f'Pinecone index  : {index_name}  (dim={stats.dimension})')
    print(f'Vectors already : {stats.total_vector_count}')

    # ── duplicate check ───────────────────────────────────────────────────────
    all_ids = [c['chunk_id'] for c in corpus]
    print(f'Checking for existing vectors...', end=' ', flush=True)
    existing_ids = _get_existing_ids(index, all_ids)
    print(f'{len(existing_ids)} already indexed')

    to_index = [c for c in corpus if c['chunk_id'] not in existing_ids]
    if not to_index:
        print('All chunks already indexed. Nothing to do.')
        _print_final_stats(index)
        return

    print(f'To upsert: {len(to_index)} chunks')
    print()

    # ── embed + upsert loop ───────────────────────────────────────────────────
    embed_batches  = list(_batches(to_index, EMBED_BATCH))
    total_upserted = 0
    t_start        = time.time()

    for batch_num, embed_batch in enumerate(embed_batches, 1):
        texts = [c['chunk_text'] for c in embed_batch]

        # Embed
        try:
            vectors = _embed_batch(co, texts)
        except Exception as exc:
            print(f'  [ERROR] Embed batch {batch_num}: {exc}')
            continue

        # Build Pinecone records
        records = [_build_record(chunk, vec)
                   for chunk, vec in zip(embed_batch, vectors)]

        # Upsert in sub-batches of UPSERT_BATCH
        for upsert_sub in _batches(records, UPSERT_BATCH):
            try:
                index.upsert(vectors=upsert_sub)
                total_upserted += len(upsert_sub)
            except Exception as exc:
                print(f'  [ERROR] Upsert: {exc}')
                continue

        elapsed = time.time() - t_start
        pct     = 100 * total_upserted // len(to_index)
        print(f'  Batch {batch_num:3d}/{len(embed_batches)}  '
              f'chunks {total_upserted:4d}/{len(to_index)}  '
              f'({pct:3d}%)  {elapsed:.1f}s elapsed')

    # ── final verification ────────────────────────────────────────────────────
    print()
    print(f'Upserted {total_upserted} vectors in {time.time()-t_start:.1f}s')
    _print_final_stats(index)


def _print_final_stats(index) -> None:
    # Brief pause so Pinecone stats refresh
    time.sleep(2)
    stats = index.describe_index_stats()
    print()
    print(f'{"-"*50}')
    print(f'Index verification')
    print(f'  Total vectors in Pinecone : {stats.total_vector_count}')
    ns = dict(stats.namespaces)
    if ns:
        for nsname, nsdata in ns.items():
            label = nsname or '(default)'
            print(f'  Namespace {label:<20}: {nsdata.vector_count} vectors')
    print(f'{"-"*50}')


if __name__ == '__main__':
    main()
