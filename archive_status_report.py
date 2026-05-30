"""
archive_status_report.py
Read-only reporting script — does NOT modify Pinecone, rag_corpus.json,
or downloaded_papers/.

Outputs:
  indexed_papers_report.csv  — publications present in the searchable archive
  missing_papers_report.csv  — publications still lacking full text

Sources read:
  enriched_publications.json  (primary metadata for all 200 pubs)
  rag_corpus.json             (which pubs are indexed + chunk counts)
"""

import csv
import json
from collections import Counter
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────

ENRICHED_PATH  = Path('enriched_publications.json')
CORPUS_PATH    = Path('rag_corpus.json')
INDEXED_CSV    = Path('indexed_papers_report.csv')
MISSING_CSV    = Path('missing_papers_report.csv')


# ── load data ──────────────────────────────────────────────────────────────────

def _load() -> tuple[list[dict], list[dict]]:
    for p in (ENRICHED_PATH, CORPUS_PATH):
        if not p.exists():
            raise FileNotFoundError(f'{p} not found')
    enriched = json.loads(ENRICHED_PATH.read_text(encoding='utf-8'))
    corpus   = json.loads(CORPUS_PATH.read_text(encoding='utf-8'))
    return enriched, corpus


# ── notes helper for missing report ───────────────────────────────────────────

def _notes(r: dict) -> str:
    parts = []
    if r.get('pmcid'):
        parts.append(f"PMC available: {r['pmcid']}")
    if r.get('pdf_url'):
        parts.append('PDF URL available')
    if r.get('is_oa') and r.get('publisher_url') and not r.get('pdf_url'):
        parts.append('Open-access publisher URL available')
    if r.get('enrichment_status') == 'not_found':
        parts.append('Not found in Crossref / PubMed')
    if not parts:
        parts.append('Manual retrieval required - no open-access source found')
    return '; '.join(parts)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    enriched, corpus = _load()

    # Index corpus by publication_id
    chunk_count:  Counter  = Counter(c['publication_id'] for c in corpus)
    corpus_meta:  dict[int, dict] = {}   # first chunk per pub (for source_file / file_type)
    for c in corpus:
        pid = c['publication_id']
        if pid not in corpus_meta:
            corpus_meta[pid] = c

    indexed_ids: set[int] = set(chunk_count.keys())

    # Build enriched lookup keyed by id
    enr_map: dict[int, dict] = {r['id']: r for r in enriched}

    indexed_rows: list[dict] = []
    missing_rows: list[dict] = []

    for pub_id in sorted(enr_map.keys()):
        r = enr_map[pub_id]

        if pub_id in indexed_ids:
            cm = corpus_meta[pub_id]
            indexed_rows.append({
                'publication_id': pub_id,
                'title':          r.get('title_guess') or '',
                'authors':        r.get('authors_guess') or '',
                'year':           r.get('year') or '',
                'journal':        r.get('journal_guess') or '',
                'doi':            r.get('doi') or '',
                'pmid':           r.get('pmid') or '',
                'pmcid':          r.get('pmcid') or '',
                'source_file':    cm['source_file'],
                'file_type':      cm['file_type'],
                'chunk_count':    chunk_count[pub_id],
            })
        else:
            missing_rows.append({
                'publication_id':    pub_id,
                'title':             r.get('title_guess') or '',
                'authors':           r.get('authors_guess') or '',
                'year':              r.get('year') or '',
                'journal':           r.get('journal_guess') or '',
                'doi':               r.get('doi') or '',
                'pmid':              r.get('pmid') or '',
                'pmcid':             r.get('pmcid') or '',
                'enrichment_status': r.get('enrichment_status') or '',
                'publisher_url':     r.get('publisher_url') or '',
                'pdf_url':           r.get('pdf_url') or '',
                'notes':             _notes(r),
            })

    # ── write CSVs ────────────────────────────────────────────────────────────

    _write_csv(INDEXED_CSV, indexed_rows, [
        'publication_id', 'title', 'authors', 'year', 'journal',
        'doi', 'pmid', 'pmcid', 'source_file', 'file_type', 'chunk_count',
    ])

    _write_csv(MISSING_CSV, missing_rows, [
        'publication_id', 'title', 'authors', 'year', 'journal',
        'doi', 'pmid', 'pmcid', 'enrichment_status',
        'publisher_url', 'pdf_url', 'notes',
    ])

    # ── summary ───────────────────────────────────────────────────────────────

    total   = len(enr_map)
    indexed = len(indexed_rows)
    missing = len(missing_rows)
    pct     = 100 * indexed // total

    def _year_range(rows: list[dict]) -> str:
        years = [int(r['year']) for r in rows if r.get('year')]
        if not years:
            return 'unknown'
        return f'{min(years)}-{max(years)}'

    W = 55
    print()
    print('=' * W)
    print('  ARCHIVE STATUS REPORT')
    print('=' * W)
    print(f'  Total publications       : {total}')
    print(f'  Indexed (searchable)     : {indexed}  ({pct}%)')
    print(f'  Missing full text        : {missing}  ({100 - pct}%)')
    print(f'  Year range - indexed     : {_year_range(indexed_rows)}')
    print(f'  Year range - missing     : {_year_range(missing_rows)}')
    print(f'  Total chunks in archive  : {sum(chunk_count.values())}')
    print('=' * W)
    print(f'  Reports written:')
    print(f'    {INDEXED_CSV}')
    print(f'    {MISSING_CSV}')
    print('=' * W)
    print()

    # ── preview first 10 rows ─────────────────────────────────────────────────

    _preview('INDEXED PAPERS (first 10)', indexed_rows[:10], [
        'publication_id', 'year', 'chunk_count', 'file_type', 'title',
    ])

    _preview('MISSING PAPERS (first 10)', missing_rows[:10], [
        'publication_id', 'year', 'enrichment_status', 'notes', 'title',
    ])


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _preview(label: str, rows: list[dict], cols: list[str]) -> None:
    print(f'--- {label} ---')
    if not rows:
        print('  (empty)')
        print()
        return
    # column widths
    widths = {c: max(len(c), max(len(str(r.get(c, ''))) for r in rows)) for c in cols}
    widths = {c: min(w, 55) for c, w in widths.items()}   # cap long columns

    header = '  ' + '  '.join(c.ljust(widths[c]) for c in cols)
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for r in rows:
        line = '  ' + '  '.join(str(r.get(c, ''))[:widths[c]].ljust(widths[c]) for c in cols)
        print(line)
    print()


if __name__ == '__main__':
    main()
