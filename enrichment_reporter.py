import json
import sys
from datetime import date
from pathlib import Path

PHASE2_PATH = Path('publication_metadata.json')
PHASE3_PATH = Path('enriched_publications.json')
REPORT_PATH = Path('enrichment_report.json')


# -- helpers -------------------------------------------------------------------

def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding='utf-8'))


def _pick(r: dict) -> dict:
    """Return a compact preview dict (abstract truncated)."""
    keys = (
        'id', 'doi', 'pmid', 'pmcid', 'year',
        'title_guess', 'authors_guess', 'journal_guess',
        'publisher_url', 'pdf_url', 'is_oa',
        'enrichment_status', 'enrichment_source',
    )
    out = {k: r.get(k) for k in keys}
    ab = r.get('abstract') or ''
    out['abstract'] = (ab[:200] + '…') if len(ab) > 200 else (ab or None)
    return out


def _row(r: dict, width: int = 55) -> None:
    """Print a compact two-line record summary."""
    title  = (r.get('title_guess') or '')[:width]
    author = (r.get('authors_guess') or '')[:40]
    print(f"  [{r['id']:3d}]  status={r.get('enrichment_status','?'):<10}"
          f"doi={r.get('doi') or '--'}")
    print(f"        abstract={bool(r.get('abstract'))}"
          f"  pdf={r.get('pdf_url') is not None}"
          f"  oa={r.get('is_oa')}"
          f"  year={r.get('year')}")
    print(f"        {author}")
    print(f"        {title}")


# -- main ----------------------------------------------------------------------

def main() -> None:
    for p in (PHASE2_PATH, PHASE3_PATH):
        if not p.exists():
            print(f'ERROR: {p} not found.', file=sys.stderr)
            if p is PHASE3_PATH:
                print('Run publication_enricher.py first.', file=sys.stderr)
            sys.exit(1)

    phase2 = _load(PHASE2_PATH)
    phase3 = _load(PHASE3_PATH)

    # -- before / after identifier counts -------------------------------------
    doi_before   = sum(1 for r in phase2 if r.get('doi'))
    doi_after    = sum(1 for r in phase3 if r.get('doi'))
    pmid_before  = sum(1 for r in phase2 if r.get('pmid'))
    pmid_after   = sum(1 for r in phase3 if r.get('pmid'))
    pmcid_before = sum(1 for r in phase2 if r.get('pmcid'))
    pmcid_after  = sum(1 for r in phase3 if r.get('pmcid'))

    # -- outcome groups --------------------------------------------------------
    full_recs     = [r for r in phase3 if r.get('enrichment_status') == 'full']
    partial_recs  = [r for r in phase3 if r.get('enrichment_status') == 'partial']
    notfound_recs = [r for r in phase3 if r.get('enrichment_status') == 'not_found']

    abstract_recs = [r for r in phase3 if r.get('abstract')]
    pdf_recs      = [r for r in phase3 if r.get('pdf_url')]
    missing_doi   = [r for r in phase3 if not r.get('doi')]
    manual_review = notfound_recs   # no identifier found → needs human lookup

    # -- build report ----------------------------------------------------------
    summary = {
        'generated_at':       str(date.today()),
        'total':              len(phase3),
        'doi_before':         doi_before,
        'doi_after':          doi_after,
        'doi_gained':         doi_after  - doi_before,
        'pmid_before':        pmid_before,
        'pmid_after':         pmid_after,
        'pmid_gained':        pmid_after  - pmid_before,
        'pmcid_before':       pmcid_before,
        'pmcid_after':        pmcid_after,
        'pmcid_gained':       pmcid_after - pmcid_before,
        'abstracts_found':    len(abstract_recs),
        'pdf_links_found':    len(pdf_recs),
        'pdfs_downloaded':    0,            # Phase 4 will download binaries
        'metadata_only':      len(partial_recs),
        'failed_lookups':     len(notfound_recs),
        'manual_review_count': len(manual_review),
        'status_full':        len(full_recs),
        'status_partial':     len(partial_recs),
        'status_not_found':   len(notfound_recs),
    }

    report = {
        'summary':            summary,
        'sample_enriched':    [_pick(r) for r in phase3[:5]],
        'sample_with_pdf':    [_pick(r) for r in pdf_recs[:10]],
        'sample_missing_doi': [_pick(r) for r in missing_doi[:10]],
        'manual_review':      [_pick(r) for r in manual_review],
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    # -- printed summary -------------------------------------------------------
    W = 60
    def hdr(t): print(f'\n{"-" * W}\n  {t}\n{"-" * W}')

    hdr('ENRICHMENT REPORT  --  Phase 3')
    print(f'  Generated          : {summary["generated_at"]}')
    print(f'  Total publications : {summary["total"]}')
    print()
    print(f'  {"Identifier":<12}  {"Before":>6}  {"After":>6}  {"Gained":>6}')
    print(f'  {"-"*12}  {"-"*6}  {"-"*6}  {"-"*6}')
    print(f'  {"DOI":<12}  {doi_before:>6}  {doi_after:>6}  {doi_after-doi_before:>+6}')
    print(f'  {"PMID":<12}  {pmid_before:>6}  {pmid_after:>6}  {pmid_after-pmid_before:>+6}')
    print(f'  {"PMCID":<12}  {pmcid_before:>6}  {pmcid_after:>6}  {pmcid_after-pmcid_before:>+6}')
    print()
    print(f'  Abstracts found    : {len(abstract_recs)}')
    print(f'  Legal PDF links    : {len(pdf_recs)}')
    print(f'  PDFs downloaded    : 0  (Phase 4)')
    print()
    print(f'  Fully enriched     : {len(full_recs)}')
    print(f'  Metadata only      : {len(partial_recs)}  (doi/year/journal but no abstract)')
    print(f'  Failed lookups     : {len(notfound_recs)}')
    print(f'  Manual review      : {len(manual_review)}')
    print()
    print(f'  Report saved to    : {REPORT_PATH}')

    # -- first 5 enriched -----------------------------------------------------
    hdr('First 5 enriched records')
    for r in phase3[:5]:
        _row(r)

    # -- 10 records with PDF links ---------------------------------------------
    hdr(f'Records with PDF links  ({len(pdf_recs)} total -- showing up to 10)')
    if pdf_recs:
        for r in pdf_recs[:10]:
            print(f"  [{r['id']:3d}]  {r.get('pdf_url')}")
            print(f"        {(r.get('title_guess') or '')[:65]}")
    else:
        print('  (none)')

    # -- 10 records still missing DOI ------------------------------------------
    hdr(f'Records still missing DOI  ({len(missing_doi)} total -- showing up to 10)')
    if missing_doi:
        for r in missing_doi[:10]:
            print(f"  [{r['id']:3d}]  year={r.get('year')}  "
                  f"{(r.get('authors_guess') or '')[:45]}")
            print(f"        {(r.get('title_guess') or '')[:65]}")
    else:
        print('  (none -- all publications have a DOI)')

    # -- manual review records -------------------------------------------------
    hdr(f'Manual review needed  ({len(manual_review)} records)')
    if manual_review:
        for r in manual_review[:10]:
            print(f"  [{r['id']:3d}]  year={r.get('year')}  "
                  f"{(r.get('authors_guess') or '')[:45]}")
            print(f"        {(r.get('title_guess') or '')[:65]}")
        if len(manual_review) > 10:
            print(f'  ... and {len(manual_review) - 10} more '
                  f'(see enrichment_report.json -> manual_review)')
    else:
        print('  (none)')

    print()


if __name__ == '__main__':
    main()
