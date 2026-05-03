import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

# ── configuration ─────────────────────────────────────────────────────────────

CROSSREF_MAILTO       = 'brothernoomsai@gmail.com'
UNPAYWALL_EMAIL       = 'brothernoomsai@gmail.com'
TITLE_MATCH_THRESHOLD = 0.55   # Jaccard similarity for accepting a title-search result
YEAR_TOLERANCE        = 2      # years ± accepted when matching search results

# Minimum seconds between successive calls per domain (conservative / polite)
_GAP = {
    'crossref':  1.0,
    'unpaywall': 1.0,
    'pubmed':    0.4,   # NCBI max 3 req/s without API key
    'openalex':  1.0,
}

# ── rate limiter ──────────────────────────────────────────────────────────────

_last: dict[str, float] = {}

def _throttle(domain: str) -> None:
    gap = _GAP.get(domain, 1.0)
    wait = gap - (time.time() - _last.get(domain, 0.0))
    if wait > 0:
        time.sleep(wait)
    _last[domain] = time.time()


# ── HTTP session ──────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers['User-Agent'] = (
        f'ProfessorPublicationAssistant/1.0 (mailto:{CROSSREF_MAILTO})'
    )
    return s


# ── title similarity ──────────────────────────────────────────────────────────

_STOP = frozenset(
    'a an the of in on at to for and or by with from is are was were its'
    .split()
)

def _sim(a: str, b: str) -> float:
    """Jaccard similarity of meaningful lowercased word sets."""
    def tok(s):
        return {w for w in re.findall(r'[a-z0-9]+', s.lower()) if w not in _STOP}
    wa, wb = tok(a), tok(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ── Crossref ──────────────────────────────────────────────────────────────────

def _crossref_by_doi(doi: str, s: requests.Session) -> dict:
    """Fetch verified metadata from Crossref for a known DOI."""
    _throttle('crossref')
    try:
        r = s.get(
            f'https://api.crossref.org/works/{quote(doi, safe="/")}',
            timeout=15,
        )
        if r.status_code != 200:
            return {}
        msg = r.json()['message']

        # Publication year — prefer print date over online/issued
        year = None
        for key in ('published', 'published-print', 'published-online', 'issued'):
            parts = (msg.get(key) or {}).get('date-parts', [[]])
            if parts and parts[0]:
                year = int(parts[0][0])
                break

        # Abstract: strip JATS XML markup that Crossref embeds
        abstract = msg.get('abstract') or ''
        abstract = re.sub(r'<[^>]+>', ' ', abstract)
        abstract = re.sub(r'\s+', ' ', abstract).strip() or None

        # Journal name
        journal = (
            (msg.get('container-title') or [None])[0]
            or (msg.get('short-container-title') or [None])[0]
        )

        # Publisher URL
        url = (
            msg.get('URL')
            or (msg.get('resource') or {}).get('primary', {}).get('URL')
        )

        return {
            'title_cr':      (msg.get('title') or [''])[0],
            'year_cr':       year,
            'journal_cr':    journal,
            'abstract':      abstract,
            'publisher_url': url,
        }
    except Exception:
        return {}


def _crossref_search(
    title: str, first_author: str, year: int | None, s: requests.Session
) -> str | None:
    """Search Crossref by title (+author) and return the best-matching DOI."""
    if not title:
        return None
    _throttle('crossref')
    params: dict = {
        'query.title': title,
        'rows': 5,
        'select': 'DOI,title,published',
    }
    if first_author:
        params['query.author'] = first_author
    try:
        r = s.get('https://api.crossref.org/works', params=params, timeout=15)
        if r.status_code != 200:
            return None
        for item in r.json()['message']['items']:
            cand_title = (item.get('title') or [''])[0]
            if _sim(title, cand_title) < TITLE_MATCH_THRESHOLD:
                continue
            if year:
                parts = (item.get('published') or {}).get('date-parts', [[]])
                if parts and parts[0]:
                    if abs(int(parts[0][0]) - year) > YEAR_TOLERANCE:
                        continue
            return item.get('DOI')
    except Exception:
        pass
    return None


# ── Unpaywall ─────────────────────────────────────────────────────────────────

def _unpaywall(doi: str, s: requests.Session) -> dict:
    """Return open-access status and best legal PDF URL."""
    _throttle('unpaywall')
    try:
        r = s.get(
            f'https://api.unpaywall.org/v2/{quote(doi, safe="/")}',
            params={'email': UNPAYWALL_EMAIL},
            timeout=15,
        )
        if r.status_code != 200:
            return {}
        data = r.json()
        best = data.get('best_oa_location') or {}
        return {
            'is_oa':   data.get('is_oa'),
            'pdf_url': best.get('url_for_pdf') or best.get('url'),
        }
    except Exception:
        return {}


# ── PubMed ────────────────────────────────────────────────────────────────────

def _pubmed_search(doi: str, s: requests.Session) -> str | None:
    """Find the PMID for a DOI via NCBI esearch."""
    _throttle('pubmed')
    try:
        r = s.get(
            'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
            params={'db': 'pubmed', 'term': f'{doi}[doi]', 'retmode': 'json'},
            timeout=15,
        )
        ids = r.json()['esearchresult']['idlist']
        return ids[0] if ids else None
    except Exception:
        return None


def _pubmed_fetch(pmid: str, s: requests.Session) -> dict:
    """Fetch abstract, PMCID, and DOI for a PMID via NCBI efetch."""
    _throttle('pubmed')
    try:
        r = s.get(
            'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',
            params={'db': 'pubmed', 'id': pmid, 'retmode': 'xml'},
            timeout=15,
        )
        xml = r.text

        # Abstract (may have multiple labelled sections)
        parts = re.findall(
            r'<AbstractText(?:[^>]* Label="([^"]*)")?[^>]*>(.*?)</AbstractText>',
            xml, re.DOTALL,
        )
        abstract_pieces = []
        for label, body in parts:
            body = re.sub(r'<[^>]+>', '', body).strip()
            abstract_pieces.append(f'{label}: {body}' if label else body)
        abstract = ' '.join(abstract_pieces).strip() or None

        pmcid_m  = re.search(r'<ArticleId IdType="pmc">(PMC\d+)</ArticleId>', xml)
        doi_m    = re.search(r'<ArticleId IdType="doi">([^<]+)</ArticleId>', xml)
        return {
            'abstract_pm': abstract,
            'pmcid':       pmcid_m.group(1) if pmcid_m else None,
            'doi_pm':      doi_m.group(1).strip() if doi_m else None,
        }
    except Exception:
        return {}


# ── OpenAlex fallback ─────────────────────────────────────────────────────────

def _openalex_search(title: str, year: int | None, s: requests.Session) -> str | None:
    """Fallback title search via OpenAlex; returns DOI string or None."""
    if not title:
        return None
    _throttle('openalex')
    params: dict = {
        'search':   title,
        'per-page': 5,
        'select':   'doi,title,publication_year',
        'mailto':   CROSSREF_MAILTO,
    }
    try:
        r = s.get('https://api.openalex.org/works', params=params, timeout=15)
        if r.status_code != 200:
            return None
        for item in r.json().get('results', []):
            cand = item.get('title') or ''
            if _sim(title, cand) < TITLE_MATCH_THRESHOLD:
                continue
            if year and item.get('publication_year'):
                if abs(item['publication_year'] - year) > YEAR_TOLERANCE:
                    continue
            doi = item.get('doi') or ''
            if doi:
                return doi.removeprefix('https://doi.org/')
    except Exception:
        pass
    return None


# ── per-record orchestration ──────────────────────────────────────────────────

def enrich_record(pub: dict, s: requests.Session) -> dict:
    rec: dict = {
        **pub,
        'abstract':          None,
        'publisher_url':     None,
        'pdf_url':           None,
        'is_oa':             None,
        'enrichment_source': None,
        'enrichment_status': 'not_found',
    }
    sources: list[str] = []
    doi  = rec.get('doi')
    pmid = rec.get('pmid')

    # ── step 1: find DOI if not known ────────────────────────────────────────
    if not doi:
        title     = rec.get('title_guess') or ''
        raw_au    = rec.get('authors_guess') or ''
        first_au  = re.split(r'[,;]', raw_au)[0].strip()
        year      = rec.get('year')

        doi = _crossref_search(title, first_au, year, s)
        if doi:
            rec['doi'] = doi
            sources.append('crossref_search')
        else:
            doi = _openalex_search(title, year, s)
            if doi:
                rec['doi'] = doi
                sources.append('openalex_search')

    # ── step 2: Crossref metadata by DOI ─────────────────────────────────────
    if doi:
        cr = _crossref_by_doi(doi, s)
        if cr:
            if not rec['year']          and cr.get('year_cr'):
                rec['year']         = cr['year_cr']
            if not rec['journal_guess'] and cr.get('journal_cr'):
                rec['journal_guess'] = cr['journal_cr']
            if cr.get('abstract'):
                rec['abstract'] = cr['abstract']
            if cr.get('publisher_url'):
                rec['publisher_url'] = cr['publisher_url']
            sources.append('crossref')
        rec['enrichment_status'] = 'partial'

        # ── step 3: Unpaywall for open-access links ───────────────────────
        uw = _unpaywall(doi, s)
        if uw.get('is_oa') is not None:
            rec['is_oa']   = uw['is_oa']
            rec['pdf_url'] = uw.get('pdf_url')
            sources.append('unpaywall')

        # ── step 4: PubMed for abstract and PMCID ────────────────────────
        if not pmid:
            pmid = _pubmed_search(doi, s)
            if pmid:
                rec['pmid'] = pmid
        if pmid:
            pm = _pubmed_fetch(pmid, s)
            if not rec['abstract'] and pm.get('abstract_pm'):
                rec['abstract'] = pm['abstract_pm']
            if not rec.get('pmcid') and pm.get('pmcid'):
                rec['pmcid'] = pm['pmcid']
            if not rec['doi'] and pm.get('doi_pm'):
                rec['doi'] = pm['doi_pm']
            if any(pm.get(k) for k in ('abstract_pm', 'pmcid', 'doi_pm')):
                sources.append('pubmed')

        rec['enrichment_status'] = 'full' if rec['abstract'] else 'partial'

    # Deduplicate sources while preserving order
    rec['enrichment_source'] = '+'.join(dict.fromkeys(sources)) or None
    return rec


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    input_path  = Path('publication_metadata.json')
    output_path = Path('enriched_publications.json')

    if not input_path.exists():
        print(f'ERROR: {input_path} not found', file=sys.stderr)
        sys.exit(1)

    pubs = json.loads(input_path.read_text(encoding='utf-8'))

    # Resume: skip records already successfully enriched
    done: dict[int, dict] = {}
    if output_path.exists():
        try:
            for r in json.loads(output_path.read_text(encoding='utf-8')):
                if r.get('enrichment_status') in ('full', 'partial', 'not_found'):
                    done[r['id']] = r
            if done:
                print(f'Resuming: {len(done)} records already enriched, skipping.')
        except Exception:
            pass

    results: list[dict] = []
    s = _make_session()
    skipped = 0

    for i, pub in enumerate(pubs, 1):
        pid = pub['id']
        if pid in done:
            results.append(done[pid])
            skipped += 1
            continue

        print(f'[{i:3d}/200] id={pid:3d} ...', end=' ', flush=True)
        rec = enrich_record(pub, s)
        results.append(rec)

        status = rec['enrichment_status']
        src    = rec['enrichment_source'] or 'none'
        print(f'{status}  ({src})')

        # Incremental save after every record
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )

    # Final save (covers skipped records that were never written incrementally)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    # ── final summary ─────────────────────────────────────────────────────────
    full    = sum(1 for r in results if r['enrichment_status'] == 'full')
    partial = sum(1 for r in results if r['enrichment_status'] == 'partial')
    nf      = sum(1 for r in results if r['enrichment_status'] == 'not_found')
    n_doi   = sum(1 for r in results if r.get('doi'))
    n_pmid  = sum(1 for r in results if r.get('pmid'))
    n_abs   = sum(1 for r in results if r.get('abstract'))
    n_pdf   = sum(1 for r in results if r.get('pdf_url'))
    n_oa    = sum(1 for r in results if r.get('is_oa'))

    print(f'\n{"-"*50}')
    print(f'Enriched {len(results)} records  ->  {output_path}')
    print(f'  status   : full={full}  partial={partial}  not_found={nf}')
    print(f'  doi      : {n_doi}')
    print(f'  pmid     : {n_pmid}')
    print(f'  abstract : {n_abs}')
    print(f'  pdf_url  : {n_pdf}  (is_oa={n_oa})')


if __name__ == '__main__':
    main()
