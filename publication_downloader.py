import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# ── configuration ──────────────────────────────────────────────────────────────

NCBI_EMAIL   = 'brothernoomsai@gmail.com'
OUT_DIR      = Path('downloaded_papers')
REPORT_PATH  = Path('download_report.json')
INPUT_PATH   = Path('enriched_publications.json')

MAX_RETRIES  = 3
RETRY_WAIT   = 2.0      # seconds; doubles each attempt
MAX_FILE_MB  = 50
PAYWALL_BYTES = 8_000   # HTML under this size is likely a gate page

_GAP: dict[str, float] = {
    'pmc':     1.2,   # conservative; PMC bot-detection triggers at higher rates
    'default': 1.5,
}

_last: dict[str, float] = {}

_PAYWALL_PHRASES = frozenset([
    'sign in', 'log in', 'login required', 'subscribe', 'access denied',
    'institutional access', 'purchase access', 'buy article',
    'request access', 'register to read', 'full access', 'get access',
    # bot / captcha detection pages
    'checking your browser', 'recaptcha', 'enable javascript',
    'ddos protection', 'please wait', 'just a moment',
])


# ── transport ──────────────────────────────────────────────────────────────────

def _domain_key(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if 'ncbi.nlm.nih.gov' in host or 'pmc.ncbi' in host:
        return 'pmc'
    return 'default'


def _throttle(domain: str) -> None:
    gap  = _GAP.get(domain, _GAP['default'])
    wait = gap - (time.time() - _last.get(domain, 0.0))
    if wait > 0:
        time.sleep(wait)
    _last[domain] = time.time()


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers['User-Agent'] = (
        f'ProfessorPublicationAssistant/1.0 (mailto:{NCBI_EMAIL})'
    )
    return s


def _get_with_retry(
    url: str,
    session: requests.Session,
    stream: bool = False,
    timeout: int = 20,
) -> 'requests.Response | None':
    domain = _domain_key(url)
    wait   = RETRY_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _throttle(domain)
            r = session.get(url, stream=stream, timeout=timeout,
                            allow_redirects=True)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                after = int(r.headers.get('Retry-After', int(wait)))
                time.sleep(after)
                wait *= 2
                continue
            if r.status_code in (401, 403, 404, 410):
                return None   # non-retriable
            time.sleep(wait)
            wait *= 2
        except requests.exceptions.RequestException:
            if attempt < MAX_RETRIES:
                time.sleep(wait)
                wait *= 2
    return None


# ── validation ─────────────────────────────────────────────────────────────────

def _pdf_magic_ok(data: bytes) -> bool:
    return data[:4] == b'%PDF'


def _is_pdf_response(r: requests.Response) -> bool:
    return 'pdf' in r.headers.get('Content-Type', '').lower()


def _is_paywall_html(html: str) -> bool:
    sample = html[:3000].lower()
    hits   = sum(1 for p in _PAYWALL_PHRASES if p in sample)
    if len(html) < PAYWALL_BYTES and hits >= 1:
        return True
    return hits >= 2


# ── HTML → TXT ─────────────────────────────────────────────────────────────────

def _html_to_txt(html: str) -> str:
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script', 'style', 'nav', 'header', 'footer',
                     'aside', 'figure', 'noscript', 'form']):
        tag.decompose()
    body = (soup.find('article')
            or soup.find(id=re.compile(r'content|article|main', re.I))
            or soup.find('main')
            or soup.body
            or soup)
    lines = []
    for el in body.descendants:
        if getattr(el, 'name', None) in ('p', 'h1', 'h2', 'h3', 'h4', 'li'):
            text = el.get_text(' ', strip=True)
            if text:
                lines.append(text)
    return '\n\n'.join(lines)


# ── filename helpers ───────────────────────────────────────────────────────────

def _safe_stem(pub_id: int, title: str) -> str:
    slug = re.sub(r'[^\w\s-]', '', (title or 'untitled'))
    slug = re.sub(r'[\s_-]+', '_', slug).strip('_')
    return f'{pub_id}_{slug[:60]}'


# ── download strategies ────────────────────────────────────────────────────────

def _try_pdf(url: str, path: Path, session: requests.Session) -> bool:
    r = _get_with_retry(url, session, stream=True)
    if r is None:
        return False
    data  = b''
    total = 0
    try:
        for chunk in r.iter_content(65_536):
            data  += chunk
            total += len(chunk)
            if total > MAX_FILE_MB * 1_048_576:
                return False
    except Exception:
        return False
    if not (_pdf_magic_ok(data) or _is_pdf_response(r)):
        return False
    path.write_bytes(data)
    return True


def _try_pmc_pdf(pmcid: str, path: Path, session: requests.Session) -> tuple[bool, str]:
    url = f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/'
    return _try_pdf(url, path, session), url


_MIN_WORDS = 80   # pages with fewer words are bot/captcha/redirect shells


def _try_pmc_html(
    pmcid: str,
    html_path: Path,
    txt_path: Path,
    session: requests.Session,
) -> tuple[str | None, str]:
    url = f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/'
    r   = _get_with_retry(url, session)
    if r is None:
        return None, url
    html = r.text
    if _is_paywall_html(html):
        return None, url
    txt = _html_to_txt(html)
    if len(txt.split()) < _MIN_WORDS:   # bot/captcha/empty shell
        return None, url
    html_path.write_text(html, encoding='utf-8')
    if len(txt) > 500:
        txt_path.write_text(txt, encoding='utf-8')
        return 'txt', url
    return 'html', url


def _try_publisher_html(
    pub_url: str,
    html_path: Path,
    txt_path: Path,
    session: requests.Session,
) -> tuple[str | None, str]:
    r = _get_with_retry(pub_url, session)
    if r is None:
        return None, pub_url
    html = r.text
    if _is_paywall_html(html):
        return None, pub_url
    txt = _html_to_txt(html)
    if len(txt.split()) < _MIN_WORDS:   # bot/captcha/empty shell
        return None, pub_url
    html_path.write_text(html, encoding='utf-8')
    if len(txt) > 500:
        txt_path.write_text(txt, encoding='utf-8')
        return 'txt', pub_url
    return 'html', pub_url


# ── per-record orchestration ───────────────────────────────────────────────────

def download_record(rec: dict, out_dir: Path, session: requests.Session) -> dict:
    pub_id  = rec['id']
    title   = rec.get('title_guess') or 'untitled'
    stem    = _safe_stem(pub_id, title)
    pdf_url = rec.get('pdf_url')
    pmcid   = rec.get('pmcid')
    pub_url = rec.get('publisher_url')
    is_oa   = bool(rec.get('is_oa'))

    result: dict = {
        'id':              pub_id,
        'title':           title,
        'doi':             rec.get('doi'),
        'pmcid':           pmcid,
        'download_status': 'metadata_only',
        'download_source': None,
        'filename':        None,
        'url_tried':       None,
        'error':           None,
    }

    # ── 1. direct legal PDF (Unpaywall) ───────────────────────────────────────
    if pdf_url:
        pdf_path = out_dir / f'{stem}.pdf'
        if _try_pdf(pdf_url, pdf_path, session):
            result.update(download_status='pdf', download_source='unpaywall',
                          filename=pdf_path.name, url_tried=pdf_url)
            return result
        result['url_tried'] = pdf_url

    # ── 2. PMC PDF ────────────────────────────────────────────────────────────
    if pmcid:
        pdf_path = out_dir / f'{stem}.pdf'
        ok, tried = _try_pmc_pdf(pmcid, pdf_path, session)
        if ok:
            result.update(download_status='pdf', download_source='pmc',
                          filename=pdf_path.name, url_tried=tried)
            return result

        # ── 3a. PMC HTML / TXT ────────────────────────────────────────────
        html_path = out_dir / f'{stem}.html'
        txt_path  = out_dir / f'{stem}.txt'
        outcome, tried = _try_pmc_html(pmcid, html_path, txt_path, session)
        if outcome:
            fname = txt_path.name if outcome == 'txt' else html_path.name
            result.update(download_status=outcome, download_source='pmc',
                          filename=fname, url_tried=tried)
            return result

    # ── 3b. Open-access publisher HTML / TXT ──────────────────────────────────
    if is_oa and pub_url:
        html_path = out_dir / f'{stem}.html'
        txt_path  = out_dir / f'{stem}.txt'
        outcome, tried = _try_publisher_html(pub_url, html_path, txt_path, session)
        if outcome:
            fname = txt_path.name if outcome == 'txt' else html_path.name
            result.update(download_status=outcome, download_source='publisher',
                          filename=fname, url_tried=tried)
            return result
        result['url_tried'] = result['url_tried'] or tried

    # ── 4. classify outcome ───────────────────────────────────────────────────
    has_sources = any([pdf_url, pmcid, (is_oa and pub_url)])
    if has_sources:
        result['download_status'] = 'failed'
        result['error']           = 'all_sources_exhausted'
    # else: metadata_only (no retrieval source existed)

    return result


# ── resume support ─────────────────────────────────────────────────────────────

_DONE_STATUSES = {'pdf', 'html', 'txt', 'metadata_only'}


def _load_done(report_path: Path) -> dict[int, dict]:
    if not report_path.exists():
        return {}
    try:
        raw = json.loads(report_path.read_text(encoding='utf-8'))
        return {
            r['id']: r
            for r in raw.get('records', [])
            if r.get('download_status') in _DONE_STATUSES
        }
    except Exception:
        return {}


# ── reporting ──────────────────────────────────────────────────────────────────

def _save_report(results: list[dict], total: int) -> None:
    summary = {
        'total':           total,
        'pdf_downloaded':  sum(1 for r in results if r['download_status'] == 'pdf'),
        'html_downloaded': sum(1 for r in results if r['download_status'] == 'html'),
        'txt_extracted':   sum(1 for r in results if r['download_status'] == 'txt'),
        'metadata_only':   sum(1 for r in results if r['download_status'] == 'metadata_only'),
        'failed':          sum(1 for r in results if r['download_status'] == 'failed'),
        'manual_candidates': [
            r['id'] for r in results
            if r['download_status'] in ('failed', 'metadata_only')
        ],
    }
    REPORT_PATH.write_text(
        json.dumps({'summary': summary, 'records': results},
                   indent=2, ensure_ascii=False),
        encoding='utf-8',
    )


def _print_summary(results: list[dict], total: int) -> None:
    pdf   = sum(1 for r in results if r['download_status'] == 'pdf')
    html  = sum(1 for r in results if r['download_status'] == 'html')
    txt   = sum(1 for r in results if r['download_status'] == 'txt')
    meta  = sum(1 for r in results if r['download_status'] == 'metadata_only')
    fail  = sum(1 for r in results if r['download_status'] == 'failed')
    full  = pdf + html + txt
    print(f'\n{"-" * 55}')
    print(f'Download complete  ({len(results)} of {total} processed)')
    print(f'  PDF              : {pdf}')
    print(f'  HTML             : {html}')
    print(f'  TXT (from HTML)  : {txt}')
    print(f'  Full text total  : {full}  ({100*full//max(total,1)}%)')
    print(f'  Metadata only    : {meta}')
    print(f'  Failed           : {fail}')
    print(f'  Report           : {REPORT_PATH}')
    print(f'  Files            : {OUT_DIR}/')


# ── main ───────────────────────────────────────────────────────────────────────

def main(limit: int | None = None) -> None:
    if not INPUT_PATH.exists():
        print(f'ERROR: {INPUT_PATH} not found', file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True)
    pubs  = json.loads(INPUT_PATH.read_text(encoding='utf-8'))
    total = len(pubs)
    if limit:
        pubs = pubs[:limit]

    done    = _load_done(REPORT_PATH)
    results: list[dict] = []
    session = _make_session()

    for i, pub in enumerate(pubs, 1):
        pid   = pub['id']
        label = f'[{i:3d}/{len(pubs):3d}] id={pid:3d}'

        if pid in done:
            results.append(done[pid])
            status = done[pid]['download_status']
            print(f'{label}  skipped ({status})')
            continue

        print(f'{label} ...', end=' ', flush=True)
        rec = download_record(pub, OUT_DIR, session)
        results.append(rec)

        status = rec['download_status']
        src    = rec.get('download_source') or '-'
        fname  = rec.get('filename') or ''
        print(f'{status:<14} src={src:<12} {fname}')

        _save_report(results, total)

    _save_report(results, total)
    _print_summary(results, total)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Download full text for enriched publications.')
    ap.add_argument('--limit', type=int, default=None,
                    help='Process only the first N records (for testing)')
    args = ap.parse_args()
    main(limit=args.limit)
