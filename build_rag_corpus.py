"""
build_rag_corpus.py  —  Phase 5 Step 1
Build RAG corpus from downloaded full-text publications.

Input:
  downloaded_papers/           PDF, TXT, HTML full-text files
  enriched_publications.json   publication metadata

Output:
  rag_corpus.json              chunked corpus ready for vectorization

File selection priority per publication_id:
  TXT  >  PDF  >  HTML

Chunking:
  400-word chunks, 50-word overlap, minimum 50 words per chunk
  chunk_id: pub{id}_c{chunk_index:04d}
"""

import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup

# ── configuration ──────────────────────────────────────────────────────────────

CORPUS_DIR    = Path('downloaded_papers')
ENRICHED_PATH = Path('enriched_publications.json')
OUTPUT_PATH   = Path('rag_corpus.json')

CHUNK_WORDS   = 400
OVERLAP_WORDS = 50
MIN_CHUNK_WORDS = 50

# ── PDF support (optional import) ─────────────────────────────────────────────
# PyMuPDF (fitz) is preferred: handles two-column academic layouts correctly.
# Falls back to pdfplumber if fitz is unavailable.

try:
    import fitz as _fitz          # PyMuPDF
    _PDF_BACKEND = 'fitz'
except ImportError:
    _fitz = None  # type: ignore
    try:
        import pdfplumber as _pdfplumber
        _PDF_BACKEND = 'pdfplumber'
    except ImportError:
        _pdfplumber = None  # type: ignore
        _PDF_BACKEND = None


# ── file gathering ─────────────────────────────────────────────────────────────

def _gather_files(corpus_dir: Path) -> dict[int, dict[str, Path]]:
    """
    Scan corpus_dir and group files by publication_id.
    Returns {pub_id: {'.txt': Path, '.pdf': Path, ...}}
    Dynamically processes whatever files are present — no hardcoded counts.
    """
    by_id: dict[int, dict[str, Path]] = defaultdict(dict)
    for f in sorted(corpus_dir.iterdir()):   # sorted for reproducibility
        if not f.is_file():
            continue
        try:
            pid = int(f.name.split('_')[0])
        except (ValueError, IndexError):
            continue
        ext = f.suffix.lower()
        if ext in ('.txt', '.pdf', '.html'):
            by_id[pid][ext] = f   # later entries overwrite earlier; sorted() makes this stable
    return dict(by_id)


def _pick_file(files: dict[str, Path]) -> tuple[Path, str]:
    """Return (path, file_type) using TXT > PDF > HTML priority."""
    for ext, ftype in [('.txt', 'txt'), ('.pdf', 'pdf'), ('.html', 'html')]:
        if ext in files:
            return files[ext], ftype
    raise ValueError('No usable file')


# ── text extraction ────────────────────────────────────────────────────────────

def _extract_txt(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='replace')


def _extract_pdf(path: Path) -> str:
    if _PDF_BACKEND is None:
        raise RuntimeError('No PDF library found; run: pip install pymupdf')
    if _PDF_BACKEND == 'fitz':
        # PyMuPDF: sort=True reads in natural reading order (column-aware)
        pages: list[str] = []
        doc = _fitz.open(str(path))
        for page in doc:
            t = page.get_text('text', sort=True) or ''
            if t.strip():
                pages.append(t)
        doc.close()
        return '\n'.join(pages)
    else:
        # pdfplumber fallback
        pages = []
        with _pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ''
                if t.strip():
                    pages.append(t)
        return '\n'.join(pages)


def _extract_html(path: Path) -> str:
    html = path.read_text(encoding='utf-8', errors='replace')
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script', 'style', 'nav', 'header', 'footer',
                     'aside', 'figure', 'noscript', 'form']):
        tag.decompose()
    body = (soup.find('article')
            or soup.find(id=re.compile(r'content|article|main', re.I))
            or soup.find('main')
            or soup.body
            or soup)
    lines: list[str] = []
    for el in body.descendants:
        if getattr(el, 'name', None) in ('p', 'h1', 'h2', 'h3', 'h4', 'li'):
            text = el.get_text(' ', strip=True)
            if text:
                lines.append(text)
    return '\n\n'.join(lines)


# ── text cleaning ──────────────────────────────────────────────────────────────

_BOILERPLATE_RE = re.compile(
    r'(?:'
    r'[Cc]opyright\s*[©(C)]+.*?(?:\n|$)'
    r'|[Dd]ownloaded\s+from\b.*?(?:\n|$)'
    r'|[Ff]or\s+personal\s+use\s+only.*?(?:\n|$)'
    r'|[Aa]ll\s+rights\s+reserved.*?(?:\n|$)'
    r'|[Pp]rinted\s+in\s+(?:U\.?S\.?A|Great Britain).*?(?:\n|$)'
    r')',
    re.MULTILINE,
)

_REFERENCES_RE = re.compile(
    r'\n\s*'
    r'(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY|Works\s+Cited|WORKS\s+CITED)'
    r'\s*\n',
)


def _clean(text: str) -> str:
    """Normalize unicode and whitespace; remove known boilerplate patterns."""
    # NFKC normalization: resolves ligatures (fi→fi), fullwidth chars, smart quotes
    text = unicodedata.normalize('NFKC', text)
    # Normalize line endings
    text = re.sub(r'\r\n?', '\n', text)
    # Rejoin PDF hyphenated line breaks: "treat-\nment" → "treatment"
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    # Strip boilerplate
    text = _BOILERPLATE_RE.sub('', text)
    # Collapse runs of blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse inline whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _remove_references(text: str) -> str:
    """Strip the trailing reference list from academic paper text."""
    m = _REFERENCES_RE.search(text)
    # Only strip if the header appears in the second half — avoids false positives
    # in introductions that mention "References [1]"
    if m and m.start() > len(text) * 0.5:
        return text[:m.start()].strip()
    return text


# ── chunking ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    """
    Sliding-window word-based chunker.

    Parameters come from module-level constants (CHUNK_WORDS, OVERLAP_WORDS,
    MIN_CHUNK_WORDS) — change them once to affect all future builds.
    """
    words = text.split()
    if not words:
        return []
    if len(words) < MIN_CHUNK_WORDS:
        return [text]

    step   = CHUNK_WORDS - OVERLAP_WORDS
    chunks: list[str] = []
    start  = 0
    while start < len(words):
        end    = min(start + CHUNK_WORDS, len(words))
        window = words[start:end]
        if len(window) >= MIN_CHUNK_WORDS:
            chunks.append(' '.join(window))
        if end == len(words):
            break
        start += step
    return chunks


# ── corpus builder ─────────────────────────────────────────────────────────────

def build_corpus(
    corpus_dir: Path = CORPUS_DIR,
    enriched_path: Path = ENRICHED_PATH,
    output_path: Path = OUTPUT_PATH,
) -> list[dict]:
    """
    Build and write rag_corpus.json. Returns the list of chunk dicts.
    Safe to re-run: output is always fully regenerated from source files.
    New files in corpus_dir are automatically picked up on the next run.
    """
    for p in (corpus_dir, enriched_path):
        if not p.exists():
            print(f'ERROR: {p} not found', file=sys.stderr)
            sys.exit(1)

    meta_map: dict[int, dict] = {
        r['id']: r
        for r in json.loads(enriched_path.read_text(encoding='utf-8'))
    }

    files_by_id = _gather_files(corpus_dir)
    pub_ids     = sorted(files_by_id)

    all_chunks: list[dict] = []
    processed   = 0
    failures    = 0
    type_counts: dict[str, int] = defaultdict(int)

    print(f'Found {len(pub_ids)} publication IDs in {corpus_dir}/')
    print()

    for pub_id in pub_ids:
        try:
            path, ftype = _pick_file(files_by_id[pub_id])
        except ValueError:
            print(f'  [SKIP] id={pub_id}: no usable file')
            failures += 1
            continue

        # ── extract ───────────────────────────────────────────────────────────
        try:
            if ftype == 'txt':
                raw = _extract_txt(path)
            elif ftype == 'pdf':
                raw = _extract_pdf(path)
            else:
                raw = _extract_html(path)
        except Exception as exc:
            print(f'  [FAIL] id={pub_id} ({path.name[:45]}): {exc}')
            failures += 1
            continue

        # ── clean ─────────────────────────────────────────────────────────────
        text = _clean(raw)
        text = _remove_references(text)

        word_count = len(text.split())
        if word_count < MIN_CHUNK_WORDS:
            print(f'  [SKIP] id={pub_id}: only {word_count} words after cleaning')
            failures += 1
            continue

        # ── chunk ─────────────────────────────────────────────────────────────
        chunk_texts = _chunk_text(text)
        if not chunk_texts:
            print(f'  [SKIP] id={pub_id}: 0 chunks produced')
            failures += 1
            continue

        # ── attach metadata ───────────────────────────────────────────────────
        meta = meta_map.get(pub_id, {})
        for idx, chunk_text in enumerate(chunk_texts):
            all_chunks.append({
                'chunk_id':       f'pub{pub_id}_c{idx:04d}',
                'publication_id': pub_id,
                'title':          (meta.get('title_guess') or '').strip(),
                'authors':        (meta.get('authors_guess') or '').strip(),
                'year':           meta.get('year'),
                'doi':            (meta.get('doi') or '').strip(),
                'pmid':           str(meta.get('pmid') or '').strip(),
                'pmcid':          (meta.get('pmcid') or '').strip(),
                'source_file':    path.name,
                'file_type':      ftype,
                'chunk_text':     chunk_text,
            })

        type_counts[ftype] += 1
        processed += 1
        print(f'  [OK]  id={pub_id:3d}  {ftype:<4}  {len(chunk_texts):3d} chunks  {path.name[:52]}')

    # ── write output ──────────────────────────────────────────────────────────
    output_path.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    # ── summary ───────────────────────────────────────────────────────────────
    W = 58
    print(f'\n{"-"*W}')
    print(f'Corpus -> {output_path}')
    print(f'  Papers processed : {processed}')
    print(f'  Chunks created   : {len(all_chunks)}')
    print(f'  Failures/skipped : {failures}')
    type_str = '  '.join(f'{k}={v}' for k, v in sorted(type_counts.items()))
    print(f'  File types       : {type_str}')
    if all_chunks:
        avg_w = sum(len(c['chunk_text'].split()) for c in all_chunks) // len(all_chunks)
        unique = len({c['publication_id'] for c in all_chunks})
        print(f'  Avg chunk words  : {avg_w}')
        print(f'  Unique pubs      : {unique}')
        output_kb = output_path.stat().st_size / 1024
        print(f'  Output size      : {output_kb:.0f} KB')

    return all_chunks


def main() -> None:
    build_corpus()


if __name__ == '__main__':
    main()
