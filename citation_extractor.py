import json
import re
import sys
from pathlib import Path


# ── identifier extraction ─────────────────────────────────────────────────────

def _extract_doi(text: str) -> str | None:
    # Covers all observed variants:
    #   doi: 10.x   doi:10.x   DOI: 10.x   DOI 10.x
    #   DOI: https://doi.org/10.x   https://doi.org/10.x
    #   https://doi.org/ 10.x (space)   doi/org/10.x (typo)
    m = re.search(
        r'(?:'
        r'(?:doi|DOI)\s*:?\s*(?:https?://doi\.org/\s*)?'
        r'|https?://doi\.org/\s*'
        r'|doi/org/'
        r')'
        r'(10\.\d{4,}/\S+)',
        text,
    )
    return m.group(1).rstrip('.,;)') if m else None


def _extract_pmid(text: str) -> str | None:
    m = re.search(r'PMID:\s*(\d+)', text)
    return m.group(1) if m else None


def _extract_pmcid(text: str) -> str | None:
    m = re.search(r'PMCID:\s*(PMC\d+)', text)
    return m.group(1).rstrip('.,;') if m else None


def _extract_year(text: str) -> int | None:
    # Strip URLs/DOI strings first so years embedded in DOI paths don't match
    clean = re.sub(r'https?://\S+', '', text)
    clean = re.sub(r'(?:doi|DOI)\s*:?\s*\S+', '', clean)
    # Priority 1: year adjacent to volume/page separator — most reliable as publication year
    # e.g. "Journal, 2016; 20(3)" or "Journal YYYY;vol" or "Journal YYYY, vol"
    m = re.search(r'\b((19|20)\d{2})\b\s*[;,:]\s*\d', clean)
    if m:
        return int(m.group(1))
    # Priority 2: first year anywhere in text
    m = re.search(r'\b((19|20)\d{2})\b', clean)
    return int(m.group(1)) if m else None


# ── author / title boundary detection ────────────────────────────────────────

_TEAM_RE = re.compile(
    r'\b(?:team|group|consortium|collaboration|investigators|committee|network|members)\b',
    re.IGNORECASE,
)


def _is_final_author_token(token: str) -> bool:
    t = token.strip().rstrip('.')
    if not t:
        return False
    # Bare initials: "TE", "MG", "A"
    if re.match(r'^[A-Z]{1,4}$', t):
        return True
    # Dotted initials: "P.G", "T.E", "L.A.R"
    if re.match(r'^[A-Z](?:\.[A-Z])+$', t):
        return True
    # LastName(s) + initials — handles simple, compound, hyphenated:
    # "Fowler MG", "Abdool Karim SS", "Nielsen-Saines K", "De Vincenzi I"
    if re.match(r'^[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*\s+[A-Z]{1,4}$', t):
        return True
    # Single surname: "Mofenson"
    if re.match(r'^[A-Z][a-z\-]+$', t):
        return True
    # Inverted: "Taha, Taha" or "Fowler, Mary Glenn"
    if re.match(r'^[A-Z][a-z\-]+,\s+[A-Z][a-z ]+$', t):
        return True
    # APA inverted with initial dot: "Taha, T. E" or "Fowler, M. G"
    if re.match(r'^[A-Z][a-z\-]+,\s+[A-Z]', t):
        return True
    # APA ampersand last author: "& Stranix-Chibanda, L"
    if re.match(r'^&\s+[A-Z]', t):
        return True
    # Full first+last (no initials): "Nishi Suryavanshi", "Mary Glenn Fowler"
    if re.match(r'^[A-Z][a-z\-]+(?:\s+[A-Z][a-z\-]+)+$', t):
        return True
    # Full first + middle initial + last: "Ahmed M. Ahmed", "Taha E. Taha", "Taha E Taha"
    if re.match(r'^[A-Z][a-z]+\s+[A-Z]\.?\s+[A-Z][a-z]+$', t):
        return True
    # et al  (must come before particle stripping — "et" is a particle)
    if re.match(r'^et\s+al\.?$', t, re.IGNORECASE):
        return True
    # Study team / group / consortium name
    if _TEAM_RE.search(t):
        return True
    # Dutch/Flemish/German/French particles: "van der Horst C", "van Rompay K"
    # Only recurse when the stripped remainder starts with uppercase (real name follows)
    particle_stripped = re.sub(r'^(?:[a-z]{1,4}\s+)+', '', t)
    if particle_stripped and particle_stripped != t and particle_stripped[0].isupper():
        return _is_final_author_token(particle_stripped)
    return False


def _looks_like_title_start(text: str) -> bool:
    """True when text reads like the start of a title sentence, not more authors."""
    text = text.strip()
    words = text.split()
    if len(words) < 3:
        return False
    # "Brown, for the MTN-025/HOPE Study Team" → consortium attribution, still authors
    if re.match(r'^[A-Z][a-z\-]+(,|;)\s+(?:for|on\s+behalf\s+of)\b', text, re.IGNORECASE):
        return False
    # "Ahmed. COVID-19 ..." → single surname followed by period → still in author names
    # Use a strict pattern: only a plain word like "Ahmed." — not "(2024)." or "COVID-19."
    if re.match(r'^[A-Z][a-z]+\.$', words[0]):
        return False
    # "LastName, CapWord..." → still in author list
    # "Noun, lowercase..." → mid-sentence comma in a title phrase, fall through
    m = re.match(r'^[A-Z][a-z\-]+(,|;)\s*', text[:30])
    if m:
        rest_after = text[m.end():]
        if not rest_after or rest_after[0].isupper():
            return False
    # Has a lowercase word among the first five → sentence structure
    if any(w[0].islower() for w in words[:5] if w):
        return True
    # Colon or question mark → common title feature
    if ':' in text[:150] or '?' in text[:150]:
        return True
    return False


def _find_author_boundary(raw: str) -> tuple[int, int]:
    """Return (period_pos, title_start_pos) or (-1, -1)."""
    for m in re.finditer(r'\.\s*(?=[A-Z(])', raw):
        pos = m.start()
        after = raw[m.end():]
        before = raw[:pos]

        # Skip: after starts with a single-letter initial — still in name list
        if re.match(r'^[A-Z][.,]', after[:2]):
            continue

        # Skip: after does not look like a title sentence
        if not _looks_like_title_start(after):
            continue

        last_token = re.split(r'[,;]\s*', before)[-1].strip()
        # Handle "X & Y" or "X and Y" → treat Y as the final author token
        m_and = re.search(r'\s+(?:and|&)\s+', last_token)
        if m_and:
            last_token = last_token[m_and.end():].strip()
        if _is_final_author_token(last_token):
            return pos, m.end()

    return -1, -1


# ── title / journal extraction ────────────────────────────────────────────────

def _strip_identifiers(text: str) -> str:
    """Remove DOI, URL, PMID, PMCID, and Epub notices."""
    # URLs (including "https://doi.org/ 10.xxx" with a space)
    text = re.sub(r'\s*https?://[^\s]*(?:\s+10\.\S+)?', '', text)
    # Bare doi: markers
    text = re.sub(r'\s*(?:doi|DOI)\s*:?\s*\S+', '', text)
    text = re.sub(r'\s*PMID:\s*\d+[.,;]?', '', text)
    text = re.sub(r'\s*PMCID:\s*PMC\d+[.,;]?', '', text)
    text = re.sub(r'[.,;]?\s*Epub\b[^.]*\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(
        r'[.,;]?\s*(?:ahead of print|Publish Ahead of Print)[^.]*\.?',
        '', text, flags=re.IGNORECASE,
    )
    return text.strip().rstrip('.,;').strip()


def _extract_journal_name(segment: str) -> str:
    """From 'Journal Name YYYY;vol...' return only the journal name."""
    # Split at year boundary (with preceding separator)
    j = re.split(r'[\s,;.]\s*(?:19|20)\d{2}', segment)[0]
    # Drop trailing volume numbers: " 22, 634" or "(2022)"
    j = re.sub(r'\s+\d[\d,\s(].*$', '', j)
    return j.strip().rstrip('.,;').strip()


_CITE_RE = re.compile(r'(?:19|20)\d{2}\s*[;,]\s*\d')


def _split_title_journal(rest: str) -> tuple[str, str]:
    """Return (title_guess, journal_guess) from post-author, identifier-stripped text."""
    # APA year-before-title: "(2022). Title..." → strip the year prefix
    rest = re.sub(r'^\(\s*(?:19|20)\d{2}\s*\)\.\s*', '', rest)
    # APA article ID artifact: "[4887202]. " → strip
    rest = re.sub(r'^\[\d+\]\.\s*', '', rest)

    parts = rest.split('. ')
    if len(parts) >= 2:
        # Citation anchor: the part containing year+volume is the journal segment
        cite_idx = next(
            (i for i, p in enumerate(parts) if _CITE_RE.search(p)),
            None,
        )
        if cite_idx is not None and cite_idx >= 1:
            title = '. '.join(parts[:cite_idx]).strip().rstrip(',')
            journal = _extract_journal_name(parts[cite_idx])
            return title, journal

        # Fallback: merge lowercase-starting parts (e.g. "E. coli")
        i = 1
        while i < len(parts) and parts[i] and parts[i][0].islower():
            i += 1
        title = '. '.join(parts[:i]).strip().rstrip(',')
        journal = _extract_journal_name(parts[i]) if i < len(parts) else ''
        return title, journal

    # Fallback: no '. ' — try comma-based split (entry [17] style)
    parts = [p.strip() for p in rest.split(',')]
    for i, part in enumerate(parts):
        if i > 0 and re.match(r'^[A-Z]', part) and 1 <= len(part.split()) <= 6:
            return ', '.join(parts[:i]).strip(), part.strip()

    # Last resort: hyphen-separated title+journal ("HIV-New England Journal of Medicine")
    m_yr = re.search(r'(?:19|20)\d{2}', rest)
    if m_yr:
        pre = rest[:m_yr.start()].rstrip(' ,')
        splits = [m.start() for m in re.finditer(r'-([A-Z][a-z])', pre)]
        if splits:
            split_pos = splits[-1]
            return pre[:split_pos].strip(), pre[split_pos + 1:].strip()

    return rest.strip(), ''


# ── main extraction ───────────────────────────────────────────────────────────

def extract_metadata(pub: dict) -> dict:
    raw = pub['raw_text']

    doi   = _extract_doi(raw)
    pmid  = _extract_pmid(raw)
    pmcid = _extract_pmcid(raw)
    year  = _extract_year(raw)

    boundary, title_start = _find_author_boundary(raw)

    if boundary >= 0:
        authors_guess = raw[:boundary].strip()
        rest_raw = raw[title_start:].strip()
    else:
        authors_guess = ''
        rest_raw = raw
        # Fallback: single-author comma boundary "Taha TE, Climate change..."
        # Only fires when the main period-based search already failed.
        m_cb = re.match(r'^([A-Z][a-z\-]+ [A-Z]{1,4}),\s+(.*)', raw, re.DOTALL)
        if m_cb and _looks_like_title_start(m_cb.group(2)):
            authors_guess = m_cb.group(1)
            rest_raw = m_cb.group(2)

    rest_clean = _strip_identifiers(rest_raw)
    title_guess, journal_guess = _split_title_journal(rest_clean)

    return {
        'id':            pub['id'],
        'raw_text':      raw,
        'source_number': pub['id'],
        'title_guess':   title_guess   or None,
        'authors_guess': authors_guess or None,
        'journal_guess': journal_guess or None,
        'year':          year,
        'doi':           doi,
        'pmid':          pmid,
        'pmcid':         pmcid,
    }


def main() -> None:
    input_path  = Path('parsed_publications.json')
    output_path = Path('publication_metadata.json')

    if not input_path.exists():
        print(f'ERROR: {input_path} not found', file=sys.stderr)
        sys.exit(1)

    pubs    = json.loads(input_path.read_text(encoding='utf-8'))
    results = [extract_metadata(p) for p in pubs]

    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    doi_count   = sum(1 for r in results if r['doi'])
    pmid_count  = sum(1 for r in results if r['pmid'])
    year_count  = sum(1 for r in results if r['year'])
    title_count = sum(1 for r in results if r['title_guess'])

    print(
        f'Extracted {len(results)} records -> {output_path}\n'
        f'  title: {title_count}  doi: {doi_count}  '
        f'pmid: {pmid_count}  year: {year_count}'
    )


if __name__ == '__main__':
    main()
