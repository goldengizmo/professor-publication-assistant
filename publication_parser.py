import json
import re
import sys
from pathlib import Path

import docx

_W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'


def _normalize(text: str) -> str:
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _is_numbered(p) -> bool:
    return p._p.find(f'.//{{{_W}}}numPr') is not None


def parse(docx_path: str) -> list[dict]:
    doc = docx.Document(docx_path)
    entries = []
    current = None

    for p in doc.paragraphs:
        text = _normalize(p.text)
        if not text:
            continue

        if _is_numbered(p):
            if current is not None:
                entries.append(current)
            current = text
        else:
            # Continuation of current entry (e.g. wrapped author list)
            if current is not None:
                current = current + ' ' + text

    if current:
        entries.append(current)

    return [{'id': i + 1, 'raw_text': t} for i, t in enumerate(entries)]


def main() -> None:
    docx_path = Path('documents/master_publications.docx')
    output_path = Path('parsed_publications.json')

    if not docx_path.exists():
        print(f'ERROR: {docx_path} not found', file=sys.stderr)
        sys.exit(1)

    publications = parse(str(docx_path))
    output_path.write_text(
        json.dumps(publications, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(f'Parsed {len(publications)} publications -> {output_path}')


if __name__ == '__main__':
    main()
