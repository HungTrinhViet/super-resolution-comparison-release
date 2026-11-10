"""Read-only DOCX checks of only declared claims, not complete semantic correctness.

CSV/JSON/text sources are verified before document matching. This script only
verifies the claims declared in the map; it cannot prove every statement correct.
Reports are new JSON files; DOCX/PDF files are never written by this script.
"""
import argparse
from collections import Counter
import csv
import math
import hashlib
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def sha256(path: Path) -> str:
    """Hash input bytes without modifying the source."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_document(path: Path) -> dict[str, str]:
    """Read paragraphs and table cells with one-based XML locations."""
    with ZipFile(path, 'r') as archive:
        member = archive.getinfo('word/document.xml')
        if member.file_size > 50_000_000:
            raise ValueError('Document XML exceeds the 50 MB read limit.')
        root = ET.fromstring(archive.read(member))
    paragraphs = {}
    for index, paragraph in enumerate(root.findall('.//w:p', NS), 1):
        paragraphs[f'P{index:04d}'] = ''.join(n.text or '' for n in paragraph.findall('.//w:t', NS))
    for table_index, table in enumerate(root.findall('.//w:tbl', NS), 1):
        for row_index, row in enumerate(table.findall('w:tr', NS), 1):
            for column, cell in enumerate(row.findall('w:tc', NS), 1):
                paragraphs[f'T{table_index}R{row_index}C{column}'] = ''.join(
                    n.text or '' for n in cell.findall('.//w:t', NS))
    return paragraphs


def at_key(value: object, keys: list) -> object:
    """Resolve an explicit JSON key/index path."""
    for key in keys:
        value = value[key]
    return value


def source_value(source: dict, root: Path) -> object:
    """Resolve exactly one source row/key and verify its hash and run identity."""
    path = (root/source['file']).resolve()
    path.relative_to(root.resolve())
    if 'sha256' in source and sha256(path) != source['sha256']:
        raise ValueError('Source SHA-256 mismatch: '+source['file'])
    if source['format'] == 'csv':
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        rows = [row for row in rows if all(row.get(k) == v for k, v in source['match'].items())]
        if len(rows) != 1:
            raise ValueError('Source must match exactly one CSV row.')
        value = rows[0]
        if 'expected_run_id' in source and value[source.get('run_id_key', 'source_run_id')] != source['expected_run_id']:
            raise ValueError('Source run_id mismatch.')
        return value[source['key']]
    if source['format'] == 'json':
        value = json.loads(path.read_text(encoding='utf-8'))
        if 'expected_run_id' in source and at_key(value, source['run_id_key']) != source['expected_run_id']:
            raise ValueError('Source run_id mismatch.')
        return at_key(value, source['key'])
    if source['format'] == 'text':
        text = path.read_text(encoding='utf-8')
        if source['key'] not in text:
            raise ValueError('Declared text evidence is missing.')
        return source['key']
    raise ValueError('Unsupported source format.')


def same_value(actual: object, expected: object) -> bool:
    """Compare parsed binary floats from .17g CSV and JSON, otherwise text."""
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        number = float(str(actual).replace(',', '.'))
        return math.isfinite(number) and number == expected
    return actual == expected


def display_value(expected: object, rounding: dict) -> str:
    """Use the map's fixed display precision; never round the source file."""
    return (f'{float(expected):.{rounding["digits"]}f}' if 'digits' in rounding else str(expected))


def check_document(path: Path, mapping: dict, root: Path = ROOT) -> dict:
    """Report FOUND_MATCH/MISMATCH/NOT_FOUND for declared claims only."""
    before = sha256(path)
    document = extract_document(path)
    results = []
    seen = set()
    for claim in mapping['claims']:
        if claim['id'] in seen:
            raise ValueError('Duplicate claim id: '+claim['id'])
        seen.add(claim['id'])
        locator = claim['locator']
        if 'table' in locator:
            key = f'T{locator["table"]}R{locator["row"]}C{locator["column"]}'
            candidates = [(key, document[key])] if key in document else []
        elif 'paragraph' in locator:
            key = f'P{locator["paragraph"]:04d}'
            candidates = [(key, document[key])] if key in document else []
        else:
            candidates = [(key, text) for key, text in document.items()
                          if key.startswith('P') and locator['find'] in text]
        result = {'id': claim['id'], 'expected': claim['expected'],
                  'source': claim['source'], 'locations': [key for key, _ in candidates]}
        try:
            value = source_value(claim['source'], root)
            if not same_value(value, claim['expected']):
                raise ValueError('Declared expectation differs from source.')
            expected = display_value(claim['expected'], claim.get('rounding', {}))
            result['expected_display'] = expected
            if not candidates:
                result['status'] = 'NOT_FOUND'
            else:
                matches = []
                for _, text in candidates:
                    pattern = locator.get('value_pattern')
                    match = re.search(pattern, text) if pattern else None
                    if pattern:
                        actual = match.group('value') if match else None
                    else:
                        actual = text.strip()
                    result.setdefault('observed', []).append(actual)
                    matches.append(actual is not None and actual.replace(',', '.') == expected.replace(',', '.'))
                result['status'] = 'FOUND_MATCH' if all(matches) else 'MISMATCH'
        except (ValueError, KeyError, IndexError, OSError, TypeError) as error:
            result.update(status='MISMATCH', error=str(error))
        results.append(result)
    if sha256(path) != before:
        raise ValueError('DOCX changed during read-only verification.')
    return {'scope': 'Checks only declared claims; not complete semantic verification.',
            'docx_sha256_before': before, 'docx_sha256_after': sha256(path),
            'claims': results, 'counts': dict(Counter(row['status'] for row in results))}


def save_report(path: Path, report: dict) -> None:
    """Write a new JSON report, never DOCX/PDF or an existing file."""
    if path.suffix.lower() != '.json':
        raise ValueError('Report output must be a new .json file.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main() -> None:
    """Inspect an original DOCX without writing or copying it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docx', required=True, type=Path)
    parser.add_argument('--claims', type=Path, default=ROOT/'docs/report/claims_map.json')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'logs/m9_report_check.json')
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT)
    if args.output.exists():
        raise FileExistsError(args.output)
    mapping = json.loads(args.claims.read_text(encoding='utf-8'))
    report = check_document(args.docx, mapping, args.root)
    save_report(args.output, report)
    print(json.dumps({'scope': report['scope'], 'counts': report['counts'],
                      'word_unchanged': report['docx_sha256_before'] == report['docx_sha256_after'],
                      'report': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
