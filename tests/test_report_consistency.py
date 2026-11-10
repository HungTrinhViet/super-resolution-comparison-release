"""Read-only report claim checks on synthetic DOCX archives."""
import importlib.util
import ast
import json
from pathlib import Path
from xml.sax.saxutils import escape
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pytest

SPEC = importlib.util.spec_from_file_location(
    'report_checker', Path(__file__).parents[1]/'scripts/check_report_consistency.py')


def checker():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


def synthetic(tmp_path, text, expected=32.12346, digits=4):
    source = tmp_path/'source.json'
    source.write_text(json.dumps({'score': expected, 'run_id': 'locked_run'}))
    claim = {'id': 'example', 'expected': expected,
             'source': {'file': 'source.json', 'format': 'json', 'key': ['score'],
                        'run_id_key': ['run_id'], 'expected_run_id': 'locked_run'},
             'rounding': {'digits': digits},
             'locator': {'find': 'PSNR Y:', 'value_pattern': r'PSNR Y:\s*(?P<value>[\d.,+-]+)'}}
    path = tmp_path/'example.docx'
    with ZipFile(path, 'w') as archive:
        archive.writestr('word/document.xml',
                        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                        '<w:body><w:p><w:r><w:t>'+escape(text)+'</w:t></w:r></w:p></w:body></w:document>')
    return path, {'schema_version': 1, 'claims': [claim]}


@pytest.mark.parametrize('text,status', [('PSNR Y: 32.1235', 'FOUND_MATCH'),
                                        ('PSNR Y: 31.1235', 'MISMATCH'),
                                        ('Chapter missing', 'NOT_FOUND')])
def test_claim_status_and_word_unchanged(tmp_path, text, status):
    path, claims = synthetic(tmp_path, text)
    original = path.read_bytes()
    report = checker().check_document(path, claims, tmp_path)
    assert report['claims'][0]['status'] == status
    assert report['claims'][0]['locations'] == ([] if status == 'NOT_FOUND' else ['P0001'])
    assert path.read_bytes() == original
    assert 'only declared claims' in report['scope']


def test_rounding_not_just_numeric_closeness(tmp_path):
    path, claims = synthetic(tmp_path, 'PSNR Y: 32.1234')
    assert checker().check_document(path, claims, tmp_path)['claims'][0]['status'] == 'MISMATCH'


def test_source_value_and_run_id_mismatch(tmp_path):
    path, claims = synthetic(tmp_path, 'PSNR Y: 32.1235')
    claims['claims'][0]['source']['expected_run_id'] = 'other_run'
    assert checker().check_document(path, claims, tmp_path)['claims'][0]['status'] == 'MISMATCH'


def test_source_hash_and_expected_mismatch(tmp_path):
    path, claims = synthetic(tmp_path, 'PSNR Y: 32.1235')
    claims['claims'][0]['source']['sha256'] = '0'*64
    assert checker().check_document(path, claims, tmp_path)['claims'][0]['status'] == 'MISMATCH'


def test_invalid_numeric_source_does_not_abort_remaining_claims(tmp_path):
    path, claims = synthetic(tmp_path, 'PSNR Y: 32.1235')
    (tmp_path/'source.json').write_text(json.dumps({'score': None, 'run_id': 'locked_run'}))
    claims['claims'].append({**claims['claims'][0], 'id': 'next_claim'})
    report = checker().check_document(path, claims, tmp_path)
    assert report['counts'] == {'MISMATCH': 2}
    assert len(report['claims']) == 2


def test_csv_full_precision_matches_same_binary_float(tmp_path):
    path, claims = synthetic(tmp_path, 'PSNR Y: 0.8113', expected=0.8112624637779332)
    (tmp_path/'source.json').write_text('{"score": 0.81126246377793321, "run_id": "locked_run"}')
    assert checker().check_document(path, claims, tmp_path)['claims'][0]['status'] == 'FOUND_MATCH'
    (tmp_path/'score.csv').write_text('algorithm,score\nBicubic,0.81126246377793321\n')
    claims['claims'][0]['source'] = {'file': 'score.csv', 'format': 'csv',
                                     'match': {'algorithm': 'Bicubic'}, 'key': 'score'}
    assert checker().check_document(path, claims, tmp_path)['claims'][0]['status'] == 'FOUND_MATCH'


def test_report_no_overwrite_or_word_output(tmp_path):
    module = checker()
    output = tmp_path/'report.json'
    module.save_report(output, {'test': True})
    with pytest.raises(FileExistsError):
        module.save_report(output, {})
    with pytest.raises(ValueError):
        module.save_report(tmp_path/'forbidden.docx', {})


def test_csv_and_table_cell_locator(tmp_path):
    module = checker()
    (tmp_path/'scores.csv').write_text('algorithm,run_id,score\nEDSR,locked_run,32.12346\n')
    path = tmp_path/'table.docx'
    with ZipFile(path, 'w') as archive:
        archive.writestr('word/document.xml',
                        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                        '<w:body><w:tbl><w:tr><w:tc><w:p><w:r><w:t>32.1235</w:t></w:r>'
                        '</w:p></w:tc></w:tr></w:tbl></w:body></w:document>')
    claim = {'id': 'table', 'expected': 32.12346, 'rounding': {'digits': 4},
             'source': {'file': 'scores.csv', 'format': 'csv', 'key': 'score',
                        'match': {'algorithm': 'EDSR'}, 'expected_run_id': 'locked_run',
                        'run_id_key': 'run_id'},
             'locator': {'table': 1, 'row': 1, 'column': 1}}
    report = module.check_document(path, {'claims': [claim]}, tmp_path)
    assert report['claims'][0]['status'] == 'FOUND_MATCH'
    assert report['claims'][0]['locations'] == ['T1R1C1']


def test_flowchart_function_references_and_svg_xml():
    root = Path(__file__).parents[1]
    for index in range(1, 7):
        source = root/f'docs/report/flowcharts/L{index}.mmd'
        blocks = [json.loads(line.split('M9_NODE ', 1)[1])
                  for line in source.read_text(encoding='utf-8').splitlines()
                  if '%% M9_NODE ' in line]
        assert blocks
        for block in blocks:
            assert block['refs'], block['id']
            for reference in block['refs']:
                module, name = reference.split(':')
                file = root/((module.replace('sr_compare.', 'src.sr_compare.')).replace('.', '/')+'.py')
                tree = ast.parse(file.read_text(encoding='utf-8'))
                nodes = tree.body
                for part in name.split('.'):
                    found = [node for node in nodes if getattr(node, 'name', None) == part]
                    assert len(found) == 1, reference
                    nodes = found[0].body
        assert ET.parse(root/f'docs/report/flowcharts/L{index}.svg').getroot().tag.endswith('svg')


def test_documented_runner_preparation_order():
    root = Path(__file__).parents[1]
    for algorithm in ('srcnn', 'fsrcnn', 'edsr', 'esrgan'):
        tree = ast.parse((root/f'scripts/run_{algorithm}.py').read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        calls = {name: min(node.lineno for node in ast.walk(main)
                           if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                           and node.func.id == name)
                 for name in ('verify_lock', 'load_pair', 'load_'+algorithm, 'environment', 'run_pairs')}
        assert list(calls.values()) == sorted(calls.values())
