"""Exploratory low-gradient, mid-luminance crops; never replace official v1."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .comparison import SOURCES, load_runs, verify_source_snapshot
from .comparison_crops import (
    digest, integer_y, window_sums, read_rgb, render_panel, save_json,
)

ROOT = Path(__file__).resolve().parents[2]
RULE = 'hr_lowgrad64_v2'
VERSION = 'low_gradient_mid_luminance_integer_v2'
V1_MANIFEST = 'docs/comparison/crop_manifest.json'
V1_SHA256 = '0f37d1901fae432e896aba6e4b8470a84cb9bc174a570a8fe9a14e08fb8e1b22'
CODE = ('src/sr_compare/comparison_crops_v2.py', 'scripts/build_exploratory_crops_v2.py',
        'src/sr_compare/comparison_crops.py', 'src/sr_compare/comparison.py')


def select_lowgrad_mid(rgb: np.ndarray) -> dict:
    """Minimize HR gradient with exact std and inclusive mean constraints."""
    y = integer_y(rgb)
    if min(y.shape) < 64:
        raise ValueError('HR must be at least 64x64.')
    scores = (window_sums(np.diff(y, axis=1)**2, 64, 63)
              + window_sums(np.diff(y, axis=0)**2, 63, 64))
    sums, squares = window_sums(y, 64, 64), window_sums(y*y, 64, 64)
    variance = 4096*squares-sums*sums
    eligible = ((variance >= 4*4096**2)
                & (sums >= 60*4096) & (sums <= 200*4096))
    if not eligible.any():
        raise ValueError('No eligible low-gradient, mid-luminance crop; no fallback.')
    row, column = np.unravel_index(np.where(
        eligible, scores, np.iinfo(np.int64).max).argmin(), scores.shape)
    return {'rule': RULE, 'rule_version': 2, 'algorithm_version': VERSION,
            'exploratory': True, 'x': int(column), 'y': int(row),
            'w': 64, 'h': 64, 'enlargement': 4,
            'gradient_score': int(scores[row, column]),
            'sum_y': int(sums[row, column]), 'sum_y2': int(squares[row, column]),
            'variance_numerator': int(variance[row, column]),
            'population_n': 4096, 'std_threshold_numerator': 4*4096**2,
            'mean_min_sum': 60*4096, 'mean_max_sum': 200*4096,
            'mean_y': int(sums[row, column])/4096,
            'tie_break': 'smallest_y_then_x'}


def annotate_v1(record: dict, previous: list[dict]) -> dict:
    """Record coordinate coincidences after selection, without selecting again."""
    matches = [row['rule'] for row in previous
               if (row['x'], row['y']) == (record['x'], record['y'])]
    return {**record, 'matches_v1': matches,
            'duplicate_coordinates_with_v1': bool(matches)}


def lock_v2(manifest: Path, receipt: Path) -> dict:
    """Select from HR only, then annotate against v1 and lock a separate manifest."""
    if manifest.exists() or receipt.exists():
        raise FileExistsError('Exploratory manifest or receipt already exists.')
    records, hr_hashes = [], {}
    for index in range(1, 6):
        name = f'img_{index:03d}_SRF_4'
        relative = f'data/set5/image_SRF_4/{name}_HR.png'
        pixels = read_rgb(ROOT/relative)
        selected = select_lowgrad_mid(pixels)
        hr_hashes[relative] = digest(ROOT/relative)
        records.append({**selected, 'image_id': name, 'hr_path': relative,
                        'hr_sha256': hr_hashes[relative],
                        'hr_width': pixels.shape[1], 'hr_height': pixels.shape[0]})
    if digest(ROOT/V1_MANIFEST) != V1_SHA256:
        raise ValueError('Official v1 manifest changed.')
    previous = json.loads((ROOT/V1_MANIFEST).read_text(encoding='utf-8'))['records']
    records = [annotate_v1(record, [r for r in previous if r['image_id'] == record['image_id']])
               for record in records]
    locked = datetime.now(timezone.utc).isoformat()
    data = {'schema_version': 1, 'exploratory': True, 'rule': RULE,
            'description': 'low-gradient, mid-luminance crop',
            'algorithm_version': VERSION, 'locked_utc': locked,
            'proposed_after_viewing_v1': True, 'not_confirmatory': True,
            'v1_manifest_sha256': V1_SHA256,
            'records': [{**record, 'locked_utc': locked} for record in records]}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    save_json(manifest, data)
    capture = {'exploratory': True, 'locked_utc': locked,
               'manifest_sha256': digest(manifest), 'v1_manifest_sha256': V1_SHA256,
               'source_sha256': {name: digest(ROOT/name) for name in CODE},
               'hr_sha256': hr_hashes}
    save_json(receipt, capture)
    return capture


def verify_v2(manifest: Path, receipt: Path) -> dict:
    """Verify the independent v2 lock, all dependencies and unchanged official v1."""
    capture = json.loads(receipt.read_text(encoding='utf-8'))
    if digest(manifest) != capture['manifest_sha256'] or digest(ROOT/V1_MANIFEST) != V1_SHA256:
        raise ValueError('Locked v2 or official v1 manifest changed.')
    if set(capture['source_sha256']) != set(CODE):
        raise ValueError('Incomplete v2 source receipt.')
    for name, expected in {**capture['source_sha256'], **capture['hr_sha256']}.items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Locked source/HR changed: {name}')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if (not data['exploratory'] or len(data['records']) != 5
            or data['locked_utc'] != capture['locked_utc']):
        raise ValueError('Invalid exploratory manifest.')
    if {r['image_id'] for r in data['records']} != {f'img_{i:03d}_SRF_4' for i in range(1, 6)}:
        raise ValueError('Expected exactly five distinct HR crops.')
    for row in data['records']:
        if (row['rule'] != RULE or not row['exploratory'] or row['w'] != 64 or row['h'] != 64
                or row['enlargement'] != 4 or row['x'] < 0 or row['y'] < 0
                or row['x']+64 > row['hr_width'] or row['y']+64 > row['hr_height']
                or not 60*4096 <= row['sum_y'] <= 200*4096
                or row['variance_numerator'] < 4*4096**2):
            raise ValueError('Invalid locked exploratory crop fields.')
    return data


def render_v2(manifest: Path, receipt: Path, output: Path) -> dict:
    """Render five separately labeled exploratory sheets without touching v1."""
    data = verify_v2(manifest, receipt)
    load_runs(ROOT)
    verify_source_snapshot(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    if datetime.fromisoformat(started) <= datetime.fromisoformat(data['locked_utc']):
        raise ValueError('Exploratory sheets must follow the v2 lock.')
    hashes = {}
    for record in data['records']:
        name = record['image_id']
        hr = read_rgb(ROOT/record['hr_path'])
        images = [('HR', 'SelfExSR HR '+record['hr_sha256'][:12], hr)]
        for algorithm, run, _ in SOURCES:
            pixels = read_rgb(ROOT/'results'/run/f'{name}_SR.png')
            if pixels.shape != hr.shape:
                raise ValueError('Reference pair shape mismatch in exploratory sheet.')
            images.append((algorithm, run, pixels))
        display = {**record, 'rule': 'EXPLORATORY v2'}
        canvas = render_panel(images, name, display)
        filename = f'crop_{name}_{RULE}.png'
        canvas.save(output/filename, compress_level=9, optimize=False)
        hashes[filename] = {'sha256': digest(output/filename),
                            'pixel_sha256': hashlib.sha256(canvas.tobytes()).hexdigest(),
                            'shape': [canvas.height, canvas.width, 3], 'dtype': 'uint8'}
    verify_source_snapshot(ROOT)
    verify_v2(manifest, receipt)
    return {'exploratory': True, 'render_started_utc': started,
            'manifest_sha256': digest(manifest), 'images': hashes}
