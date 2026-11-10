"""Read approved result artifacts and publish numeric comparisons without inference."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SOURCES = (
    ('Bicubic', 'set5_numpy_x4_20261108_1627', '798f7a67c4dc5c7214a4fb3b43718f1061e41a293bdb1296f92a752ebb33403f'),
    ('SRCNN', 'set5_srcnn955_x4_20261108_201641', '43ad15c4dc1612d69a0f0e29019abbf2c590aab0d5392a2613ff998e8876a5c2'),
    ('FSRCNN', 'set5_fsrcnn56124_x4_20261109_043641', 'f746f4b0f3998935014bfc6025bd88793d3baf4603455e0d9a3b1877faaaa40b'),
    ('EDSR', 'set5_edsr_baseline_x4_20261109_034132', 'c6cc92658ad2c83851a09832212d91e0c39e7b50f2ec26ce8b3abef7a646dc87'),
    ('ESRGAN', 'set5_esrgan_x4_20261109_060235', 'c4c616ff5a34ece81c696cc3f397b33843902de963ad80fcba01f103fb27633f'),
)
METRICS = ('y_psnr', 'y_ssim', 'rgb_psnr', 'rgb_ssim')
NAMES = tuple(f'img_{i:03d}_SRF_4' for i in range(1, 6))


def sha256(path: Path) -> str:
    """Hash immutable source bytes without decoding or rewriting."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timing_record(timing: dict, repeats: int) -> dict:
    """Validate original samples and derive asymmetric quartile endpoints."""
    samples = np.asarray(timing['samples_ms'], dtype=np.float64)
    if (timing['warmup'] != 3 or timing['repeats'] != repeats
            or samples.shape != (repeats,) or not np.isfinite(samples).all()
            or np.any(samples <= 0)):
        raise ValueError('Invalid timing counts or samples.')
    q25, q75 = np.percentile(samples, [25, 75], method='linear')
    median = float(np.median(samples))
    if (abs(median-timing['median_ms']) > 1e-9
            or abs(q75-q25-timing['iqr_ms']) > 1e-9):
        raise ValueError('Stored timing summary differs from raw samples.')
    return {'median_ms': median, 'iqr_ms': float(q75-q25),
            'q25_ms': float(q25), 'q75_ms': float(q75),
            'warmup': 3, 'repeats': repeats}


def load_runs(root: Path = Path('.'), sources: tuple = SOURCES) -> list[dict]:
    """Require exact approved bytes and all five records in each source run."""
    rows = []
    try:
        for algorithm, run_id, digest in sources:
            path = root/'results'/run_id/'metrics.json'
            if sha256(path) != digest:
                raise ValueError(f'Source JSON SHA-256 mismatch: {run_id}')
            data = json.loads(path.read_text(encoding='utf-8'))
            if data['scale'] != 4:
                raise ValueError('Expected scale4.')
            if algorithm == 'Bicubic':
                if data['backend'] != 'numpy':
                    raise ValueError('Expected the approved NumPy baseline.')
            elif data['run_id'] != run_id:
                raise ValueError('Source run_id mismatch.')
            images = data['images']
            if len(images) != 5 or {r['name'] for r in images} != set(NAMES):
                raise ValueError('Expected exactly five unique Set5 records.')
            for image in sorted(images, key=lambda r: r['name']):
                row = {'algorithm': algorithm, 'image_id': image['name'],
                       'source_run_id': run_id, 'source_json_sha256': digest}
                for space in ('y', 'rgb'):
                    for metric in ('psnr', 'ssim'):
                        value = float(image[space][metric])
                        if not math.isfinite(value) or (metric == 'ssim' and not 0 <= value <= 1):
                            raise ValueError('Missing or invalid reference metric.')
                        row[f'{space}_{metric}'] = value
                row['pipeline'] = timing_record(
                    image['timing'] if algorithm == 'Bicubic' else image['timing_pipeline'],
                    30 if algorithm == 'Bicubic' else 10)
                row['network'] = None if algorithm == 'Bicubic' else timing_record(image['timing_forward'], 10)
                rows.append(row)
            for key in METRICS:
                space, metric = key.split('_')
                actual = math.fsum(r[key] for r in rows if r['algorithm'] == algorithm)/5
                if abs(actual-data['mean_per_image'][space][metric]) > 1e-9:
                    raise ValueError('Stored mean differs from full precision records.')
    except (KeyError, TypeError) as error:
        raise ValueError('Incomplete source schema.') from error
    if len(rows) != 25:
        raise ValueError('Expected five algorithms times five images.')
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Average per-image floats before applying any display rounding."""
    return [{'algorithm': algorithm, 'n_images': 5,
             **{key: math.fsum(r[key] for r in rows if r['algorithm'] == algorithm)/5
                for key in METRICS}}
            for algorithm in dict.fromkeys(r['algorithm'] for r in rows)]


def verify_source_snapshot(root: Path = Path('.')) -> dict:
    """Reject any source bytes changed since the separate source-snapshot lock."""
    path = root/'docs/comparison/manifests/sources.json'
    receipt = json.loads((root/'logs/m7_sources_lock.log').read_text(encoding='utf-8'))
    if sha256(path) != receipt['source_snapshot_sha256']:
        raise ValueError('Locked source snapshot changed.')
    sources = json.loads(path.read_text(encoding='utf-8'))
    for name, expected in sources.items():
        if sha256(root/name) != expected:
            raise ValueError(f'Locked source artifact changed: {name}')
    return sources


def write_table(path: Path, records: list[dict], columns: list[str]) -> None:
    """Write full-precision CSV and display-only rounded Markdown deterministically."""
    def value(item: object) -> object:
        return format(item, '.17g') if isinstance(item, float) else item

    with path.with_suffix('.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(columns)
        writer.writerows([[value(row[key]) for key in columns] for row in records])
    lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for row in records:
        cells = []
        for key in columns:
            item = row[key]
            digits = 6 if 'ssim' in key else 3 if key.endswith('_ms') else 4
            cells.append(format(item, f'.{digits}f') if isinstance(item, float) else str(item))
        lines.append('| '+' | '.join(cells)+' |')
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')


def write_outputs(rows: list[dict], output: Path) -> None:
    """Create numeric tables and six pure-vector charts in an exclusive directory."""
    from .comparison_plots import write_charts

    if (output/'tables').exists() or (output/'figures').exists():
        raise FileExistsError('Comparison output children already exist.')
    output.mkdir(parents=True, exist_ok=True)
    tables = output/'tables'
    tables.mkdir()
    columns = ['algorithm', 'image_id', 'source_run_id', 'source_json_sha256', *METRICS]
    write_table(tables/'per_image', rows, columns)
    write_table(tables/'mean', summarize(rows), ['algorithm', 'n_images', *METRICS])
    baseline = {r['image_id']: r for r in rows if r['algorithm'] == 'Bicubic'}
    delta = [{**row, **{key: row[key]-baseline[row['image_id']][key] for key in METRICS}} for row in rows]
    write_table(tables/'delta_bicubic', delta, columns)
    for scope in ('pipeline', 'network'):
        timing = [{**{key: row[key] for key in ('algorithm', 'image_id', 'source_run_id')},
                   **row[scope]} for row in rows if row[scope] is not None]
        write_table(tables/f'timing_{scope}', timing,
                    ['algorithm', 'image_id', 'source_run_id', 'median_ms', 'iqr_ms',
                     'q25_ms', 'q75_ms', 'warmup', 'repeats'])
    write_charts(rows, output/'figures')
