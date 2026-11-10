"""One reproducible Bicubic experiment with exclusive output creation."""
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import winreg

import numpy as np

from .bicubic import upscale
from .degradation import make_lr, modcrop
from .io import load_pair, load_rgb, save_rgb
from .metrics import evaluate
from .timing import measure, validate_timing

LOGGER = logging.getLogger(__name__)


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, value: dict) -> None:
    """Write JSON exclusively, converting nonfinite floats to null."""
    with path.open('x', encoding='utf-8') as stream:
        json.dump(_json_safe(value), stream, indent=2, allow_nan=False)


def environment() -> dict:
    """Capture CPU/platform, dependencies and thread settings."""
    cpu_key = r'HARDWARE\DESCRIPTION\System\CentralProcessor\0'
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                        cpu_key) as key:
        cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    packages = ['numpy', 'Pillow', 'scikit-image', 'pytest', 'scipy',
                'networkx', 'imageio', 'tifffile', 'packaging', 'lazy-loader',
                'colorama', 'iniconfig', 'pluggy']
    return {
        'cpu': cpu, 'logical_cpus': os.cpu_count(), 'os': platform.platform(),
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name)
                     for name in packages},
        'thread_environment': {name: os.environ.get(name) for name in [
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']},
    }


def run_experiment(data_dir: Path | None, output_dir: Path,
                   backend: str = 'numpy', warmup: int = 3, repeats: int = 30,
                   hr_path: Path | None = None) -> dict:
    """Run unchanged paired images or one modcropped standalone HR."""
    if output_dir.exists():
        raise FileExistsError(f'Output directory already exists: {output_dir}')
    validate_timing(warmup, repeats)
    if backend not in {'numpy', 'pillow'}:
        raise ValueError(f'Unknown backend: {backend}')
    if (data_dir is None) == (hr_path is None):
        raise ValueError('Choose one of data_dir or standalone hr_path.')
    if hr_path is not None:
        hr = modcrop(load_rgb(hr_path))
        inputs = [(hr_path.stem, hr, make_lr(hr, backend))]
    else:
        paths = sorted(data_dir.glob('*_HR.png'))
        if not paths:
            raise FileNotFoundError(f'No *_HR.png files in {data_dir}')
        inputs = [(path.stem[:-3], *load_pair(
            path, path.with_name(path.name.replace('_HR.png', '_LR.png'))))
                  for path in paths]
    if any(min(hr.shape[:2]) < 19 for _, hr, _ in inputs):
        raise ValueError('HR must be at least 19x19 for crop4 and SSIM11.')
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    LOGGER.info('METRICS_RUN_START utc=%s backend=%s', started, backend)
    rows = []
    for name, hr, lr in inputs:
        sr = upscale(lr, backend)
        metrics = evaluate(hr, sr)
        timing = measure(lambda: upscale(lr, backend), warmup, repeats)
        row = {'name': name, 'hr_shape': list(hr.shape),
               'lr_shape': list(lr.shape),
               **metrics, 'timing': timing}
        rows.append(row)
        save_rgb(output_dir / f'{name}_SR.png', sr)
        LOGGER.info('%s Y_PSNR=%.8f Y_SSIM=%.8f median_ms=%.6f', name,
                    metrics['y']['psnr'], metrics['y']['ssim'],
                    timing['median_ms'])
    summary = {space: {metric: float(np.mean([r[space][metric] for r in rows]))
                       for metric in ['mse', 'psnr', 'ssim']}
               for space in ['y', 'rgb']}
    result = {'started_utc': started, 'backend': backend, 'scale': 4,
              'input_kind': 'standalone_hr' if hr_path else 'provided_pairs',
              'protocol': ('quantize RGB uint8 -> Y float64 -> crop4 '
                           '-> metrics; range255'),
              'images': rows, 'mean_per_image': summary}
    write_json(output_dir / 'metrics.json', result)
    write_json(output_dir / 'environment.json', environment())
    csv_path = output_dir / 'metrics.csv'
    with csv_path.open('x', newline='', encoding='utf-8') as stream:
        columns = ['name', 'y_mse', 'y_psnr', 'y_ssim', 'rgb_mse', 'rgb_psnr',
                   'rgb_ssim', 'median_ms', 'iqr_ms']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({'name': row['name'],
                             **{f'{space}_{metric}': row[space][metric]
                                for space in ['y', 'rgb']
                                for metric in ['mse', 'psnr', 'ssim']},
                             'median_ms': row['timing']['median_ms'],
                             'iqr_ms': row['timing']['iqr_ms']})
    return result
