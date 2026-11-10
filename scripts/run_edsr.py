"""Run the locked EDSR-baseline x4 CPU configuration on five Set5 pairs."""
import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from sr_compare.edsr import (
    CHECKPOINT, CHECKPOINT_SHA256, Edsr, load_edsr, prepare_input, super_resolve,
)
from sr_compare.experiment import environment, write_json
from sr_compare.io import load_pair, save_rgb
from sr_compare.metrics import evaluate
from sr_compare.timing import measure

LOGGER = logging.getLogger(__name__)
LOCK_DOCUMENT = Path('docs/M5_EDSR_RESULTS.md')
LOCK_RECEIPT = Path('logs/m5_lock.log')
LOCK_SOURCES = ('src/sr_compare/edsr.py', 'scripts/run_edsr.py')


def config_section(document: str) -> str:
    """Extract the exact UTF-8 content between the immutable lock markers."""
    return document.split('<!-- LOCKED_CONFIG_BEGIN -->', 1)[1].split(
        '<!-- LOCKED_CONFIG_END -->', 1)[0]


def verify_lock() -> dict:
    """Require the two locked sources and unchanged configuration before inference."""
    receipt = json.loads(LOCK_RECEIPT.read_text(encoding='utf-8'))
    if set(receipt['source_sha256']) != set(LOCK_SOURCES):
        raise ValueError('Receipt does not contain exactly the two EDSR sources.')
    for name, expected in receipt['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Locked EDSR source changed: {name}')
    section = config_section(LOCK_DOCUMENT.read_text(encoding='utf-8'))
    if hashlib.sha256(section.encode('utf-8')).hexdigest() != receipt['config_sha256']:
        raise ValueError('Locked EDSR configuration changed.')
    return receipt


def physical_ram_bytes() -> int:
    """Query Windows physical RAM without introducing a dependency."""
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    *[(name, ctypes.c_ulonglong) for name in (
                        'total_physical', 'available_physical', 'total_pagefile',
                        'available_pagefile', 'total_virtual', 'available_virtual',
                        'available_extended')]]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('GlobalMemoryStatusEx failed.')
    return int(status.total_physical)


def run_pairs(pairs: list[tuple[str, np.ndarray, np.ndarray]], output: Path,
              model: Edsr, captured_environment: dict) -> dict:
    """Save fixed-count timings and metrics without overwriting any prior output."""
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    LOGGER.info('METRICS_RUN_START utc=%s output=%s', started, output)
    model.eval()
    rows = []
    for name, hr, lr in pairs:
        result = super_resolve(model, lr)
        metrics = evaluate(hr, result.rgb)
        values = result.prequant_float
        below, above = values < 0, values > 255
        statistics = {
            'half_tie_pixels_any_channel': result.half_tie_count,
            'half_tie_channel_values': result.half_tie_channel_count,
            'clip_below_zero_percent': float(100*np.mean(below)),
            'clip_above_255_percent': float(100*np.mean(above)),
            'clip_pixels_any_channel_percent': float(100*np.mean(
                np.any(below | above, axis=-1))),
            'prequant_float_min': float(values.min()),
            'prequant_float_max': float(values.max()),
        }
        pipeline_timing = measure(lambda: super_resolve(model, lr), 3, 10)
        tensor = prepare_input(lr)
        with torch.inference_mode():
            forward_timing = measure(lambda: model(tensor), 3, 10)
        save_rgb(output/f'{name}_SR.png', result.rgb)
        row = {'name': name, 'hr_shape': list(hr.shape), 'lr_shape': list(lr.shape),
               **metrics, **statistics, 'timing_pipeline': pipeline_timing,
               'timing_forward': forward_timing}
        rows.append(row)
        LOGGER.info('%s y_psnr=%.8f y_ssim=%.8f pipeline_ms=%.6f forward_ms=%.6f',
                    name, metrics['y']['psnr'], metrics['y']['ssim'],
                    pipeline_timing['median_ms'], forward_timing['median_ms'])
    summary = {space: {metric: float(np.mean([row[space][metric] for row in rows]))
                       for metric in ('mse', 'psnr', 'ssim')}
               for space in ('y', 'rgb')}
    report = {'run_id': output.name, 'started_utc': started, 'scale': 4,
              'model': 'EDSR_baseline_16blocks_64features_x4', 'precision': 'FP32',
              'protocol': 'E2-E4 RGB half-even uint8 -> unchanged M1 evaluate',
              'images': rows, 'mean_per_image': summary}
    write_json(output/'metrics.json', report)
    write_json(output/'environment.json', captured_environment)
    statistic_keys = ['half_tie_pixels_any_channel', 'half_tie_channel_values',
                      'clip_below_zero_percent', 'clip_above_255_percent',
                      'clip_pixels_any_channel_percent']
    columns = ['name', 'y_mse', 'y_psnr', 'y_ssim', 'rgb_mse', 'rgb_psnr',
               'rgb_ssim', *statistic_keys, 'pipeline_median_ms', 'pipeline_iqr_ms',
               'forward_median_ms', 'forward_iqr_ms']
    with (output/'metrics.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({'name': row['name'],
                             **{f'{space}_{metric}': row[space][metric]
                                for space in ('y', 'rgb')
                                for metric in ('mse', 'psnr', 'ssim')},
                             **{key: row[key] for key in statistic_keys},
                             **{f'{scope}_{stat}': row[f'timing_{scope}'][stat]
                                for scope in ('pipeline', 'forward')
                                for stat in ('median_ms', 'iqr_ms')}})
    return report


def main() -> None:
    """Verify the lock, set four/one threads once, then run one complete benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/set5/image_SRF_4'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(message)s')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    receipt = verify_lock()
    output = args.output_dir or Path('results')/datetime.now(timezone.utc).strftime(
        'set5_edsr_baseline_x4_%Y%m%d_%H%M%S')
    names = [f'img_{i:03d}_SRF_4' for i in range(1, 6)]
    pairs = [(name, *load_pair(args.data_dir/f'{name}_HR.png',
                               args.data_dir/f'{name}_LR.png')) for name in names]
    model = load_edsr()
    captured = {**environment(), 'ram_bytes': physical_ram_bytes(),
                'torch_version': torch.__version__, 'device': 'cpu',
                'precision': 'FP32', 'batch_size': 1,
                'torch_threads': torch.get_num_threads(),
                'torch_interop_threads': torch.get_num_interop_threads(),
                'checkpoint': CHECKPOINT.as_posix(),
                'checkpoint_sha256': CHECKPOINT_SHA256, 'lock_receipt': receipt}
    run_pairs(pairs, output, model, captured)


if __name__ == '__main__':
    main()
