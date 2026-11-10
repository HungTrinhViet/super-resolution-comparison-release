"""Run the locked author SRCNN x4 configuration once on five Set5 pairs."""
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
sys.path.insert(0, str(ROOT / 'src'))

from sr_compare.experiment import environment, write_json
from sr_compare.io import load_pair, save_rgb
from sr_compare.matlab_color import rgb2ycbcr_uint8
from sr_compare.matlab_resize import imresize_bicubic_float
from sr_compare.metrics import evaluate
from sr_compare.srcnn import CHECKPOINT_SHA256, Srcnn, load_srcnn, super_resolve
from sr_compare.timing import measure

LOGGER = logging.getLogger(__name__)
CHECKPOINT = Path('data/checkpoints/srcnn/SRCNN/model/9-5-5(ImageNet)/x4.mat')
LOCK_DOCUMENT = Path('docs/M3_SRCNN_RESULTS.md')
LOCK_RECEIPT = Path('logs/m3_lock.log')


def config_section(document: str) -> str:
    """Extract the byte-stable configuration between explicit lock markers."""
    return document.split('<!-- LOCKED_CONFIG_BEGIN -->', 1)[1].split(
        '<!-- LOCKED_CONFIG_END -->', 1)[0]


def verify_lock() -> dict:
    """Verify source and configuration receipt before any Set5 inference."""
    receipt = json.loads(LOCK_RECEIPT.read_text(encoding='utf-8'))
    for name, expected in receipt['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Locked source changed: {name}')
    section = config_section(LOCK_DOCUMENT.read_text(encoding='utf-8'))
    if hashlib.sha256(section.encode('utf-8')).hexdigest() != receipt['config_sha256']:
        raise ValueError('Locked configuration section changed.')
    return receipt


def physical_ram_bytes() -> int:
    """Query physical RAM using the Windows system API without extra packages."""
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


def diagnostic_psnr(hr: np.ndarray, y_uint8: np.ndarray) -> dict:
    """Record K4 uint8-Y PSNR after crop4, separately from principal metrics."""
    hr_y = rgb2ycbcr_uint8(hr)[..., 0]
    if hr_y.shape != y_uint8.shape or min(hr_y.shape) < 19:
        raise ValueError('Diagnostic Y requires matching shapes and crop margin.')
    difference = hr_y[4:-4, 4:-4].astype(float) - y_uint8[4:-4, 4:-4]
    mse = float(np.mean(difference**2))
    return {'mse': mse, 'psnr': float('inf') if mse == 0 else
            float(10*np.log10(255**2/mse)), 'identical_images': mse == 0,
            'role': 'diagnostic_only_not_ranking'}


def prepare_forward(lr: np.ndarray) -> torch.Tensor:
    """Prepare the P5 input once, outside the network-only timing interval."""
    y = rgb2ycbcr_uint8(lr)[..., 0].astype(np.float64)/255
    values = imresize_bicubic_float(y, 4).astype(np.float32)
    return torch.from_numpy(values)[None, None]


def run_pairs(pairs: list[tuple[str, np.ndarray, np.ndarray]], output: Path,
              model: Srcnn, captured_environment: dict) -> dict:
    """Save reference metrics and all fixed-count timing samples exclusively."""
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    LOGGER.info('METRICS_RUN_START utc=%s output=%s', started, output)
    rows = []
    model.eval()
    for name, hr, lr in pairs:
        result = super_resolve(model, lr)
        metrics = evaluate(hr, result.rgb)
        diagnostic = diagnostic_psnr(hr, result.y_uint8)
        pipeline_timing = measure(lambda: super_resolve(model, lr), 3, 10)
        tensor = prepare_forward(lr)
        with torch.inference_mode():
            forward_timing = measure(lambda: model(tensor), 3, 10)
        save_rgb(output / f'{name}_SR.png', result.rgb)
        row = {'name': name, 'hr_shape': list(hr.shape), 'lr_shape': list(lr.shape),
               **metrics, 'diagnostic_k4': diagnostic,
               'timing_pipeline': pipeline_timing, 'timing_forward': forward_timing}
        rows.append(row)
        LOGGER.info('%s y_psnr=%.8f y_ssim=%.8f pipeline_ms=%.6f forward_ms=%.6f',
                    name, metrics['y']['psnr'], metrics['y']['ssim'],
                    pipeline_timing['median_ms'], forward_timing['median_ms'])
    summary = {space: {metric: float(np.mean([r[space][metric] for r in rows]))
                       for metric in ('mse', 'psnr', 'ssim')}
               for space in ('y', 'rgb')}
    summary['diagnostic_k4_psnr'] = float(np.mean(
        [row['diagnostic_k4']['psnr'] for row in rows]))
    report = {'run_id': output.name, 'started_utc': started, 'scale': 4,
              'model': 'SRCNN_9-5-5_ImageNet_x4', 'precision': 'FP32',
              'protocol': 'P2-P8 RGB uint8 -> M1 evaluate; K4 diagnostic only',
              'images': rows, 'mean_per_image': summary}
    write_json(output / 'metrics.json', report)
    write_json(output / 'environment.json', captured_environment)
    columns = ['name', 'y_mse', 'y_psnr', 'y_ssim', 'rgb_mse', 'rgb_psnr',
               'rgb_ssim', 'diagnostic_k4_psnr', 'pipeline_median_ms',
               'pipeline_iqr_ms', 'forward_median_ms', 'forward_iqr_ms']
    with (output / 'metrics.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({'name': row['name'],
                             **{f'{space}_{metric}': row[space][metric]
                                for space in ('y', 'rgb')
                                for metric in ('mse', 'psnr', 'ssim')},
                             'diagnostic_k4_psnr': row['diagnostic_k4']['psnr'],
                             **{f'{kind}_{stat}': row[f'timing_{kind}'][stat]
                                for kind in ('pipeline', 'forward')
                                for stat in ('median_ms', 'iqr_ms')}})
    return report


def main() -> None:
    """Apply fixed threads and a verified lock, then run exactly five pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/set5/image_SRF_4'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(message)s')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    receipt = verify_lock()
    output = args.output_dir or Path('results') / datetime.now(timezone.utc).strftime(
        'set5_srcnn955_x4_%Y%m%d_%H%M%S')
    names = [f'img_{index:03d}_SRF_4' for index in range(1, 6)]
    pairs = [(name, *load_pair(args.data_dir / f'{name}_HR.png',
                               args.data_dir / f'{name}_LR.png')) for name in names]
    model = load_srcnn(CHECKPOINT)
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
