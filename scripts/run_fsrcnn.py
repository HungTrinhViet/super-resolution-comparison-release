"""Run the locked reposted FSRCNN(56,12,4) configuration once on Set5."""
import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import subprocess

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sr_compare.experiment import environment, write_json
from sr_compare.io import load_pair, save_rgb
from sr_compare.matlab_color import rgb2ycbcr_uint8
from sr_compare.metrics import evaluate
from sr_compare.fsrcnn import (
    CHECKPOINT, CHECKPOINT_SHA256, Fsrcnn, load_fsrcnn, super_resolve,
)
from sr_compare.timing import measure

LOGGER = logging.getLogger(__name__)
SOURCE_NOTICE = ('Trọng số lấy từ bản đăng lại của gói FSRCNN_test (Chao Dong, ECCV 2016); '
                 'hai bản sao độc lập trùng SHA-256; nguồn gốc trực tiếp từ CUHK và '
                 'cách chọn checkpoint chưa được xác minh; chỉ sử dụng cho mục đích học thuật.')
LOCK_SOURCES = ('src/sr_compare/fsrcnn.py', 'scripts/run_fsrcnn.py',
                'src/sr_compare/matlab_color.py', 'src/sr_compare/matlab_resize.py',
                'src/sr_compare/io.py', 'src/sr_compare/metrics.py',
                'src/sr_compare/timing.py')
LOCK_DOCUMENT = Path('docs/M4_FSRCNN_RESULTS.md')
LOCK_RECEIPT = Path('logs/m4_lock.log')
BENCHMARK = {'warmup': 3, 'repeats': 10, 'batch_size': 1,
             'precision': 'FP32', 'threads': 4, 'interop_threads': 1,
             'pipeline': 'F2-F8', 'forward': 'F4-F5'}
PREPROCESSING = 'RGB uint8 -> MATLAB YCbCr uint8 -> Y/255 FP32 LR; CbCr bicubic x4; MATLAB round; RGB uint8'


def config_section(document: str) -> str:
    """Extract the byte-stable configuration between explicit lock markers."""
    return document.split('<!-- LOCKED_CONFIG_BEGIN -->', 1)[1].split(
        '<!-- LOCKED_CONFIG_END -->', 1)[0]


def verify_lock() -> dict:
    """Verify source and configuration receipt before any Set5 inference."""
    receipt = json.loads(LOCK_RECEIPT.read_text(encoding='utf-8'))
    if (receipt['benchmark'] != BENCHMARK
            or receipt['preprocessing'] != PREPROCESSING
            or receipt['checkpoint_sha256'] != CHECKPOINT_SHA256):
        raise ValueError('Receipt differs from fixed benchmark configuration.')
    datetime.fromisoformat(receipt['locked_utc'])
    if set(receipt['source_sha256']) != set(LOCK_SOURCES):
        raise ValueError('Incomplete locked source list.')
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != receipt['checkpoint_sha256']:
        raise ValueError('Locked checkpoint changed.')
    current = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    if sorted(current.splitlines()) != receipt['pip_freeze']:
        raise ValueError('Locked dependency versions changed.')
    if (torch.get_num_threads(), torch.get_num_interop_threads()) != (4, 1):
        raise ValueError('Locked thread settings changed.')
    if receipt['hardware'] != hardware_snapshot():
        raise ValueError('Locked hardware or thread environment changed.')
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


def hardware_snapshot() -> dict:
    """Capture stable hardware and runtime fields for pre-inference validation."""
    captured = environment()
    return {**{key: captured[key] for key in (
        'cpu', 'logical_cpus', 'os', 'python', 'thread_environment')},
        'ram_bytes': physical_ram_bytes(), 'torch_version': torch.__version__,
        'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads()}


def prepare_forward(lr: np.ndarray) -> torch.Tensor:
    """Prepare the F4-F5 input once, outside the network-only timing interval."""
    values = rgb2ycbcr_uint8(lr)[..., 0].astype(np.float32)/255
    return torch.from_numpy(values)[None, None]


def run_pairs(pairs: list[tuple[str, np.ndarray, np.ndarray]], output: Path,
              model: Fsrcnn, captured_environment: dict) -> dict:
    """Save reference metrics and all fixed-count timing samples exclusively."""
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    LOGGER.info('METRICS_RUN_START utc=%s output=%s', started, output)
    captured_environment = {**captured_environment, 'metrics_run_start_utc': started,
                            'source_notice': SOURCE_NOTICE}
    if 'lock_receipt' in captured_environment:
        if datetime.fromisoformat(started) <= datetime.fromisoformat(
                captured_environment['lock_receipt']['locked_utc']):
            raise ValueError('Metrics must start after configuration lock.')
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
               'clip_below_zero_percent': float(100*np.mean(result.y_float < 0)),
               'clip_above_one_percent': float(100*np.mean(result.y_float > 1)),
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
              'source_notice': SOURCE_NOTICE, 'model': 'FSRCNN_56_12_4_x4', 'precision': 'FP32',
              'protocol': 'F2-F8 RGB uint8 -> M1 evaluate; uint8 Y diagnostic only',
              'images': rows, 'mean_per_image': summary}
    write_json(output / 'metrics.json', report)
    write_json(output / 'environment.json', captured_environment)
    columns = ['name', 'y_mse', 'y_psnr', 'y_ssim', 'rgb_mse', 'rgb_psnr',
               'rgb_ssim', 'diagnostic_k4_psnr', 'clip_below_zero_percent', 'clip_above_one_percent', 'pipeline_median_ms',
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
                             'clip_below_zero_percent': row['clip_below_zero_percent'],
                             'clip_above_one_percent': row['clip_above_one_percent'],
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
        'set5_fsrcnn56124_x4_%Y%m%d_%H%M%S')
    names = [f'img_{index:03d}_SRF_4' for index in range(1, 6)]
    pairs = [(name, *load_pair(args.data_dir / f'{name}_HR.png',
                               args.data_dir / f'{name}_LR.png')) for name in names]
    model = load_fsrcnn(CHECKPOINT)
    captured = {**environment(), 'ram_bytes': physical_ram_bytes(),
                'torch_version': torch.__version__, 'device': 'cpu',
                'precision': 'FP32', 'batch_size': 1,
                'torch_threads': torch.get_num_threads(),
                'torch_interop_threads': torch.get_num_interop_threads(),
                'checkpoint': CHECKPOINT.as_posix(),
                'checkpoint_sha256': CHECKPOINT_SHA256, 'lock_receipt': receipt,
                'benchmark': BENCHMARK, 'preprocessing': PREPROCESSING,
                'pip_freeze': receipt['pip_freeze']}
    run_pairs(pairs, output, model, captured)


if __name__ == '__main__':
    main()
