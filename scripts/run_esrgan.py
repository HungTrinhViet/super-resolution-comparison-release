"""Run one locked Original ESRGAN CPU experiment with converted BasicSR weights."""
import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from sr_compare.esrgan import (
    CHECKPOINT, CHECKPOINT_SHA256, SOURCE_NOTICE, RrdbNet,
    load_esrgan, prepare_input, super_resolve,
)
from sr_compare.experiment import environment, write_json
from sr_compare.io import load_pair, save_rgb
from sr_compare.metrics import evaluate
from sr_compare.timing import measure

LOGGER = logging.getLogger(__name__)
LOCK_DOCUMENT = Path('docs/M6_ESRGAN_RESULTS.md')
LOCK_RECEIPT = Path('logs/m6_lock.log')
LOCK_SOURCES = (
    'scripts/run_esrgan.py', 'src/sr_compare/esrgan.py',
    'src/sr_compare/experiment.py', 'src/sr_compare/io.py',
    'src/sr_compare/metrics.py', 'src/sr_compare/timing.py',
    'src/sr_compare/bicubic.py', 'src/sr_compare/degradation.py',
    'src/sr_compare/__init__.py',
)
BENCHMARK = {'warmup': 3, 'repeats': 10, 'batch_size': 1,
             'precision': 'FP32', 'threads': 4, 'interop_threads': 1,
             'pipeline': 'RGB preparation + network + quantization',
             'forward': 'network only, prepared input'}
PREPROCESSING = ('RGB uint8 -> FP32 /255 NCHW; no mean shift; network; '
                 'finite check; clamp [0,1]; x255; half-to-even; RGB uint8')


def config_section(document: str) -> str:
    """Extract the exact configuration text inside explicit lock markers."""
    return document.split('<!-- LOCKED_CONFIG_BEGIN -->', 1)[1].split(
        '<!-- LOCKED_CONFIG_END -->', 1)[0]


def physical_ram_bytes() -> int:
    """Query installed physical RAM through the Windows API."""
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    *[(name, ctypes.c_ulonglong) for name in (
                        'total_physical', 'available_physical', 'total_pagefile',
                        'available_pagefile', 'total_virtual', 'available_virtual',
                        'available_extended')]]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return int(status.total_physical)


def hardware_snapshot() -> dict:
    """Capture stable hardware, thread and runtime fields for lock validation."""
    captured = environment()
    return {**{key: captured[key] for key in (
        'cpu', 'logical_cpus', 'os', 'python', 'thread_environment')},
        'ram_bytes': physical_ram_bytes(), 'torch_version': torch.__version__,
        'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads()}


def verify_lock() -> dict:
    """Reject changed configuration, sources, checkpoint or environment."""
    receipt = json.loads(LOCK_RECEIPT.read_text(encoding='utf-8'))
    datetime.fromisoformat(receipt['locked_utc'])
    if (receipt['benchmark'] != BENCHMARK
            or receipt['preprocessing'] != PREPROCESSING
            or receipt['checkpoint_sha256'] != CHECKPOINT_SHA256):
        raise ValueError('Receipt differs from the fixed ESRGAN configuration.')
    if set(receipt['source_sha256']) != set(LOCK_SOURCES):
        raise ValueError('Receipt has an incomplete or unexpected source list.')
    for name, expected in receipt['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Locked source changed: {name}')
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('Locked checkpoint changed.')
    section = config_section(LOCK_DOCUMENT.read_text(encoding='utf-8'))
    if hashlib.sha256(section.encode('utf-8')).hexdigest() != receipt['config_sha256']:
        raise ValueError('Locked configuration section changed.')
    current = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    if sorted(current.splitlines()) != receipt['pip_freeze']:
        raise ValueError('Locked dependencies changed.')
    if (torch.get_num_threads(), torch.get_num_interop_threads()) != (4, 1):
        raise ValueError('Locked thread settings changed.')
    if receipt['hardware'] != hardware_snapshot():
        raise ValueError('Locked hardware or thread environment changed.')
    return receipt


def output_statistics(values: np.ndarray) -> dict:
    """Count exact half ties and clipping on unclamped RGB intensities."""
    below, above = values < 0, values > 255
    ties = (values-np.floor(values)) == .5
    return {
        'half_tie_pixels_any_channel': int(np.any(ties, axis=-1).sum()),
        'half_tie_channel_values': int(ties.sum()),
        'clip_below_zero_percent': float(100*np.mean(below)),
        'clip_above_255_percent': float(100*np.mean(above)),
        'clip_pixels_any_channel_percent': float(100*np.mean(
            np.any(below | above, axis=-1))),
        'prequant_float_min': float(values.min()),
        'prequant_float_max': float(values.max()),
    }


def run_pairs(pairs: list[tuple[str, np.ndarray, np.ndarray]], output: Path,
              model: RrdbNet, captured_environment: dict) -> dict:
    """Write one exclusive run with all timing samples and prequant floats."""
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    if 'lock_receipt' in captured_environment:
        locked = captured_environment['lock_receipt']['locked_utc']
        if datetime.fromisoformat(started) <= datetime.fromisoformat(locked):
            raise ValueError('Metrics run must start after configuration lock.')
    LOGGER.info('METRICS_RUN_START utc=%s output=%s', started, output)
    captured = {**captured_environment, 'metrics_run_start_utc': started,
                'source_notice': SOURCE_NOTICE}
    rows = []
    model.eval()
    for name, hr, lr in pairs:
        result = super_resolve(model, lr)
        metrics = evaluate(hr, result.rgb)
        statistics = output_statistics(result.prequant_float)
        pipeline = measure(lambda: super_resolve(model, lr), 3, 10)
        tensor = prepare_input(lr)
        with torch.inference_mode():
            forward = measure(lambda: model(tensor), 3, 10)
        save_rgb(output/f'{name}_SR.png', result.rgb)
        with (output/f'{name}_prequant.npy').open('xb') as stream:
            np.save(stream, result.prequant_float, allow_pickle=False)
        row = {'name': name, 'hr_shape': list(hr.shape), 'lr_shape': list(lr.shape),
               **metrics, **statistics, 'timing_pipeline': pipeline,
               'timing_forward': forward,
               'prequant_file': f'{name}_prequant.npy'}
        rows.append(row)
        LOGGER.info('%s y_psnr=%.8f y_ssim=%.8f pipeline_ms=%.6f forward_ms=%.6f',
                    name, metrics['y']['psnr'], metrics['y']['ssim'],
                    pipeline['median_ms'], forward['median_ms'])
    mean = {space: {metric: float(np.mean([row[space][metric] for row in rows]))
                    for metric in ('mse', 'psnr', 'ssim')}
            for space in ('y', 'rgb')}
    report = {'run_id': output.name, 'started_utc': started, 'scale': 4,
              'model': 'Original_ESRGAN_RRDB23_BasicSR_converted',
              'source_notice': SOURCE_NOTICE, 'precision': 'FP32',
              'protocol': PREPROCESSING+' -> unchanged M1 evaluate',
              'images': rows, 'mean_per_image': mean}
    write_json(output/'metrics.json', report)
    write_json(output/'environment.json', captured)
    stat_keys = ['half_tie_pixels_any_channel', 'half_tie_channel_values',
                 'clip_below_zero_percent', 'clip_above_255_percent',
                 'clip_pixels_any_channel_percent']
    columns = ['name', 'y_mse', 'y_psnr', 'y_ssim', 'rgb_mse', 'rgb_psnr',
               'rgb_ssim', *stat_keys, 'pipeline_median_ms', 'pipeline_iqr_ms',
               'forward_median_ms', 'forward_iqr_ms']
    with (output/'metrics.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({'name': row['name'],
                             **{f'{space}_{metric}': row[space][metric]
                                for space in ('y', 'rgb')
                                for metric in ('mse', 'psnr', 'ssim')},
                             **{key: row[key] for key in stat_keys},
                             **{f'{scope}_{stat}': row[f'timing_{scope}'][stat]
                                for scope in ('pipeline', 'forward')
                                for stat in ('median_ms', 'iqr_ms')}})
    return report


def main() -> None:
    """Validate the receipt before running exactly five fixed Set5 inputs."""
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
        'set5_esrgan_x4_%Y%m%d_%H%M%S')
    names = [f'img_{index:03d}_SRF_4' for index in range(1, 6)]
    pairs = [(name, *load_pair(args.data_dir/f'{name}_HR.png',
                               args.data_dir/f'{name}_LR.png')) for name in names]
    model = load_esrgan()
    captured = {**environment(), **hardware_snapshot(), 'device': 'cpu',
                'precision': 'FP32', 'batch_size': 1, 'lock_receipt': receipt,
                'checkpoint': CHECKPOINT.as_posix(),
                'checkpoint_sha256': CHECKPOINT_SHA256,
                'pip_freeze': receipt['pip_freeze'], 'benchmark': BENCHMARK,
                'preprocessing': PREPROCESSING}
    run_pairs(pairs, output, model, captured)


if __name__ == '__main__':
    main()
