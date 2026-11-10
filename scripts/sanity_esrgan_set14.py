"""Check Original ESRGAN architecture on Set14; BasicSR conversion UNVERIFIED."""
import argparse
import ctypes
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from sr_compare.bicubic import upscale
from sr_compare.degradation import make_lr, modcrop
from sr_compare.esrgan import (
    CHECKPOINT_SHA256, SOURCE_NOTICE, load_esrgan, super_resolve,
)
from sr_compare.matlab_color import rgb2ycbcr_uint8
from sr_compare.metrics import evaluate

ARCHIVE = Path('data/quarantine/srcnn/SRCNN_v1.zip')
ARCHIVE_SHA256 = 'bfa68ca613c1326a59e0c34353205a254ab2b67e34df7f04e28eef567980af30'


def displacement(reference: NDArray, moving: NDArray) -> tuple[int, int]:
    """Return moving-minus-reference lag at the full correlation peak."""
    if (reference.shape != moving.shape or reference.ndim != 2
            or not np.isfinite(reference).all() or not np.isfinite(moving).all()):
        raise ValueError('Alignment requires equal finite 2-D arrays.')
    first, second = reference.astype(float), moving.astype(float)
    correlation = correlate(second-second.mean(), first-first.mean(),
                            mode='full', method='fft')
    peak = np.unravel_index(correlation.argmax(), correlation.shape)
    return tuple(int(peak[i]-reference.shape[i]+1) for i in (0, 1))


def pixel_record(image: NDArray) -> dict:
    """Describe raw C-order pixels rather than encoded PNG bytes."""
    return {'sha256_raw_pixels': hashlib.sha256(image.tobytes()).hexdigest(),
            'shape': list(image.shape), 'dtype': str(image.dtype)}


def pearson_record(first: NDArray, second: NDArray) -> float | None:
    """Record correlation, with null when the selected region lacks variance."""
    a, b = first.astype(np.float64).ravel(), second.astype(np.float64).ravel()
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def extra_diagnostics(sr: NDArray, bicubic: NDArray, hr: NDArray) -> dict:
    """Record low-frequency and content-region correlations without thresholds."""
    smoothed = {}
    for sigma in (1, 2, 4):
        a, b, c = (gaussian_filter(values.astype(np.float64), sigma=sigma,
                                   mode='reflect') for values in (sr, bicubic, hr))
        smoothed[str(sigma)] = {'sr_bicubic': pearson_record(a, b),
                               'sr_hr': pearson_record(a, c)}
    mask = hr > 20
    return {'gaussian_sigma_pixels': smoothed,
            'raw_sr_hr': pearson_record(sr, hr),
            'content_mask': 'HR Y uint8 > 20',
            'content_pixel_count': int(mask.sum()),
            'content_sr_bicubic': pearson_record(sr[mask], bicubic[mask]),
            'content_sr_hr': pearson_record(sr[mask], hr[mask])}


def peak_working_set_bytes() -> int:
    """Query this process's lifetime peak working set using Windows APIs."""
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong),
                    *[(name, ctypes.c_size_t) for name in (
                        'peak', 'working', 'peak_pool_paged', 'pool_paged',
                        'peak_pool_nonpaged', 'pool_nonpaged', 'pagefile', 'peak_pagefile')]]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess
    process.argtypes, process.restype = [], ctypes.c_void_p
    memory = ctypes.windll.psapi.GetProcessMemoryInfo
    memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    memory.restype = ctypes.c_int
    if not memory(process(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return int(counters.peak)


def synthetic_image() -> NDArray:
    """Create fixed texture, edges and blocks for pipeline checks."""
    rng = np.random.default_rng(190)
    image = np.zeros((96, 128), dtype=np.uint8)
    image[16:80, 20:108] = rng.integers(40, 210, (64, 88), dtype=np.uint8)
    image[25:45, 30:55] = 220
    return np.repeat(image[..., None], 3, axis=2)


def inputs() -> dict[str, NDArray]:
    """Read two verified archive BMPs into memory without extracting files."""
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('SRCNN archive SHA-256 differs from verified bytes.')
    images = {'synthetic': synthetic_image()}
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in ('zebra', 'monarch'):
            with Image.open(BytesIO(archive.read(f'SRCNN/Set14/{name}.bmp'))) as image:
                if image.mode != 'RGB':
                    raise ValueError(f'{name}: expected RGB, got {image.mode}.')
                images[name] = np.array(image)
    return images


def save_views(directory: Path, name: str, images: dict[str, NDArray]) -> None:
    """Save full views and matching central crops enlarged with nearest."""
    for kind, array in images.items():
        image = Image.fromarray(array)
        image.save(directory / f'{name}_{kind}.png')
        row, column = array.shape[0]//2, array.shape[1]//2
        crop = image.crop((column-32, row-32, column+32, row+32))
        crop.resize((256, 256), Image.Resampling.NEAREST).save(
            directory / f'{name}_{kind}_crop_nearest_x4.png')


def sanity(directory: Path) -> dict:
    """Check required criteria and record diagnostics without selecting settings."""
    directory.mkdir(parents=True, exist_ok=False)
    model = load_esrgan()
    reference = synthetic_image()[..., 0].astype(float)
    for shift in ((3, -2), (0, 0)):
        moving = reference.copy() if shift == (0, 0) else np.zeros_like(reference)
        if shift != (0, 0):
            moving[3:, :-2] = reference[:-3, 2:]
        observed = displacement(reference, moving)
        if observed != shift:
            raise ValueError(f'G-T9 known shift failed: {shift}, {observed}')
        print('G-T9 known shift PASS', shift, observed)
    reports, records = {}, {}
    for name, original in inputs().items():
        hr = modcrop(original)
        lr = make_lr(original)
        first, second = super_resolve(model, lr), super_resolve(model, lr)
        if (first.rgb.shape != hr.shape or first.rgb.dtype != np.uint8
                or not np.isfinite(first.prequant_float).all()):
            raise ValueError(f'G-T9 output validity failed: {name}')
        if not np.array_equal(first.rgb, second.rgb):
            raise ValueError(f'G-T9 same-process reproducibility failed: {name}')
        records[name] = pixel_record(first.rgb)
        bicubic = upscale(lr)
        base_y = rgb2ycbcr_uint8(bicubic)[..., 0]
        hr_y = rgb2ycbcr_uint8(hr)[..., 0]
        sr_y = rgb2ycbcr_uint8(first.rgb)[..., 0]
        pearson = float(np.corrcoef(sr_y.ravel(), base_y.ravel())[0, 1])
        if not np.isfinite(pearson):
            raise ValueError(f'G-T9 nonfinite Pearson: {name}')
        shift_base = displacement(base_y, sr_y)
        shift_hr = displacement(hr_y, sr_y)
        warnings = []
        fatal = []
        if pearson < .95:
            warnings.append('Pearson below locked warning threshold 0.95')
        if max(map(abs, shift_hr)) > 1:
            fatal.append('HR correlation peak outside locked +/-1 pixel range')
        if max(map(abs, shift_base)) > 1:
            fatal.append('Bicubic correlation peak outside locked +/-1 pixel range')
        reports[name] = {
            'raw_pixels': records[name], 'same_process_equal': True,
            'finite_prequantization_rgb': True,
            'rgb_float_min': float(first.prequant_float.min()),
            'rgb_float_max': float(first.prequant_float.max()),
            'clip_below_zero_percent': float(100*np.mean(first.prequant_float < 0)),
            'clip_above_255_percent': float(100*np.mean(first.prequant_float > 255)),
            'pearson_vs_bicubic_y': pearson, 'lag_vs_bicubic_y': shift_base,
            'lag_vs_hr_y': shift_hr, 'warnings': warnings, 'fatal_conditions': fatal,
            'status': 'WARNING_RECORDED' if warnings else 'PASS',
            'diagnostic_only_no_thresholds': extra_diagnostics(sr_y, base_y, hr_y),
            'metrics_record_only_not_selection': evaluate(hr, first.rgb),
        }
        save_views(directory, name, {'sr': first.rgb, 'bicubic': bicubic, 'hr': hr})
        print(name, json.dumps(reports[name], allow_nan=False))
        if fatal:
            report = {'source_notice': SOURCE_NOTICE, 'status': 'FAIL_STOP',
                      'images': reports}
            (directory / 'report.json').write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
            raise ValueError(f'G-T9 fatal alignment condition: {name}; stop for review.')
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                           check=True, capture_output=True, text=True, encoding='utf-8')
    child_payload = json.loads(child.stdout)
    if (child_payload['source_notice'] != SOURCE_NOTICE
            or child_payload['records'] != records):
        raise ValueError('G-T9 fresh-process raw-pixel hash/shape/dtype mismatch.')
    for name in reports:
        reports[name]['fresh_process_raw_pixels_equal'] = True
    report = {'source_notice': SOURCE_NOTICE,
              'checkpoint_sha256': CHECKPOINT_SHA256,
              'status': ('WARNING_RECORDED' if any(r['warnings'] for r in reports.values())
                         else 'PASS_REQUIRED_AND_NUMERIC_DIAGNOSTICS'),
              'required_checks': 'PASS',
              'visual_review': 'PENDING', 'images': reports,
              'child_process_records': child_payload['records']}
    (directory / 'report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def main() -> None:
    """Set fixed threads once per process and check Set14 only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('logs/m6_t9'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.worker:
        model = load_esrgan()
        records = {name: pixel_record(super_resolve(model, make_lr(image)).rgb)
                   for name, image in inputs().items()}
        print(json.dumps({'source_notice': SOURCE_NOTICE, 'records': records}))
        return
    logging.basicConfig(level=logging.INFO)
    print(SOURCE_NOTICE)
    config = StringIO()
    with redirect_stdout(config):
        np.show_config()
    environment = {
        'source_notice': SOURCE_NOTICE,
        'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'numpy_config': config.getvalue(),
        'thread_environment': {name: value for name, value in os.environ.items()
                               if name.startswith(('OMP', 'MKL', 'OPENBLAS'))},
        'torch_version': torch.__version__,
    }
    print(json.dumps(environment, indent=2))
    started = time.perf_counter()
    report = sanity(args.output_dir)
    environment['sanity_wall_seconds_including_child'] = time.perf_counter()-started
    environment['parent_lifetime_peak_working_set_bytes'] = peak_working_set_bytes()
    (args.output_dir / 'environment.json').write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding='utf-8')
    print('G-T9', report['status'], '; visual review PENDING')


if __name__ == '__main__':
    main()
