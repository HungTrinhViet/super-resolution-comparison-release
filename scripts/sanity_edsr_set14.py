"""EDSR sanity on author Set14 images and synthetic alignment; never Set5."""
import argparse
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.signal import correlate
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from sr_compare.bicubic import upscale
from sr_compare.degradation import make_lr, modcrop
from sr_compare.edsr import CHECKPOINT_SHA256, load_edsr, super_resolve
from sr_compare.experiment import environment
from sr_compare.metrics import evaluate, rgb_to_y

ARCHIVE = Path('data/quarantine/srcnn/SRCNN_v1.zip')
ARCHIVE_SHA256 = 'bfa68ca613c1326a59e0c34353205a254ab2b67e34df7f04e28eef567980af30'


def displacement(reference: NDArray, moving: NDArray) -> tuple[int, int]:
    """Return row/column moving-minus-reference lag of full correlation peak."""
    if (reference.shape != moving.shape or reference.ndim != 2
            or not np.isfinite(reference).all() or not np.isfinite(moving).all()):
        raise ValueError('Alignment requires matching finite 2-D arrays.')
    first, second = reference.astype(float), moving.astype(float)
    correlation = correlate(second-second.mean(), first-first.mean(),
                            mode='full', method='fft')
    peak = np.unravel_index(correlation.argmax(), correlation.shape)
    return tuple(int(peak[i]-reference.shape[i]+1) for i in (0, 1))


def pixel_record(image: NDArray) -> dict:
    """Hash raw C-order pixels with explicit shape and dtype metadata."""
    return {'sha256_raw_pixels': hashlib.sha256(image.tobytes()).hexdigest(),
            'shape': list(image.shape), 'dtype': str(image.dtype)}


def inputs() -> dict[str, NDArray]:
    """Read two BMP members in memory and create one fixed diagnostic pattern."""
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('Author Set14 archive differs from verified bytes.')
    gray = np.zeros((96, 128), dtype=np.uint8)
    gray[16:80, 20:108] = np.random.default_rng(190).integers(
        40, 210, (64, 88), dtype=np.uint8)
    gray[25:45, 30:55] = 220
    result = {'synthetic': np.repeat(gray[..., None], 3, axis=2)}
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in ('zebra', 'monarch'):
            with Image.open(BytesIO(archive.read(f'SRCNN/Set14/{name}.bmp'))) as image:
                if image.mode != 'RGB':
                    raise ValueError(f'{name}: expected RGB, got {image.mode}.')
                result[name] = np.array(image)
    return result


def save_views(directory: Path, name: str, images: dict[str, NDArray]) -> None:
    """Save full views and labeled nearest-enlarged central crops."""
    for kind, array in images.items():
        image = Image.fromarray(array)
        image.save(directory/f'{name}_{kind}.png')
        row, column = array.shape[0]//2, array.shape[1]//2
        image.crop((column-32, row-32, column+32, row+32)).resize(
            (256, 256), Image.Resampling.NEAREST).save(
                directory/f'{name}_{kind}_crop_nearest_x4.png')


def sanity(directory: Path) -> dict:
    """Run required checks and record diagnostics without selecting parameters."""
    directory.mkdir(parents=True, exist_ok=False)
    model = load_edsr()
    samples = inputs()
    reference = samples['synthetic'][..., 0].astype(float)
    for shift in ((3, -2), (0, 0)):
        moving = reference.copy() if shift == (0, 0) else np.zeros_like(reference)
        if shift != (0, 0):
            moving[3:, :-2] = reference[:-3, 2:]
        found = displacement(reference, moving)
        if found != shift:
            raise ValueError(f'E-T9 known-shift alignment failed: {shift}, {found}')
        print('E-T9 known shift PASS', shift)
    reports, records = {}, {}
    for name, image in samples.items():
        hr, lr = modcrop(image), make_lr(image)
        first, second = super_resolve(model, lr), super_resolve(model, lr)
        if (first.rgb.dtype != np.uint8 or first.rgb.shape != hr.shape
                or not np.isfinite(first.prequant_float).all()):
            raise ValueError(f'E-T9 invalid output: {name}')
        if not np.array_equal(first.rgb, second.rgb):
            raise ValueError(f'E-T9 within-process mismatch: {name}')
        records[name] = pixel_record(first.rgb)
        bicubic = upscale(lr)
        result_y, base_y, hr_y = rgb_to_y(first.rgb), rgb_to_y(bicubic), rgb_to_y(hr)
        pearson = float(np.corrcoef(result_y.ravel(), base_y.ravel())[0, 1])
        if not np.isfinite(pearson):
            raise ValueError(f'E-T9 nonfinite Pearson: {name}')
        lag_base, lag_hr = displacement(base_y, result_y), displacement(hr_y, result_y)
        if name == 'synthetic' and max(map(abs, lag_base)) > 1:
            raise ValueError(f'E-T9 synthetic pipeline alignment failed: {lag_base}')
        warnings = []
        if pearson < .95:
            warnings.append('Pearson below locked .95 warning threshold')
        if max(map(abs, lag_hr)) > 1:
            warnings.append('HR correlation peak outside locked +/-1 pixel range')
        reports[name] = {
            'raw_pixels': records[name], 'same_process_equal': True,
            'finite_prequantization': True, 'float_min': float(first.prequant_float.min()),
            'float_max': float(first.prequant_float.max()),
            'clip_below_zero_percent': float(100*np.mean(first.prequant_float < 0)),
            'clip_above_255_percent': float(100*np.mean(first.prequant_float > 255)),
            'half_tie_pixels_any_channel': first.half_tie_count,
            'half_tie_channel_values': first.half_tie_channel_count,
            'pearson_vs_bicubic_y': pearson, 'lag_vs_bicubic_y': lag_base,
            'lag_vs_hr_y': lag_hr, 'warnings': warnings,
            'metrics_record_only_not_selection': evaluate(hr, first.rgb),
        }
        save_views(directory, name, {'sr': first.rgb, 'bicubic': bicubic, 'hr': hr})
        print(name, json.dumps(reports[name], allow_nan=False))
        if warnings:
            (directory/'report.json').write_text(
                json.dumps({'status': 'WARNING_STOP', 'images': reports}, indent=2),
                encoding='utf-8')
            raise ValueError(f'E-T9 diagnostic warning: {name}; stop for review.')
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                           capture_output=True, text=True, check=True)
    child_records = json.loads(child.stdout)
    if records != child_records:
        raise ValueError('E-T9 fresh-process raw pixel hash/shape/dtype mismatch.')
    for row in reports.values():
        row['fresh_process_raw_pixels_equal'] = True
    report = {'status': 'PASS_REQUIRED_AND_NUMERIC_DIAGNOSTICS',
              'visual_review': 'PENDING', 'images': reports,
              'child_process_records': child_records}
    (directory/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main() -> None:
    """Set four/one threads once and run only synthetic/Set14 sanity inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('logs/m5_t9'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.worker:
        model = load_edsr()
        print(json.dumps({name: pixel_record(super_resolve(model, make_lr(image)).rgb)
                          for name, image in inputs().items()}))
        return
    logging.basicConfig(level=logging.INFO)
    config = StringIO()
    with redirect_stdout(config):
        np.show_config()
    captured = {**environment(), 'torch_version': torch.__version__,
                'torch_threads': torch.get_num_threads(),
                'torch_interop_threads': torch.get_num_interop_threads(),
                'thread_environment_all': {name: value for name, value in os.environ.items()
                                            if name.startswith(('OMP', 'MKL', 'OPENBLAS'))},
                'checkpoint_sha256': CHECKPOINT_SHA256, 'numpy_config': config.getvalue()}
    print(json.dumps(captured, indent=2))
    report = sanity(args.output_dir)
    (args.output_dir/'environment.json').write_text(json.dumps(captured, indent=2),
                                                    encoding='utf-8')
    print('E-T9', report['status'], '; visual review PENDING')


if __name__ == '__main__':
    main()
