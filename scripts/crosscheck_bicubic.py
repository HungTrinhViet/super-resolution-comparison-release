"""Independently check saved Bicubic x4 pixels and Y metrics, read-only."""
import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.signal import convolve2d

TOLERANCE = 1e-6


def cubic(distance: NDArray) -> NDArray:
    """Evaluate the Keys cubic kernel with a=-0.5."""
    x = np.abs(np.asarray(distance, dtype=np.float64))
    return ((1.5 * x**3 - 2.5 * x**2 + 1) * (x <= 1)
            + (-.5 * x**3 + 2.5 * x**2 - 4 * x + 2)
            * ((x > 1) & (x <= 2)))


def contributions(length: int) -> tuple[NDArray, NDArray]:
    """Build x4 weights using 1-based centers and symmetric sample indices."""
    positions = np.arange(1, 4 * length + 1)
    centers = positions / 4 + .5 * (1 - 1 / 4)
    left = np.floor(centers - 2)
    indices = left[:, None] + np.arange(6)[None, :]
    weights = cubic(centers[:, None] - indices)
    weights /= weights.sum(axis=1, keepdims=True)
    reflected = np.concatenate([np.arange(1, length + 1),
                                np.arange(length, 0, -1)])
    indices = reflected[np.mod(indices - 1, len(reflected)).astype(int)] - 1
    return weights, indices


def upsample(image: NDArray) -> NDArray:
    """Resize x4 in two float64 passes, quantizing once after both axes."""
    if (image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or min(image.shape[:2]) < 1):
        raise ValueError('Expected a nonempty uint8 RGB image.')
    result = image.astype(np.float64)
    for axis in [0, 1]:
        weights, indices = contributions(result.shape[axis])
        pixels = np.moveaxis(result, axis, 0)[indices]
        result = np.moveaxis(
            np.einsum('op,op...->o...', weights, pixels), 0, axis)
    return np.floor(np.clip(result, 0, 255) + .5).astype(np.uint8)


def read_rgb(path: Path) -> NDArray:
    """Read an RGB file without mode conversion or reference resizing."""
    with Image.open(path) as image:
        if image.mode != 'RGB':
            raise ValueError(f'{path}: expected RGB, found {image.mode}.')
        return np.array(image, dtype=np.uint8)


def luma(image: NDArray) -> NDArray:
    """Convert RGB to float BT.601 Y on the 0..255 intensity scale."""
    values = image.astype(np.float64)
    return 16 + (65.481 * values[..., 0] + 128.553 * values[..., 1]
                 + 24.966 * values[..., 2]) / 255


def ssim_wang(reference: NDArray, result: NDArray) -> float:
    """Compute population-moment Gaussian SSIM with valid 2D convolutions."""
    radius = np.arange(11) - 5
    window = np.exp(-(radius[:, None]**2 + radius[None, :]**2)
                    / (2 * 1.5**2))
    window /= window.sum()
    mean_ref = convolve2d(reference, window, mode='valid')
    mean_result = convolve2d(result, window, mode='valid')
    var_ref = (convolve2d(reference**2, window, mode='valid')
               - mean_ref**2)
    var_result = (convolve2d(result**2, window, mode='valid')
                  - mean_result**2)
    covariance = (convolve2d(reference * result, window, mode='valid')
                  - mean_ref * mean_result)
    c1, c2 = (.01 * 255)**2, (.03 * 255)**2
    index = (((2 * mean_ref * mean_result + c1) * (2 * covariance + c2))
             / ((mean_ref**2 + mean_result**2 + c1)
                * (var_ref + var_result + c2)))
    return float(index.mean())


def metrics_y(hr: NDArray, sr: NDArray) -> tuple[float, float]:
    """Compute Y PSNR and SSIM after exact shape checking and crop 4."""
    if hr.shape != sr.shape or min(hr.shape[:2]) < 19:
        raise ValueError('Images must match and remain >=11x11 after crop 4.')
    reference, result = luma(hr)[4:-4, 4:-4], luma(sr)[4:-4, 4:-4]
    mse = float(np.mean((reference - result)**2))
    psnr = math.inf if mse == 0 else 10 * math.log10(255**2 / mse)
    return psnr, ssim_wang(reference, result)


def crosscheck(data_dir: Path, run_dir: Path) -> bool:
    """Compare every Set5 saved image and CSV score, printing all outcomes."""
    with (run_dir / 'metrics.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    names = [f'img_{i:03d}_SRF_4' for i in range(1, 6)]
    reported = {row['name']: row for row in rows}
    if len(rows) != 5 or set(reported) != set(names):
        raise ValueError('metrics.csv must contain exactly the five Set5 IDs.')
    print(f'UTC={datetime.now(timezone.utc).isoformat()}')
    print(f'Absolute metric tolerance={TOLERANCE:g}; relative tolerance=0')
    print('image | pixels | PSNR_Y | SSIM_Y | abs_delta_PSNR | '
          'abs_delta_SSIM | status')
    passed, scores = True, []
    for name in names:
        try:
            hr = read_rgb(data_dir / f'{name}_HR.png')
            lr = read_rgb(data_dir / f'{name}_LR.png')
            saved = read_rgb(run_dir / f'{name}_SR.png')
            generated = upsample(lr)
            same = generated.shape == saved.shape and np.array_equal(
                generated, saved)
            psnr, ssim = metrics_y(hr, generated)
            expected_psnr = float(reported[name]['y_psnr'])
            expected_ssim = float(reported[name]['y_ssim'])
            metrics_match = (math.isclose(psnr, expected_psnr, rel_tol=0,
                                          abs_tol=TOLERANCE)
                             and math.isclose(ssim, expected_ssim, rel_tol=0,
                                              abs_tol=TOLERANCE))
            ok = same and metrics_match
            scores.append((psnr, ssim))
            print(f'{name} | {"PASS" if same else "FAIL"} | {psnr:.8f} | '
                  f'{ssim:.8f} | {abs(psnr - expected_psnr):.3e} | '
                  f'{abs(ssim - expected_ssim):.3e} | '
                  f'{"PASS" if ok else "FAIL"}')
        except (OSError, ValueError, KeyError) as error:
            print(f'{name} | FAIL: {error}')
            ok = False
        passed = passed and ok
    if len(scores) == 5:
        mean = np.mean(scores, axis=0)
        print(f'MEAN PSNR_Y={mean[0]:.8f} SSIM_Y={mean[1]:.8f}')
    return passed


def main() -> int:
    """Run a read-only independent check; return nonzero for any failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=Path('data/set5/image_SRF_4'))
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        return 0 if crosscheck(args.data_dir, args.run_dir) else 1
    except (OSError, ValueError, KeyError) as error:
        print(f'FAIL: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
