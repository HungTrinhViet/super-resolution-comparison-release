"""RGB and float BT.601 Y metrics with explicit range and border rules."""
import math

import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity

from .io import validate_rgb


def rgb_to_y(image: NDArray) -> NDArray[np.float64]:
    """Convert uint8 RGB to studio-range BT.601 luma, retaining float64."""
    validate_rgb(image)
    return 16 + image.astype(np.float64) @ np.array(
        [65.481, 128.553, 24.966]) / 255


def _measure(reference: NDArray, result: NDArray,
             channel_axis: int | None) -> dict:
    mse = float(np.mean((reference - result)**2, dtype=np.float64))
    return {
        'mse': mse,
        'psnr': math.inf if mse == 0 else 10 * math.log10(255**2 / mse),
        'ssim': float(structural_similarity(
            reference, result, data_range=255.0, channel_axis=channel_axis,
            gaussian_weights=True, sigma=1.5, win_size=11,
            use_sample_covariance=False, K1=.01, K2=.03)),
        'identical_images': mse == 0,
    }


def evaluate(hr: NDArray, sr: NDArray, border: int = 4) -> dict:
    """Convert colors, crop, then measure equal images without resizing."""
    validate_rgb(hr)
    validate_rgb(sr)
    if hr.shape != sr.shape:
        raise ValueError(f'Metric shape mismatch: {hr.shape} vs {sr.shape}.')
    if type(border) is not int or border < 0:
        raise ValueError('Border must be a nonnegative integer.')
    if min(hr.shape[:2]) - 2 * border < 11:
        raise ValueError('Metrics need at least 11x11 after border crop.')
    region = (slice(border, -border or None), slice(border, -border or None))
    hr_y, sr_y = rgb_to_y(hr)[region], rgb_to_y(sr)[region]
    return {
        'y': _measure(hr_y, sr_y, None),
        'rgb': _measure(hr[region].astype(np.float64),
                        sr[region].astype(np.float64), -1),
    }
