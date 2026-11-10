"""Specified cubic resizing; no claim of independent MATLAB validation."""
import math

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .io import validate_rgb


def cubic_kernel(distance: NDArray) -> NDArray[np.float64]:
    """Keys cubic kernel a=-0.5 with support [-2, 2]."""
    x = np.abs(np.asarray(distance, dtype=np.float64))
    return np.where(x <= 1, 1.5 * x**3 - 2.5 * x**2 + 1,
                    np.where(x <= 2, -.5 * x**3 + 2.5 * x**2 - 4 * x + 2, 0))


def quantize(image: NDArray) -> NDArray[np.uint8]:
    """Clip and round half up once, after interpolation."""
    if not np.isfinite(image).all():
        raise ValueError('Cannot quantize nonfinite values.')
    return np.floor(np.clip(image, 0, 255) + .5).astype(np.uint8)


def _resize_axis(image: NDArray, scale: float, axis: int) -> NDArray:
    length = image.shape[axis]
    output_length = math.ceil(length * scale)
    kernel_scale = min(scale, 1.0)
    width = 4 / kernel_scale
    center = (np.arange(output_length) + .5) / scale - .5
    left = np.floor(center - width / 2).astype(np.int64)
    indices = left[:, None] + np.arange(math.ceil(width) + 2)[None, :]
    weights = kernel_scale * cubic_kernel(
        (center[:, None] - indices) * kernel_scale)
    weights /= weights.sum(axis=1, keepdims=True)
    # Symmetric extension includes the boundary sample, unlike reflect padding.
    mirrored = indices % (2 * length)
    mirrored = np.where(mirrored < length, mirrored, 2 * length - 1 - mirrored)
    pixels = np.take(np.moveaxis(image, axis, 0), mirrored, axis=0)
    shape = weights.shape + (1,) * (pixels.ndim - 2)
    output = np.sum(pixels * weights.reshape(shape), axis=1)
    return np.moveaxis(output, 0, axis)


def resize(image: NDArray, scale: float, backend: str = 'numpy') -> NDArray:
    """Resize RGB with center alignment; NumPy shrinks with antialiasing."""
    validate_rgb(image)
    if isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Scale must be positive and finite.')
    if backend == 'pillow':
        size = (math.ceil(image.shape[1] * scale),
                math.ceil(image.shape[0] * scale))
        return np.asarray(Image.fromarray(image).resize(
            size, Image.Resampling.BICUBIC), dtype=np.float64)
    if backend != 'numpy':
        raise ValueError(f'Unknown backend: {backend}.')
    intermediate = _resize_axis(image.astype(np.float64), scale, 0)
    return _resize_axis(intermediate, scale, 1)


def upscale(image: NDArray, backend: str = 'numpy') -> NDArray[np.uint8]:
    """Run x4 inference including final uint8 quantization."""
    return quantize(resize(image, 4, backend))
