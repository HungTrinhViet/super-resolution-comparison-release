"""Strict RGB image I/O without automatic reference corrections."""
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def validate_rgb(image: NDArray) -> None:
    """Require a nonempty HWC RGB uint8 array."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError('Expected a uint8 NumPy RGB array.')
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError('Expected nonempty HWC RGB shape.')


def load_rgb(path: Path) -> NDArray[np.uint8]:
    """Read RGB only; reject mode conversion and EXIF rotation."""
    with Image.open(path) as image:
        if image.mode != 'RGB':
            raise ValueError(f'{path}: expected RGB, found {image.mode}.')
        if image.getexif().get(274, 1) != 1:
            raise ValueError(f'{path}: EXIF orientation requires review.')
        return np.array(image, dtype=np.uint8)


def load_pair(hr_path: Path, lr_path: Path) -> tuple[NDArray, NDArray]:
    """Load an unchanged benchmark pair and require exact x4 dimensions."""
    hr, lr = load_rgb(hr_path), load_rgb(lr_path)
    if hr.shape != (lr.shape[0] * 4, lr.shape[1] * 4, 3):
        raise ValueError(f'Pair shape mismatch: HR={hr.shape}, LR={lr.shape}.')
    return hr, lr


def save_rgb(path: Path, image: NDArray) -> None:
    """Write a PNG exclusively, preserving existing files."""
    validate_rgb(image)
    with path.open('xb') as stream:
        Image.fromarray(image).save(stream, format='PNG')
