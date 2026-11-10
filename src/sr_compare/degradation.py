"""Explicit degradation for standalone HR, never benchmark pair repair."""
import logging

from numpy.typing import NDArray

from .bicubic import quantize, resize
from .io import validate_rgb

LOGGER = logging.getLogger(__name__)


def modcrop(image: NDArray) -> NDArray:
    """Discard bottom/right remainders for dimensions divisible by four."""
    validate_rgb(image)
    height, width = image.shape[:2]
    if min(height, width) < 4:
        raise ValueError('Standalone HR must be at least 4x4.')
    cropped = image[:height - height % 4, :width - width % 4].copy()
    LOGGER.info('Standalone HR modcrop: %s -> %s (bottom/right)',
                image.shape, cropped.shape)
    return cropped


def make_lr(image: NDArray, backend: str = 'numpy') -> NDArray:
    """Generate x4-degraded LR only for the standalone-HR workflow."""
    return quantize(resize(modcrop(image), .25, backend))
