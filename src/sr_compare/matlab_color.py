"""BT.601 studio-range transforms on the 0..255 scale, per documentation."""
import numpy as np
from numpy.typing import NDArray

MATRIX = np.array([[65.481, 128.553, 24.966],
                   [-37.797, -74.203, 112.0],
                   [112.0, -93.786, -18.214]], dtype=np.float64)/255
OFFSET = np.array([16., 128., 128.])
INVERSE = np.linalg.inv(MATRIX)


def _color_values(image: NDArray) -> NDArray:
    """Require finite triples; float transforms preserve out-of-range values."""
    values = np.asarray(image, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3 or values.size == 0:
        raise ValueError('Expected nonempty color triples.')
    if not np.isfinite(values).all():
        raise ValueError('Color values must be finite.')
    return values


def matlab_uint8_round(values: NDArray) -> NDArray:
    """Round half away from zero, then saturate; reject NaN and infinity."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Quantization values must be finite.')
    rounded = np.copysign(np.floor(np.abs(values) + .5), values)
    return np.clip(rounded, 0, 255).astype(np.uint8)


def rgb2ycbcr_float(image: NDArray) -> NDArray:
    """Convert RGB 0..255 to unquantized BT.601 YCbCr on the same scale."""
    return _color_values(image) @ MATRIX.T + OFFSET


def ycbcr2rgb_float(image: NDArray) -> NDArray:
    """Invert the forward matrix exactly, without clipping or quantizing."""
    return (_color_values(image) - OFFSET) @ INVERSE.T


def rgb2ycbcr_uint8(image: NDArray) -> NDArray:
    """Convert uint8 RGB to uint8 YCbCr with documented rounding."""
    if np.asarray(image).dtype != np.uint8:
        raise ValueError('Expected uint8 RGB.')
    return matlab_uint8_round(rgb2ycbcr_float(image))


def ycbcr2rgb_uint8(image: NDArray) -> NDArray:
    """Convert uint8 YCbCr to saturated uint8 RGB."""
    if np.asarray(image).dtype != np.uint8:
        raise ValueError('Expected uint8 YCbCr.')
    return matlab_uint8_round(ycbcr2rgb_float(image))
