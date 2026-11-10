"""Float-only enlargement using Keys bicubic and symmetric boundaries."""
import numpy as np
from numpy.typing import NDArray


def _weights(length: int, scale: float) -> tuple[NDArray, NDArray]:
    """Map 1-based output centers to normalized symmetric input samples."""
    centers = np.arange(1, int(np.ceil(length*scale)) + 1)/scale + .5*(1 - 1/scale)
    left = np.floor(centers - 2)
    indices = left[:, None] + np.arange(6)
    distance = np.abs(centers[:, None] - indices)
    weights = np.where(distance <= 1, 1.5*distance**3 - 2.5*distance**2 + 1,
                       np.where(distance <= 2, -.5*distance**3
                                + 2.5*distance**2 - 4*distance + 2, 0))
    weights /= weights.sum(axis=1, keepdims=True)
    folded = (indices.astype(np.int64) - 1) % (2*length)
    indices = np.where(folded < length, folded, 2*length - 1 - folded)
    return weights, indices


def imresize_bicubic_float(image: NDArray, scale: float = 4) -> NDArray:
    """Enlarge a finite 2-D float64 array, rows first, without quantization."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.float64
            or image.ndim != 2 or min(image.shape) < 1
            or not np.isfinite(image).all()):
        raise ValueError('Expected a nonempty finite 2-D float64 array.')
    if not np.isfinite(scale) or scale < 1:
        raise ValueError('Only enlargement with finite scale >= 1 is supported.')
    result = image
    for axis in (0, 1):
        weights, indices = _weights(result.shape[axis], scale)
        source = np.moveaxis(result, axis, 0)
        result = np.moveaxis(np.sum(source[indices]*weights[..., None], axis=1),
                             0, axis)
    return result
