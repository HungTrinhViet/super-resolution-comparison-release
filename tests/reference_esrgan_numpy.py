"""Independent float64 Original ESRGAN equations; converted weights provenance unverified."""
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray


def load_weights(path: Path) -> dict[str, NDArray]:
    """Read a temporary numerical archive without pickle or model imports."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].astype(np.float64) for key in archive.files}


def conv(image: NDArray, weight: NDArray, bias: NDArray) -> NDArray:
    """Cross-correlate CHW with OIHW kernels and one-pixel zero padding."""
    patches = sliding_window_view(np.pad(image, ((0, 0), (1, 1), (1, 1))),
                                  (3, 3), axis=(1, 2))
    return np.einsum('oikl,ihwkl->ohw', weight, patches,
                     optimize=True) + bias[:, None, None]


def leaky(image: NDArray) -> NDArray:
    """Apply slope 0.2 to negative features without modifying the input."""
    return np.maximum(image, 0) + .2*np.minimum(image, 0)


def dense(image: NDArray, weights: dict[str, NDArray], prefix: str) -> NDArray:
    """Concatenate original input before every newly computed dense feature."""
    features = [image]
    for index in range(1, 6):
        key = f'{prefix}.conv{index}'
        value = conv(np.concatenate(features, axis=0),
                     weights[key+'.weight'], weights[key+'.bias'])
        if index < 5:
            features.append(leaky(value))
    return image + .2*value


def reference_forward(weights: dict[str, NDArray], rgb: NDArray) -> NDArray:
    """Return unclamped RGB HWC intensities on the 0..255 scale."""
    def convolution(image: NDArray, key: str) -> NDArray:
        return conv(image, weights[key+'.weight'], weights[key+'.bias'])

    head = convolution(rgb.astype(np.float64).transpose(2, 0, 1)/255,
                       'conv_first')
    body = head.copy()
    for block in range(23):
        identity = body
        for index in range(1, 4):
            body = dense(body, weights, f'body.{block}.rdb{index}')
        body = identity + .2*body
    body = head + convolution(body, 'conv_body')
    for index in (1, 2):
        body = leaky(convolution(np.repeat(np.repeat(body, 2, axis=1), 2, axis=2),
                                 f'conv_up{index}'))
    output = convolution(leaky(convolution(body, 'conv_hr')), 'conv_last')
    return output.transpose(1, 2, 0)*255
