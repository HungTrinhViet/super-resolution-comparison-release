"""Independent float64 EDSR equations; weights enter only through NumPy NPZ."""
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def conv3(image: NDArray, weights: NDArray, bias: NDArray) -> NDArray:
    """Cross-correlate CHW with OIHW kernels and one-pixel zero padding."""
    padded = np.pad(image, ((0, 0), (1, 1), (1, 1)))
    patches = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))
    return np.einsum('ockl,chwkl->ohw', weights, patches, optimize=True) + bias[:, None, None]


def pixel_shuffle(image: NDArray, scale: int = 2) -> NDArray:
    """Apply out[c,h*r+i,w*r+j]=in[c*r*r+i*r+j,h,w] directly."""
    channels, height, width = image.shape
    result = np.empty((channels//(scale*scale), height*scale, width*scale),
                      dtype=image.dtype)
    for channel in range(result.shape[0]):
        for row in range(scale):
            for column in range(scale):
                result[channel, row::scale, column::scale] = image[
                    channel*scale*scale + row*scale + column]
    return result


def reference_forward(path: Path, rgb: NDArray) -> tuple[NDArray, NDArray]:
    """Compute HWC 0..255 float output and half-even uint8 without clipping early."""
    with np.load(path, allow_pickle=False) as archive:
        state = {name: archive[name].astype(np.float64) for name in archive.files}

    def convolution(image: NDArray, name: str) -> NDArray:
        return conv3(image, state[name+'.weight'], state[name+'.bias'])

    mean = np.array([.4488, .4371, .4040])[:, None, None]
    x = (rgb.transpose(2, 0, 1).astype(np.float64)/255 - mean)*255
    head = convolution(x, 'conv_first')
    body = head
    for block in range(16):
        residual = convolution(np.maximum(convolution(body, f'body.{block}.conv1'), 0),
                               f'body.{block}.conv2')
        body = body + residual
    body = convolution(body, 'conv_after_body') + head
    body = pixel_shuffle(convolution(body, 'upsample.0'))
    body = pixel_shuffle(convolution(body, 'upsample.2'))
    result = (convolution(body, 'conv_last')/255 + mean)*255
    result = result.transpose(1, 2, 0)
    return result, np.rint(np.clip(result, 0, 255)).astype(np.uint8)
