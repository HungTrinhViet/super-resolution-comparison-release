"""Independent float64 translation of the reposted FSRCNN_test MATLAB equations.

Bản đăng lại của gói FSRCNN_test (Chao Dong, ECCV 2016); hai bản sao độc lập
trùng SHA-256; nguồn gốc trực tiếp từ CUHK và cách chọn checkpoint chưa xác
minh; chỉ dùng học thuật.
"""
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat
from scipy.ndimage import correlate


def scatter_buffer(features: NDArray, kernels: NDArray) -> NDArray:
    """Translate deconv.m scatter into its full, deliberately oversized buffer."""
    channels, height, width = features.shape
    padded = np.pad(features, ((0, 0), (1, 1), (1, 1)), mode='edge')
    output = np.zeros(((height+2)*4+8, (width+2)*4+8), dtype=np.float64)
    for channel in range(channels):
        for row in range(height+2):
            for column in range(width+2):
                output[row*4:row*4+9, column*4:column*4+9] += (
                    padded[channel, row, column]*kernels[channel])
    return output


def reference_forward(path: Path, image: NDArray) -> NDArray:
    """Read MAT directly and compute every output pixel, without torch imports."""
    arrays = loadmat(path)
    features = image.astype(np.float64)[..., None]
    for layer in range(7):
        weights = arrays['weights_conv'][layer, 0]
        bias = arrays['biases_conv'][layer, 0]
        slope = float(arrays['prelu_conv'][layer, 0][0, 0])
        channels, area, outputs = weights.shape
        size = int(np.sqrt(area))
        next_features = np.zeros((*image.shape, outputs), dtype=np.float64)
        for output in range(outputs):
            for channel in range(channels):
                kernel = weights[channel, :, output].reshape(size, size, order='F')
                next_features[..., output] += correlate(features[..., channel], kernel,
                                                         mode='nearest')
            z = next_features[..., output] + float(bias[output, 0])
            next_features[..., output] = np.maximum(z, 0) + slope*np.minimum(z, 0)
        features = next_features
    weights = arrays['weights_conv'][7, 0]
    kernels = np.stack([weights[0, :, channel].reshape(9, 9, order='F')
                        for channel in range(weights.shape[2])])
    full = scatter_buffer(features.transpose(2, 0, 1), kernels)
    height, width = image.shape
    return full[8:8+height*4, 8:8+width*4] + float(arrays['biases_conv'][7, 0][0, 0])
