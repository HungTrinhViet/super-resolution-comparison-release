"""Direct float64 translation of SRCNN.m for independent test comparisons."""
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import correlate


def reference_forward(path: Path, image: np.ndarray) -> np.ndarray:
    """Apply author MATLAB layouts and nearest-boundary correlations."""
    arrays = loadmat(path)
    first = np.zeros((*image.shape, 64), dtype=np.float64)
    for output in range(64):
        kernel = arrays['weights_conv1'][:, output].reshape(9, 9, order='F')
        first[..., output] = np.maximum(correlate(image, kernel, mode='nearest')
                                        + arrays['biases_conv1'][output, 0], 0)
    second = np.zeros((*image.shape, 32), dtype=np.float64)
    for output in range(32):
        for channel in range(64):
            kernel = arrays['weights_conv2'][channel, :, output].reshape(
                5, 5, order='F')
            second[..., output] += correlate(first[..., channel], kernel,
                                              mode='nearest')
        second[..., output] = np.maximum(second[..., output]
                                          + arrays['biases_conv2'][output, 0], 0)
    result = np.zeros(image.shape, dtype=np.float64)
    for channel in range(32):
        kernel = arrays['weights_conv3'][channel, :].reshape(5, 5, order='F')
        result += correlate(second[..., channel], kernel, mode='nearest')
    return result + arrays['biases_conv3'][0, 0]
