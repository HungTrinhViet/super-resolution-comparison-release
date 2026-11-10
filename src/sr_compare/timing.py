"""Repeated CPU wall-clock timing, excluding input/output and metrics."""
from collections.abc import Callable
from time import perf_counter_ns

import numpy as np


def validate_timing(warmup: int, repeats: int) -> None:
    """Reject invalid repetition settings before creating experiment output."""
    if (type(warmup) is not int or warmup < 0
            or type(repeats) is not int or repeats < 1):
        raise ValueError('Warmup must be >=0 and repeats >=1, both integers.')


def measure(operation: Callable[[], object], warmup: int = 3,
            repeats: int = 30) -> dict:
    """Time one complete inference call and report milliseconds and IQR."""
    validate_timing(warmup, repeats)
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(repeats):
        start = perf_counter_ns()
        operation()
        samples.append((perf_counter_ns() - start) / 1_000_000)
    quartiles = np.percentile(samples, [25, 75])
    return {'warmup': warmup, 'repeats': repeats, 'samples_ms': samples,
            'median_ms': float(np.median(samples)),
            'iqr_ms': float(quartiles[1] - quartiles[0])}
