"""Synthetic run-core checks without checkpoint inference or benchmark data."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_synthetic_output_and_fixed_timing(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('run_edsr_test', 'scripts/run_edsr.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    hr = np.full((24, 28, 3), 128, dtype=np.uint8)
    lr = np.full((6, 7, 3), 128, dtype=np.uint8)
    fake = SimpleNamespace(rgb=hr, prequant_float=hr.astype(np.float32),
                           half_tie_count=0, half_tie_channel_count=0)
    monkeypatch.setattr(script, 'super_resolve', lambda model, image: fake)
    calls = []

    def measure(operation, warmup, repeats):
        calls.append((warmup, repeats))
        operation()
        return {'warmup': warmup, 'repeats': repeats, 'samples_ms': [1.]*10,
                'median_ms': 1., 'iqr_ms': 0.}

    monkeypatch.setattr(script, 'measure', measure)
    output = tmp_path/'synthetic_only'
    result = script.run_pairs([('synthetic', hr, lr)], output,
                               torch.nn.Identity(), {'synthetic_unit_test': True})
    assert calls == [(3, 10), (3, 10)]
    saved = json.loads((output/'metrics.json').read_text())
    assert saved['images'][0]['y']['psnr'] is None
    assert saved['images'][0]['y']['identical_images'] is True
    assert result['images'][0]['half_tie_pixels_any_channel'] == 0
    assert result['images'][0]['clip_below_zero_percent'] == 0
    assert (output/'synthetic_SR.png').is_file()
    assert (output/'metrics.csv').is_file()
    assert (output/'environment.json').is_file()
    with pytest.raises(FileExistsError):
        script.run_pairs([('synthetic', hr, lr)], output, torch.nn.Identity(), {})
