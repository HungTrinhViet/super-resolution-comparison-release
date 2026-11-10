"""Synthetic CLI-core output and fixed-timing checks without author weights."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_synthetic_run_output_and_no_overwrite(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('run_srcnn_test',
                                                 Path('scripts/run_srcnn.py'))
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    hr = np.full((24, 28, 3), 128, dtype=np.uint8)
    lr = np.full((6, 7, 3), 128, dtype=np.uint8)
    fake = SimpleNamespace(rgb=hr, y_uint8=np.full((24, 28), 126, dtype=np.uint8))
    monkeypatch.setattr(script, 'super_resolve', lambda model, image: fake)
    calls = []

    def measure(operation, warmup, repeats):
        calls.append((warmup, repeats))
        operation()
        return {'warmup': warmup, 'repeats': repeats, 'samples_ms': [1.]*10,
                'median_ms': 1., 'iqr_ms': 0.}

    monkeypatch.setattr(script, 'measure', measure)
    output = tmp_path / 'synthetic_only'
    result = script.run_pairs([('synthetic', hr, lr)], output,
                               torch.nn.Identity(), {'synthetic_unit_test': True})
    assert calls == [(3, 10), (3, 10)]
    saved = json.loads((output / 'metrics.json').read_text())
    assert saved['images'][0]['y']['psnr'] is None
    assert saved['images'][0]['y']['identical_images'] is True
    assert result['images'][0]['y']['ssim'] == 1
    assert (output / 'synthetic_SR.png').is_file()
    assert (output / 'metrics.csv').is_file()
    assert (output / 'environment.json').is_file()
    with pytest.raises(FileExistsError):
        script.run_pairs([('synthetic', hr, lr)], output,
                          torch.nn.Identity(), {})
