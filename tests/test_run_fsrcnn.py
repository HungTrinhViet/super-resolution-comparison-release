"""Synthetic CLI-core output and fixed-timing checks without author weights."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_synthetic_run_output_and_no_overwrite(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('run_fsrcnn_test',
                                                 Path('scripts/run_fsrcnn.py'))
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    hr = np.full((24, 28, 3), 128, dtype=np.uint8)
    lr = np.full((6, 7, 3), 128, dtype=np.uint8)
    fake = SimpleNamespace(rgb=hr, y_float=np.full((24, 28), .5), y_uint8=np.full((24, 28), 126, dtype=np.uint8))
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
    captured = json.loads((output / 'environment.json').read_text())
    assert captured['metrics_run_start_utc'] == result['started_utc']
    assert captured['source_notice'] == script.SOURCE_NOTICE
    np.testing.assert_array_equal(script.prepare_forward(lr).numpy(),
                                   np.full((1, 1, 6, 7), 126/255, np.float32))
    with pytest.raises(FileExistsError):
        script.run_pairs([('synthetic', hr, lr)], output,
                          torch.nn.Identity(), {})


def test_lock_validation(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('run_fsrcnn_lock',
                                                 Path('scripts/run_fsrcnn.py'))
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    source, checkpoint = tmp_path/'source.py', tmp_path/'checkpoint.mat'
    source.write_bytes(b'fixed source')
    checkpoint.write_bytes(b'fixed checkpoint')
    document, receipt_path = tmp_path/'results.md', tmp_path/'lock.json'
    document.write_text('<!-- LOCKED_CONFIG_BEGIN -->fixed<!-- LOCKED_CONFIG_END -->')
    digest = lambda path: script.hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {'locked_utc': '2026-11-09T00:00:00+00:00',
               'source_sha256': {str(source): digest(source)},
               'config_sha256': script.hashlib.sha256(b'fixed').hexdigest(),
               'checkpoint_sha256': digest(checkpoint), 'pip_freeze': ['test==1'],
               'benchmark': script.BENCHMARK, 'preprocessing': script.PREPROCESSING,
               'hardware': {'synthetic_hardware': True}}
    monkeypatch.setattr(script, 'LOCK_DOCUMENT', document)
    monkeypatch.setattr(script, 'LOCK_RECEIPT', receipt_path)
    monkeypatch.setattr(script, 'LOCK_SOURCES', (str(source),))
    monkeypatch.setattr(script, 'CHECKPOINT', checkpoint)
    monkeypatch.setattr(script, 'CHECKPOINT_SHA256', digest(checkpoint))
    monkeypatch.setattr(script.subprocess, 'check_output', lambda *a, **k: 'test==1\n')
    monkeypatch.setattr(script.torch, 'get_num_threads', lambda: 4)
    monkeypatch.setattr(script.torch, 'get_num_interop_threads', lambda: 1)
    monkeypatch.setattr(script, 'hardware_snapshot', lambda: {'synthetic_hardware': True})
    receipt_path.write_text(json.dumps(receipt))
    assert script.verify_lock() == receipt
    receipt['hardware'] = {'synthetic_hardware': False}
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='hardware'):
        script.verify_lock()
    receipt['hardware'] = {'synthetic_hardware': True}
    receipt_path.write_text(json.dumps(receipt))
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source changed'):
        script.verify_lock()
    source.write_bytes(b'fixed source')
    document.write_text('<!-- LOCKED_CONFIG_BEGIN -->changed<!-- LOCKED_CONFIG_END -->')
    with pytest.raises(ValueError, match='configuration section changed'):
        script.verify_lock()
