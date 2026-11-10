"""Synthetic-only runner and lock tests; no real checkpoint or Set5 inference."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def runner():
    spec = importlib.util.spec_from_file_location('run_esrgan_test', 'scripts/run_esrgan.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_outputs_and_no_overwrite(tmp_path, monkeypatch):
    script = runner()
    hr = np.full((24, 28, 3), 128, dtype=np.uint8)
    lr = np.full((6, 7, 3), [16, 128, 240], dtype=np.uint8)
    values = hr.astype(np.float32)
    values[0, 0] = [-1, 255.5, 128.5]
    fake = SimpleNamespace(rgb=hr, prequant_float=values)
    monkeypatch.setattr(script, 'super_resolve', lambda model, image: fake)
    calls = []

    def measure(operation, warmup, repeats):
        calls.append((warmup, repeats))
        operation()
        return {'warmup': warmup, 'repeats': repeats, 'samples_ms': [1.]*10,
                'median_ms': 1., 'iqr_ms': 0.}

    monkeypatch.setattr(script, 'measure', measure)
    output = tmp_path/'synthetic_only'
    report = script.run_pairs([('synthetic', hr, lr)], output,
                              torch.nn.Identity(), {'synthetic_unit_test': True})
    assert calls == [(3, 10), (3, 10)]
    saved = json.loads((output/'metrics.json').read_text())
    assert saved['images'][0]['y']['psnr'] is None
    assert saved['images'][0]['y']['identical_images'] is True
    assert report['images'][0]['half_tie_channel_values'] == 2
    assert report['images'][0]['half_tie_pixels_any_channel'] == 1
    np.testing.assert_array_equal(np.load(output/'synthetic_prequant.npy', allow_pickle=False), values)
    captured = json.loads((output/'environment.json').read_text())
    assert captured['metrics_run_start_utc'] == report['started_utc']
    assert captured['source_notice'] == script.SOURCE_NOTICE
    assert (output/'metrics.csv').is_file()
    assert (output/'synthetic_SR.png').is_file()
    with pytest.raises(FileExistsError):
        script.run_pairs([('synthetic', hr, lr)], output, torch.nn.Identity(), {})


def test_receipt_validation(tmp_path, monkeypatch):
    script = runner()
    source, checkpoint = tmp_path/'source.py', tmp_path/'checkpoint.pth'
    source.write_bytes(b'fixed source')
    checkpoint.write_bytes(b'fixed checkpoint')
    document, receipt_path = tmp_path/'results.md', tmp_path/'receipt.json'
    document.write_text('<!-- LOCKED_CONFIG_BEGIN -->fixed<!-- LOCKED_CONFIG_END -->')
    digest = lambda p: script.hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = {'locked_utc': '2026-11-09T00:00:00+00:00',
               'source_sha256': {str(source): digest(source)},
               'config_sha256': script.hashlib.sha256(b'fixed').hexdigest(),
               'checkpoint_sha256': digest(checkpoint), 'pip_freeze': ['test==1'],
               'benchmark': script.BENCHMARK, 'preprocessing': script.PREPROCESSING,
               'hardware': {'unit_test': True}}
    monkeypatch.setattr(script, 'LOCK_DOCUMENT', document)
    monkeypatch.setattr(script, 'LOCK_RECEIPT', receipt_path)
    monkeypatch.setattr(script, 'LOCK_SOURCES', (str(source),))
    monkeypatch.setattr(script, 'CHECKPOINT', checkpoint)
    monkeypatch.setattr(script, 'CHECKPOINT_SHA256', digest(checkpoint))
    monkeypatch.setattr(script.subprocess, 'check_output', lambda *a, **k: 'test==1\n')
    monkeypatch.setattr(script, 'hardware_snapshot', lambda: {'unit_test': True})
    monkeypatch.setattr(script.torch, 'get_num_threads', lambda: 4)
    monkeypatch.setattr(script.torch, 'get_num_interop_threads', lambda: 1)
    receipt_path.write_text(json.dumps(receipt))
    assert script.verify_lock() == receipt
    for field, changed in [('hardware', {'unit_test': False}),
                           ('pip_freeze', ['test==2']), ('benchmark', {})]:
        wrong = {**receipt, field: changed}
        receipt_path.write_text(json.dumps(wrong))
        with pytest.raises(ValueError):
            script.verify_lock()
    receipt_path.write_text(json.dumps(receipt))
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source'):
        script.verify_lock()
    source.write_bytes(b'fixed source')
    document.write_text('<!-- LOCKED_CONFIG_BEGIN -->changed<!-- LOCKED_CONFIG_END -->')
    with pytest.raises(ValueError, match='configuration'):
        script.verify_lock()
