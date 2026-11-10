"""Nonvisual registry checks using synthetic RGB only, with CPU subprocesses."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import weakref

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_fresh(source, tmp_path):
    """Keep startup thread settings independent of older test fixtures."""
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'src')
    completed = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(source)], cwd=tmp_path,
        env=env, capture_output=True, text=True, encoding='utf-8', timeout=180)
    print(completed.stdout, end='')
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def test_algorithms_and_bicubic_metadata():
    from sr_compare import bicubic, inference

    assert inference.ALGORITHMS == ('Bicubic', 'SRCNN', 'FSRCNN', 'EDSR', 'ESRGAN')
    lr = np.random.default_rng(10).integers(0, 256, (8, 9, 3), dtype=np.uint8)
    result = inference.InferenceRegistry().predict('Bicubic', lr)
    assert isinstance(result, inference.InferenceResult)
    np.testing.assert_array_equal(result.rgb, bicubic.upscale(lr, backend='numpy'))
    assert result.algorithm == 'Bicubic'
    assert result.checkpoint_sha256 is None
    assert result.source_notice == ''


def test_thread_configuration_is_once_at_process_startup(tmp_path):
    run_fresh('''
        from sr_compare.inference import configure_threads
        import torch
        calls = {'intra': 0, 'inter': 0}
        original_intra = torch.set_num_threads
        original_inter = torch.set_num_interop_threads
        def intra(value):
            calls['intra'] += 1
            original_intra(value)
        def inter(value):
            calls['inter'] += 1
            original_inter(value)
        torch.set_num_threads = intra
        torch.set_num_interop_threads = inter
        configure_threads()
        configure_threads()
        assert calls == {'intra': 1, 'inter': 1}, calls
        assert torch.get_num_threads() == 4
        assert torch.get_num_interop_threads() == 1
        print('thread configuration: setters once each; intra=4 inter=1')
    ''', tmp_path)


def test_all_five_match_direct_library_calls_exactly(tmp_path):
    output = run_fresh('''
        import gc
        import json
        from pathlib import Path
        from sr_compare.inference import configure_threads
        configure_threads()
        import numpy as np
        from sr_compare import bicubic, srcnn, fsrcnn, edsr, esrgan, inference

        root = Path(inference.__file__).resolve().parents[2]
        lr = np.random.default_rng(10).integers(0, 256, (8, 9, 3), dtype=np.uint8)
        specifications = (
            ('Bicubic', None, None, None),
            ('SRCNN', srcnn, srcnn.load_srcnn,
             Path('data/checkpoints/srcnn/SRCNN/model/9-5-5(ImageNet)/x4.mat')),
            ('FSRCNN', fsrcnn, fsrcnn.load_fsrcnn, fsrcnn.CHECKPOINT),
            ('EDSR', edsr, edsr.load_edsr, edsr.CHECKPOINT),
            ('ESRGAN', esrgan, esrgan.load_esrgan, esrgan.CHECKPOINT),
        )
        registry = inference.InferenceRegistry()
        for algorithm, module, loader, path in specifications:
            registry.release()
            if module is None:
                expected = bicubic.upscale(lr, backend='numpy')
            else:
                model = loader(root / path)
                expected = module.super_resolve(model, lr).rgb
                del model
                gc.collect()
            result = registry.predict(algorithm, lr)
            diff = np.abs(result.rgb.astype(np.int16) - expected.astype(np.int16))
            record = {'algorithm': algorithm,
                      'pixel_count': int(np.prod(expected.shape[:2])),
                      'different_pixels': int(np.any(diff != 0, axis=2).sum()),
                      'max_diff': int(diff.max())}
            print(json.dumps(record), flush=True)
            assert result.rgb.shape == (32, 36, 3)
            assert result.rgb.dtype == np.uint8
            assert result.algorithm == algorithm
            assert record['different_pixels'] == 0
            assert record['max_diff'] == 0
            if module is not None:
                assert result.checkpoint_sha256 == module.CHECKPOINT_SHA256
                assert result.source_notice == getattr(module, 'SOURCE_NOTICE', '')
        registry.release()
    ''', tmp_path)
    records = [json.loads(line) for line in output.splitlines()]
    assert [record['algorithm'] for record in records] == [
        'Bicubic', 'SRCNN', 'FSRCNN', 'EDSR', 'ESRGAN']


def test_cache_reuses_one_model_releases_before_next_load(monkeypatch):
    from sr_compare import inference

    class Model:
        pass

    references = []
    loads = []
    output = np.zeros((32, 36, 3), dtype=np.uint8)

    def load(path):
        assert all(reference() is None for reference in references)
        model = Model()
        references.append(weakref.ref(model))
        loads.append(path)
        return model

    for module, name in ((inference.srcnn, 'load_srcnn'),
                         (inference.fsrcnn, 'load_fsrcnn')):
        monkeypatch.setattr(module, name, load)
        monkeypatch.setattr(module, 'super_resolve',
                            lambda model, lr: SimpleNamespace(rgb=output))
    registry = inference.InferenceRegistry()
    assert loads == []
    lr = np.zeros((8, 9, 3), dtype=np.uint8)
    first = registry.predict('SRCNN', lr)
    assert first.rgb is output
    registry.predict('SRCNN', lr)
    assert len(loads) == 1
    registry.predict('FSRCNN', lr)
    assert len(loads) == 2
    assert references[0]() is None
    assert references[1]() is not None
    collected = []
    monkeypatch.setattr(inference.gc, 'collect', lambda: collected.append(True))
    registry.release()
    assert references[1]() is None
    assert collected == [True]
    registry.predict('SRCNN', lr)
    registry.predict('Bicubic', lr)
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize('algorithm', ['SRCNN', 'FSRCNN', 'EDSR', 'ESRGAN'])
def test_missing_checkpoint_has_recovery_instructions(algorithm, tmp_path, monkeypatch):
    from sr_compare import inference

    monkeypatch.setattr(inference, 'ROOT', tmp_path)
    with pytest.raises(FileNotFoundError, match=r'scripts/download_models\.py') as caught:
        inference.InferenceRegistry().predict(algorithm, np.zeros((8, 9, 3), np.uint8))
    if algorithm == 'FSRCNN':
        assert 'manual' in str(caught.value).lower()
        assert 'README' in str(caught.value)
        assert 'no automatic' in str(caught.value).lower()


@pytest.mark.parametrize('algorithm,module_name', [
    ('SRCNN', 'srcnn'), ('FSRCNN', 'fsrcnn'), ('EDSR', 'edsr'), ('ESRGAN', 'esrgan')])
def test_wrong_hash_is_rejected_by_existing_loader(
        algorithm, module_name, tmp_path, monkeypatch):
    from sr_compare import inference

    monkeypatch.setattr(inference, 'ROOT', tmp_path)
    relative = (Path('data/checkpoints/srcnn/SRCNN/model/9-5-5(ImageNet)/x4.mat')
                if algorithm == 'SRCNN' else getattr(inference, module_name).CHECKPOINT)
    checkpoint = tmp_path / relative
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'not the locked checkpoint')
    with pytest.raises(ValueError, match='SHA-256'):
        inference.InferenceRegistry().predict(algorithm, np.zeros((8, 9, 3), np.uint8))


def test_inference_value_errors_propagate(monkeypatch):
    from sr_compare import inference

    monkeypatch.setattr(inference.srcnn, 'load_srcnn', lambda path: object())
    error = ValueError('nonfinite library output')

    def reject(model, lr):
        raise error

    monkeypatch.setattr(inference.srcnn, 'super_resolve', reject)
    with pytest.raises(ValueError) as caught:
        inference.InferenceRegistry().predict('SRCNN', np.zeros((8, 9, 3), np.uint8))
    assert caught.value is error


@pytest.mark.parametrize('lr,error', [
    (np.zeros((8, 9), np.uint8), ValueError),
    (np.zeros((8, 9, 4), np.uint8), ValueError),
    (np.zeros((0, 9, 3), np.uint8), ValueError),
    (np.zeros((8, 9, 3), np.float32), TypeError),
])
def test_invalid_rgb_is_rejected(lr, error):
    from sr_compare.inference import InferenceRegistry

    with pytest.raises(error):
        InferenceRegistry().predict('Bicubic', lr)


def test_unknown_algorithm_is_rejected():
    from sr_compare.inference import InferenceRegistry

    with pytest.raises(ValueError, match='Unknown algorithm'):
        InferenceRegistry().predict('Other', np.zeros((8, 9, 3), np.uint8))
