"""G-T1..G-T7 for Original ESRGAN architecture and converted BasicSR checkpoint."""
import ast
import ctypes
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.nn import functional as functional

from reference_esrgan_numpy import load_weights, reference_forward
from sr_compare.esrgan import (
    CHECKPOINT, SOURCE_NOTICE, Rdb, Rrdb, RrdbNet, load_esrgan,
    prepare_input, quantize_values, super_resolve,
)


@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(4)
    print(SOURCE_NOTICE)
    return load_esrgan()


def samples():
    """Return exactly six predetermined small RGB patterns and archive crops."""
    gradient = np.arange(7*9).reshape(7, 9)*3
    checker = (np.indices((6, 8)).sum(axis=0) % 2).astype(np.uint8)*255
    result = {
        'noise': np.random.default_rng(6).integers(0, 256, (6, 7, 3), dtype=np.uint8),
        'gradient': np.stack([gradient, 255-gradient, gradient//2], -1).astype(np.uint8),
        'checker': np.stack([checker, 255-checker, checker//2], -1),
        'constant': np.full((8, 9, 3), [32, 128, 224], dtype=np.uint8),
    }
    with zipfile.ZipFile('data/quarantine/srcnn/SRCNN_v1.zip') as archive:
        for name in ('zebra', 'monarch'):
            with Image.open(BytesIO(archive.read(f'SRCNN/Set14/{name}.bmp'))) as image:
                assert image.mode == 'RGB'
                result[name] = np.array(image)[30:36, 41:48].copy()
    return result


def peak_ram_bytes():
    """Query lifetime peak working set using the existing Windows process API."""
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong),
                    *[(key, ctypes.c_size_t) for key in (
                        'peak', 'working', 'peak_pool_paged', 'pool_paged',
                        'peak_pool_nonpaged', 'pool_nonpaged', 'pagefile', 'peak_pagefile')]]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.argtypes = []
    get_process.restype = ctypes.c_void_p
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    get_memory.restype = ctypes.c_int
    if not get_memory(get_process(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return int(counters.peak)


def differences(actual, expected):
    error = np.abs(actual.astype(float)-expected)
    a = np.rint(np.clip(actual, 0, 255)).astype(np.uint8)
    b = np.rint(np.clip(expected, 0, 255)).astype(np.uint8)
    quantized = np.abs(a.astype(int)-b.astype(int))
    return {'float_max': float(error.max()), 'float_mean': float(error.mean()),
            'uint8_max': int(quantized.max()),
            'different_pixels': int(np.any(quantized != 0, axis=-1).sum()),
            'different_pixel_percent': float(100*np.mean(np.any(quantized != 0, axis=-1))),
            'channels': {name: {'float_max': float(error[..., i].max()),
                                'float_mean': float(error[..., i].mean()),
                                'uint8_max': int(quantized[..., i].max()),
                                'different_values': int((quantized[..., i] != 0).sum())}
                         for i, name in enumerate('RGB')}}


def accepted(stats):
    return (stats['float_max'] <= .02 and stats['float_mean'] <= .002
            and stats['uint8_max'] <= 1 and stats['different_pixel_percent'] <= .5)


def report_path(name):
    """Keep numerical evidence inside the project, alongside pytest basetemp."""
    directory = Path(__file__).resolve().parents[1]/'logs'
    directory.mkdir(exist_ok=True)
    return directory/name


def test_g_t1_schema_and_hash(model, tmp_path, monkeypatch):
    assert len(model.state_dict()) == 702
    assert sum(p.numel() for p in model.parameters()) == 16697987
    assert len(model.body) == 23
    assert sum(isinstance(m, Rdb) for m in model.modules()) == 69
    assert not any(isinstance(m, (nn.BatchNorm2d, nn.PReLU)) for m in model.modules())
    path = tmp_path/'wrong.pth'
    path.write_bytes(b'wrong checkpoint')
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Hash must precede torch.load'))
    with pytest.raises(ValueError, match='SHA-256'):
        load_esrgan(path)


@pytest.fixture(scope='module')
def comparisons(model, tmp_path_factory):
    path = tmp_path_factory.mktemp('esrgan_reference')/'weights.npz'
    state = torch.load(CHECKPOINT, weights_only=True, map_location='cpu')['params']
    np.savez(path, **{key: value.numpy() for key, value in state.items()})
    weights = load_weights(path)
    images, expected, reports = samples(), {}, {}
    started = time.perf_counter()
    reference_seconds = 0.
    for name, image in images.items():
        tick = time.perf_counter()
        expected[name] = reference_forward(weights, image)
        reference_seconds += time.perf_counter()-tick
        assert reference_seconds <= 7200, 'Locked NumPy reference budget exceeded'
        actual = super_resolve(model, image).prequant_float
        reports[name] = differences(actual, expected[name])
        print('G-T2', name, json.dumps(reports[name]), flush=True)
        assert accepted(reports[name]), f'Locked G-T2 threshold failed: {name}'
    result = {'source_notice': SOURCE_NOTICE, 'images': reports,
              'reference_seconds': reference_seconds,
              'wall_seconds_including_exported_weights_loaded': time.perf_counter()-started,
              'lifetime_peak_working_set_bytes': peak_ram_bytes()}
    report_path('m6_reference_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('G-T2 RESOURCES', json.dumps(result), flush=True)
    return images, expected


def test_g_t2_reference(comparisons):
    assert len(comparisons[0]) == 6


def test_g_t3_concat_and_orientation():
    block = Rdb()
    with torch.no_grad():
        for conv in (block.conv1, block.conv2, block.conv3, block.conv4, block.conv5):
            conv.weight.zero_()
            conv.bias.zero_()
        block.conv1.bias.fill_(1)
        block.conv2.bias.fill_(2)
        block.conv3.bias.fill_(3)
        block.conv4.bias.fill_(4)
        block.conv5.weight[0, 0, 0, 2] = 1
    captured = {}
    hooks = [getattr(block, f'conv{i}').register_forward_pre_hook(
        lambda module, args, index=i: captured.update({index: args[0].clone()}))
        for i in range(2, 6)]
    image = torch.zeros(1, 64, 5, 7)
    image[0, 0, 2, 3] = 1
    with torch.inference_mode():
        output = block(image)
    for hook in hooks:
        hook.remove()
    for i in range(2, 6):
        np.testing.assert_array_equal(captured[i][:, :64].numpy(), image.numpy())
        for feature in range(1, i):
            assert torch.all(captured[i][:, 64+(feature-1)*32:64+feature*32] == feature)
    expected = image.clone()
    expected[0, 0, 3, 2] += .2
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=0)


def test_g_t4_residual_and_activation():
    block = Rdb()
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
        block.conv5.bias.fill_(-2)
    image = torch.ones(1, 64, 3, 5)
    torch.testing.assert_close(block(image), image-.4, atol=1e-6, rtol=0)
    rrdb = Rrdb()
    class Add(nn.Module):
        def forward(self, x):
            return x+1
    rrdb.rdb1, rrdb.rdb2, rrdb.rdb3 = Add(), Add(), Add()
    torch.testing.assert_close(rrdb(image), image+.2*(image+3), atol=1e-6, rtol=0)
    np.testing.assert_allclose(block.lrelu(torch.tensor([-2., 2.])).numpy(), [-.4, 2.], atol=1e-6)


@pytest.mark.parametrize('shape', [(1, 1), (1, 5), (5, 1), (5, 7)])
def test_g_t5_geometry(model, shape):
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, *shape)).shape == (1, 3, shape[0]*4, shape[1]*4)
    values = torch.arange(shape[0]*shape[1]).reshape(1, 1, *shape).float()
    torch.testing.assert_close(functional.interpolate(values, scale_factor=2, mode='nearest'),
                               values.repeat_interleave(2, 2).repeat_interleave(2, 3), atol=0, rtol=0)


def test_g_t5_quantization_and_g_t7_adapter(model):
    np.testing.assert_array_equal(quantize_values(torch.tensor([-.1, .5, 1.5, 2.5, 254.5, 256.])).numpy(),
                                  [0, 0, 2, 2, 254, 255])
    for bad in (torch.tensor([float('nan')]), torch.tensor([float('inf')])):
        with pytest.raises(ValueError, match='finite'):
            quantize_values(bad)
    with pytest.raises(ValueError):
        prepare_input(np.zeros((3, 5, 3), dtype=float))
    image = np.full((3, 5, 3), [16, 128, 240], dtype=np.uint8)
    np.testing.assert_allclose(prepare_input(image).numpy()[0, :, 0, 0], np.array([16, 128, 240])/255)
    result = super_resolve(model, image)
    assert result.rgb.shape == (12, 20, 3) and result.rgb.dtype == np.uint8
    assert result.source_notice == SOURCE_NOTICE
    source = Path('tests/reference_esrgan_numpy.py').read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith(('torch', 'sr_compare')) for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not node.module.startswith(('torch', 'sr_compare'))


@pytest.mark.parametrize('fault', ['missing', 'shape', 'dtype', 'nonfinite', 'wrapper'])
def test_g_t1_rejects_invalid_state(model, tmp_path, monkeypatch, fault):
    import sr_compare.esrgan as module
    path = tmp_path/'synthetic_checkpoint.pth'
    path.write_bytes(b'unit test state validation only')
    monkeypatch.setattr(module, 'CHECKPOINT_SHA256',
                        module.hashlib.sha256(path.read_bytes()).hexdigest())
    state = dict(model.state_dict())
    if fault == 'missing':
        state.pop('conv_first.weight')
    elif fault == 'shape':
        state['conv_first.weight'] = torch.zeros(1)
    elif fault == 'dtype':
        state['conv_first.weight'] = state['conv_first.weight'].double()
    elif fault == 'nonfinite':
        state['conv_first.weight'] = torch.full_like(state['conv_first.weight'], float('nan'))
    monkeypatch.setattr(module.torch, 'load', lambda *a, **k:
                        {'wrong': state} if fault == 'wrapper' else {'params': state})
    with pytest.raises((ValueError, RuntimeError)):
        module.load_esrgan(path)


def test_g_t5_manual_zero_padding_and_upsample(model):
    image = torch.ones(1, 1, 3, 5)
    kernel = torch.ones(1, 1, 3, 3)
    actual = functional.conv2d(image, kernel, padding=1)[0, 0].numpy()
    np.testing.assert_array_equal(actual, [[4, 6, 6, 6, 4], [6, 9, 9, 9, 6], [4, 6, 6, 6, 4]])
    assert all(conv.padding == (1, 1) and conv.padding_mode == 'zeros'
               for conv in model.modules() if isinstance(conv, nn.Conv2d))
    impulse = torch.zeros(1, 1, 3, 5)
    impulse[0, 0, 1, 2] = 1
    enlarged = functional.interpolate(functional.interpolate(
        impulse, scale_factor=2, mode='nearest'), scale_factor=2, mode='nearest')
    expected = torch.zeros(1, 1, 12, 20)
    expected[0, 0, 4:8, 8:12] = 1
    torch.testing.assert_close(enlarged, expected, atol=0, rtol=0)


def test_g_t7_nonfinite_file_handling(tmp_path):
    from sr_compare.experiment import write_json
    from sr_compare.io import save_rgb
    class Nonfinite(nn.Module):
        def forward(self, image):
            return torch.full((1, 3, 12, 20), float('inf'))
    with pytest.raises(ValueError, match='finite'):
        super_resolve(Nonfinite(), np.zeros((3, 5, 3), dtype=np.uint8))
    with pytest.raises(FileNotFoundError):
        load_esrgan(tmp_path/'missing.pth')
    path = tmp_path/'unit_only.png'
    save_rgb(path, np.zeros((3, 5, 3), dtype=np.uint8))
    with pytest.raises(FileExistsError):
        save_rgb(path, np.zeros((3, 5, 3), dtype=np.uint8))
    output = tmp_path/'unit.json'
    write_json(output, {'psnr': float('inf'), 'identical_images': True})
    assert json.loads(output.read_text()) == {'psnr': None, 'identical_images': True}


def test_g_t6_negative_controls(model, comparisons, monkeypatch):
    images, expected = comparisons
    image = images['noise']
    tensor = prepare_input(image)
    reports = {}
    for fault in ('bgr', 'concat', 'rdb_no_scale', 'rdb_scale_01',
                  'rrdb_no_scale', 'rrdb_scale_01', 'subtract_identity', 'bilinear'):
        with monkeypatch.context() as patch:
            if fault == 'concat':
                original = torch.cat
                patch.setattr(torch, 'cat', lambda values, dim=0: original(tuple(reversed(values)), dim))
            if fault.startswith('rdb_'):
                for block in model.modules():
                    if isinstance(block, Rdb):
                        patch.setattr(block, 'res_scale', 1. if fault == 'rdb_no_scale' else .1)
            if fault.startswith('rrdb_'):
                for block in model.body:
                    patch.setattr(block, 'res_scale', 1. if fault == 'rrdb_no_scale' else .1)
            if fault == 'subtract_identity':
                for block in model.body:
                    patch.setattr(block, 'forward', lambda x, block=block:
                                  x+.2*(block.rdb3(block.rdb2(block.rdb1(x)))-x))
            if fault == 'bilinear':
                original = functional.interpolate
                patch.setattr(functional, 'interpolate', lambda x, **kwargs:
                              original(x, scale_factor=2, mode='bilinear', align_corners=False))
            with torch.inference_mode():
                values = model(tensor[:, [2, 1, 0]] if fault == 'bgr' else tensor)
            actual = values[0].permute(1, 2, 0).numpy()*255
            reports[fault] = differences(actual, expected['noise'])
            print('G-T6 NEGATIVE', fault, json.dumps(reports[fault]), flush=True)
            assert not accepted(reports[fault]), f'Negative control escaped: {fault}'
    report_path('m6_negative_report.json').write_text(json.dumps(
        {'source_notice': SOURCE_NOTICE, 'controls': reports}, indent=2), encoding='utf-8')


@pytest.mark.parametrize('shape', [(1, 1), (1, 5), (5, 1), (5, 7)])
def test_g_t4_g_t5_manual_network_signed_routing(shape):
    """Exercise the actual forward graph with signed paths and zero boundaries."""
    network = RrdbNet()

    class Twice(nn.Module):
        def forward(self, image):
            return image*2

    network.body = Twice()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.conv_first.weight[0, 0, 1, 1] = -.5
        network.conv_body.weight[0, 0, 1, 1] = -.25
        network.conv_up1.weight[0, 0, 1, 1] = -2
        network.conv_up2.weight[0, 0, 0, 0] = -2
        network.conv_hr.weight[0, 0, 1, 1] = -2
        network.conv_last.weight[0, 0, 1, 1] = -1
    image = np.zeros(shape, dtype=np.float64)
    image[0, 0] = .25
    image[-1, -1] -= .5
    if min(shape) > 1:
        image[shape[0]//2, 0] = -.25
    tensor = torch.zeros(1, 3, *shape)
    tensor[0, 0] = torch.from_numpy(image.astype(np.float32))
    captured = {}
    names = ('conv_first', 'conv_body', 'conv_up1', 'conv_up2', 'conv_hr', 'conv_last')
    hooks = [getattr(network, name).register_forward_hook(
        lambda module, args, output, name=name: captured.update(
            {name: output.detach().numpy().copy()})) for name in names]
    head = -.5*image
    trunk = -.25*(2*head)
    features = head+trunk
    enlarged1 = np.repeat(np.repeat(features, 2, axis=0), 2, axis=1)
    up1 = -2*enlarged1
    active1 = np.where(up1 >= 0, up1, .2*up1)
    enlarged2 = np.repeat(np.repeat(active1, 2, axis=0), 2, axis=1)
    shifted = np.pad(enlarged2, ((1, 0), (1, 0)))[:-1, :-1]
    up2 = -2*shifted
    active2 = np.where(up2 >= 0, up2, .2*up2)
    hr = -2*active2
    last = -np.where(hr >= 0, hr, .2*hr)
    with torch.inference_mode():
        actual = network(tensor).numpy()
    for hook in hooks:
        hook.remove()
    manual = dict(zip(names, (head, trunk, up1, up2, hr, last)))
    errors = {}
    for name, expected in manual.items():
        errors[name] = float(np.max(np.abs(captured[name][0, 0]-expected)))
        np.testing.assert_allclose(captured[name][0, 0], expected, atol=1e-6, rtol=0)
        assert not np.any(captured[name][0, 1:])
    np.testing.assert_allclose(actual[0, 0], last, atol=1e-6, rtol=0)
    assert not np.any(actual[0, 1:])
    print('G-T4/G-T5 manual signed network', shape, errors)


def test_g_t9_alignment_tool():
    spec = importlib.util.spec_from_file_location('sanity_esrgan', 'scripts/sanity_esrgan_set14.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = np.zeros((64, 80))
    image[16:48, 18:62] = np.random.default_rng(18).random((32, 44))
    moved = np.zeros_like(image)
    moved[3:, :-2] = image[:-3, 2:]
    assert module.displacement(image, moved) == (3, -2)
    assert module.displacement(image, image) == (0, 0)
