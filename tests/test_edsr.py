"""Locked E-T1–E-T6 and E-T9 alignment tests, without Set5 inference."""
from io import BytesIO
import importlib.util
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as functional

from reference_edsr_numpy import pixel_shuffle, reference_forward
from sr_compare.edsr import (
    CHECKPOINT, Edsr, add_mean, load_edsr, prepare_input, quantize_values,
    subtract_mean, super_resolve, tensor_to_rgb,
)


@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(4)
    return load_edsr(CHECKPOINT)


def samples():
    rng = np.random.default_rng(175)
    yy, xx = np.indices((9, 13))
    values = {
        'noise': rng.integers(0, 256, (9, 13, 3), dtype=np.uint8),
        'gradient': np.stack([xx*19, yy*27, (xx+yy)*11], axis=-1).astype(np.uint8),
        'checker_color': np.stack([(xx+yy)%2*255, xx%2*190, yy%2*230], axis=-1).astype(np.uint8),
        'distinct_rgb': np.stack([(xx*17+yy*3)%256, (yy*23+40)%256,
                                   (200-xx*9+yy*2)%256], axis=-1).astype(np.uint8),
        'constant_color': np.full((9, 13, 3), [10, 120, 230], dtype=np.uint8),
    }
    with zipfile.ZipFile('data/quarantine/srcnn/SRCNN_v1.zip') as archive:
        for name in ['zebra', 'monarch']:
            with Image.open(BytesIO(archive.read(f'SRCNN/Set14/{name}.bmp'))) as image:
                values[name] = np.array(image)[31:40, 43:56].copy()
    return values


def test_e_t1_state_and_count(model):
    assert len(model.state_dict()) == 74
    assert sum(p.numel() for p in model.parameters()) == 1517571
    assert len(model.body) == 16
    assert all(block.res_scale == 1 for block in model.body)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert 'mean' not in model.state_dict()


def test_e_t2_and_e_t6c_d_independent_reference(model, tmp_path):
    wrapper = torch.load(CHECKPOINT, weights_only=True, map_location='cpu')
    path = tmp_path / 'author_converted_weights_unit_test.npz'
    np.savez(path, **{name: value.numpy() for name, value in wrapper['params'].items()})
    for name, sample in samples().items():
        expected_float, expected_rgb = reference_forward(path, sample)
        actual = super_resolve(model, sample)
        error = np.abs(expected_float - actual.prequant_float)
        differences = np.abs(expected_rgb.astype(int) - actual.rgb.astype(int))
        pixel_count = int(np.any(differences != 0, axis=-1).sum())
        fraction = pixel_count / differences.shape[0] / differences.shape[1]
        print('E-T2', name, 'float max/mean', error.max(), error.mean(),
              'channel max/mean', error.max(axis=(0, 1)).tolist(),
              error.mean(axis=(0, 1)).tolist(), 'uint8 pixels different', pixel_count,
              '/', differences.shape[0]*differences.shape[1], 'max', differences.max(),
              'fraction', fraction, 'half ties', actual.half_tie_count)
        assert error.max() <= 1e-2
        assert differences.max() <= 1 and fraction <= .001
        for channel in range(3):
            assert error[..., channel].max() <= 1e-2
            assert np.mean(differences[..., channel] != 0) <= .001
        if name == 'distinct_rgb':
            wrong = super_resolve(model, sample[..., ::-1].copy())
            negative_error = np.max(np.abs(wrong.prequant_float - expected_float))
            print('E-T6d RGB/BGR negative control max error', negative_error,
                  'must EXCEED 1e-2')
            assert negative_error > 1e-2


def test_e_t3_shuffle_definition_and_negative_order():
    source = np.broadcast_to(np.arange(12)[:, None, None], (12, 3, 5)).copy()
    expected = pixel_shuffle(source, 2)
    actual = functional.pixel_shuffle(torch.tensor(source)[None], 2)[0].numpy()
    np.testing.assert_array_equal(actual, expected)
    wrong = expected.reshape(3, 3, 2, 5, 2).swapaxes(2, 4).reshape(3, 6, 10)
    assert not np.array_equal(actual, wrong)


def test_e_t4_and_e_t6b_mean_shift():
    x = torch.rand((1, 3, 7, 11), generator=torch.Generator().manual_seed(19))
    mean = torch.tensor([.4488, .4371, .4040]).reshape(1, 3, 1, 1)
    weight = torch.eye(3).reshape(3, 3, 1, 1)
    original = functional.conv2d(x*255, weight, bias=-255*mean.flatten())
    error = torch.max(torch.abs(subtract_mean(x)-original)).item()
    inverse = torch.max(torch.abs(add_mean(original)*255
                                 - functional.conv2d(original, weight,
                                                     bias=255*mean.flatten()))).item()
    rgb = torch.tensor([10., 120., 230.]).reshape(1, 3, 1, 1)/255
    expected = torch.tensor([10-255*.4488, 120-255*.4371, 230-255*.4040])
    channel_error = torch.max(torch.abs(subtract_mean(rgb).flatten()-expected)).item()
    roundtrip = torch.max(torch.abs(add_mean(subtract_mean(rgb))*255
                                   - torch.tensor([10.,120.,230.]).reshape(1,3,1,1))).item()
    print('E-T4/E-T6b errors', error, inverse, channel_error, roundtrip)
    assert max(error, inverse, channel_error, roundtrip) <= 1e-4


def test_e_t5_and_e_t6a_rgb_adapter(model):
    image = np.full((7, 11, 3), [10, 120, 230], dtype=np.uint8)
    tensor = prepare_input(image)
    np.testing.assert_array_equal(tensor[0, :, 2, 3].numpy(),
                                  np.array([10,120,230], dtype=np.float32)/255)
    recovered = tensor_to_rgb(tensor[0]*255)
    np.testing.assert_array_equal(recovered, image)
    result = super_resolve(model, image)
    assert result.rgb.shape == result.prequant_float.shape == (28,44,3)
    assert result.rgb.dtype == np.uint8
    assert np.isfinite(result.prequant_float).all()
    for value in [float('nan'),float('inf'),-float('inf')]:
        with pytest.raises(ValueError,match='finite'):
            quantize_values(torch.tensor([value]))


def test_e_t5b_half_even():
    values=torch.tensor([.5,1.5,2.5,3.5,254.5,-.5,255.5,300,.49])
    np.testing.assert_array_equal(quantize_values(values).numpy(),
                                  [0,2,2,4,254,0,255,255,0])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_e_t5_adapter_rejects_nonfinite_before_quantization(value):
    class NonfiniteModel(torch.nn.Module):
        def forward(self, image):
            return torch.full((1, 3, image.shape[2]*4, image.shape[3]*4), value)

    with pytest.raises(ValueError, match='finite'):
        super_resolve(NonfiniteModel(), np.zeros((5, 7, 3), dtype=np.uint8))


def test_e_t9_alignment_tool():
    spec=importlib.util.spec_from_file_location('sanity_edsr','scripts/sanity_edsr_set14.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image=np.zeros((64,80))
    image[16:48,18:62]=np.random.default_rng(18).random((32,44))
    moved=np.zeros_like(image)
    moved[3:,:-2]=image[:-3,2:]
    assert module.displacement(image,moved)==(3,-2)
    assert module.displacement(image,image)==(0,0)
