"""Synthetic-only checkpoint-tool validation without network downloads."""
import importlib.util
from pathlib import Path
import zipfile

import numpy as np
import pytest
from scipy.io import savemat


def tool(name: str):
    path = Path(__file__).parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_payload_security():
    downloader = tool('download_models')
    downloader.check_prefix(b'PK\x03\x04payload', 'application/zip')
    for value, content_type in [(b'<html>confirmation</html>', 'text/html'),
                                 (b'\x80\x02pickle', 'application/octet-stream')]:
        with pytest.raises(ValueError):
            downloader.check_prefix(value, content_type)


def test_existing_download_is_not_overwritten(tmp_path):
    downloader = tool('download_models')
    destination = tmp_path / 'existing.zip'
    destination.write_bytes(b'not changed')
    with pytest.raises(FileExistsError):
        downloader.download('srcnn', destination)
    assert destination.read_bytes() == b'not changed'


def test_zip_paths_and_mat73():
    inspector = tool('inspect_checkpoint')
    inspector.check_zip_entry(zipfile.ZipInfo('models/x4.mat'))
    for name in ['../bad.mat', '/absolute.mat', 'C:/bad.mat',
                 'model/../../bad.mat', 'model\\..\\bad.mat']:
        with pytest.raises(ValueError):
            inspector.check_zip_entry(zipfile.ZipInfo(name))
    assert inspector.is_mat73(b'MATLAB 7.3 MAT-file')
    assert not inspector.is_mat73(b'MATLAB 5.0 MAT-file')


def test_small_mat_inspection(tmp_path):
    inspector = tool('inspect_checkpoint')
    path = tmp_path / 'synthetic_unit_only.mat'
    savemat(path, {'weights_conv1': np.zeros((2, 3), dtype=np.float32),
                   'biases_conv1': np.zeros((2, 1), dtype=np.float32)})
    result = inspector.inspect_mat(path)
    assert result['parameter_candidates_total'] == 8
    assert result['variables'][0]['dtype'] == 'float32'


def test_tiny_safe_tensor_checkpoint(tmp_path):
    import torch

    inspector = tool('inspect_checkpoint')
    path = tmp_path / 'synthetic_unit_only.pth'
    torch.save({'params': {'weight': torch.zeros(2, 3)}}, path)
    result = inspector.inspect_torch(path)
    assert result['schema'] == 'params'
    assert result['total_elements'] == 6


@pytest.mark.parametrize('signature', [
    b'PK\x03\x04',
    bytes.fromhex('80 02 8A 0A 6C FC 9C 46 F9 20 6A A8 50 19'),
])
def test_approved_signatures(signature):
    tool('download_models').check_prefix(signature, 'application/octet-stream')


@pytest.mark.parametrize('payload', [
    b'', b'\x80\x02', b'<html>download confirmation</html>',
    bytes.fromhex('80 02 8A 0A 6C FC 9C 46 F9 20 6A A8 50 18'),
])
def test_incomplete_or_wrong_signatures(payload):
    with pytest.raises(ValueError):
        tool('download_models').check_prefix(payload, 'application/octet-stream')


def test_legacy_tensor_checkpoint_weights_only(tmp_path):
    import torch

    path = tmp_path / 'synthetic_legacy.pth'
    torch.save({'weight': torch.arange(6).reshape(2, 3)}, path,
               _use_new_zipfile_serialization=False)
    header = path.read_bytes()[:14]
    assert header == bytes.fromhex('80 02 8A 0A 6C FC 9C 46 F9 20 6A A8 50 19')
    tool('download_models').check_prefix(header, 'application/octet-stream')
    result = tool('inspect_checkpoint').inspect_torch(path)
    assert result['schema'] == 'raw'
    assert result['total_elements'] == 6
    loaded = torch.load(path, weights_only=True, map_location='cpu')
    assert torch.equal(loaded['weight'], torch.arange(6).reshape(2, 3))


def test_publisher_filename():
    downloader = tool('download_models')
    downloader.check_filename(None, 'approved.pth')
    downloader.check_filename('attachment; filename="approved.pth"', 'approved.pth')
    with pytest.raises(ValueError):
        downloader.check_filename('attachment; filename="other.pth"', 'approved.pth')
