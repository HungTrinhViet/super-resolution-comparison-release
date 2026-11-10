"""Inspect data and state dictionaries without constructing SR networks."""
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

import numpy as np
from scipy.io import loadmat

MAX_UNCOMPRESSED_BYTES = 200_000_000
MAX_COMPRESSION_RATIO = 100


def check_zip_entry(entry: zipfile.ZipInfo) -> None:
    """Reject absolute, drive, traversal and symlink archive entries."""
    name = entry.filename.replace('\\', '/')
    path = PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts or ':' in name
            or stat.S_ISLNK(entry.external_attr >> 16)):
        raise ValueError(f'Unsafe archive path: {entry.filename}')


def is_mat73(header: bytes) -> bool:
    """Recognize MATLAB v7.3/HDF5 before calling the classic MAT parser."""
    signature = b'\x89HDF\r\n\x1a\n'
    return (b'MATLAB 7.3 MAT-file' in header or header.startswith(signature)
            or header[512:520] == signature)


def inspect_mat(path: Path) -> dict:
    """List classic MATLAB variables; refuse v7.3 rather than adding h5py."""
    with path.open('rb') as stream:
        if is_mat73(stream.read(520)):
            raise ValueError('MATLAB v7.3/HDF5 requires unapproved h5py; stop.')
    arrays = loadmat(path)
    rows, parameters = [], 0
    for name, value in sorted(arrays.items()):
        if name.startswith('__'):
            continue
        rows.append({'name': name, 'shape': list(value.shape),
                     'dtype': str(value.dtype), 'elements': int(value.size)})
        if 'weight' in name.lower() or 'bias' in name.lower():
            parameters += value.size
    return {'path': path.as_posix(), 'format': 'classic_mat',
            'variables': rows, 'parameter_candidates_total': int(parameters)}


def inspect_torch(path: Path) -> dict:
    """Use restricted CPU tensor loading, reporting all state entries."""
    import torch

    loaded = torch.load(path, weights_only=True, map_location='cpu')
    if not isinstance(loaded, dict):
        raise ValueError('Checkpoint is not a dictionary; stop.')
    schema, state = 'raw', loaded
    for key in ['params', 'params_ema', 'state_dict']:
        if key in loaded:
            schema, state = key, loaded[key]
            break
    if not isinstance(state, dict) or not state:
        raise ValueError('State dictionary is empty or malformed; stop.')
    rows = []
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError('Unexpected non-tensor state entry; stop.')
        rows.append({'name': name, 'shape': list(value.shape),
                     'dtype': str(value.dtype), 'elements': value.numel()})
    return {'path': path.as_posix(), 'schema': schema,
            'top_level_keys': list(loaded), 'state': rows,
            'total_elements': sum(row['elements'] for row in rows)}


def inspect_zip(path: Path, extract_x4: bool) -> None:
    """Validate the bounded archive, print MATLAB text, optionally copy x4 MATs."""
    print(f'ZIP_LIMITS total_bytes={MAX_UNCOMPRESSED_BYTES} '
          f'compression_ratio={MAX_COMPRESSION_RATIO}')
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        total = sum(entry.file_size for entry in entries)
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ValueError('Archive exceeds uncompressed budget; stop.')
        seen = set()
        for entry in entries:
            check_zip_entry(entry)
            if not entry.is_dir():
                normalized = entry.filename.replace('\\', '/').casefold()
                if normalized in seen:
                    raise ValueError('Duplicate case-insensitive zip file; stop.')
                seen.add(normalized)
            ratio = entry.file_size / max(entry.compress_size, 1)
            if ratio > MAX_COMPRESSION_RATIO:
                raise ValueError('Archive exceeds compression ratio; stop.')
            print(f'ZIP_ENTRY {entry.filename!r} compressed={entry.compress_size} '
                  f'uncompressed={entry.file_size} CRC={entry.CRC:08x}')
        corrupted = archive.testzip()
        if corrupted is not None:
            raise ValueError(f'Archive CRC mismatch: {corrupted}')
        print(f'ZIP_CRC PASS total_uncompressed={total}')
        selected = []
        for entry in entries:
            name = entry.filename
            if entry.is_dir():
                continue
            if name.lower().endswith('.m'):
                text = archive.read(entry).decode('utf-8-sig')
                print(f'MATLAB_TEXT_ONLY {name}')
                for line, content in enumerate(text.splitlines(), 1):
                    print(f'{name}:{line}: {content}')
            x4 = name.lower().endswith('.mat') and re.search(
                r'(?:^|[^0-9])(?:x|scale[_ -]?)?4(?:[^0-9]|$)', name.lower())
            text_document = PurePosixPath(name).name.lower().startswith(
                ('readme', 'license', 'copying'))
            if x4 or text_document:
                selected.append(entry)
        if not any(entry.filename.lower().endswith('.mat') for entry in selected):
            raise ValueError('Archive has no identifiable x4 MAT checkpoint; stop.')
        if extract_x4:
            root = Path('data/checkpoints/srcnn')
            for entry in selected:
                relative = PurePosixPath(entry.filename.replace('\\', '/'))
                target = root.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(archive.read(entry))
                print(f'EXTRACTED {target.as_posix()}')
                if target.suffix.lower() == '.mat':
                    print(json.dumps(inspect_mat(target), indent=2))


def main() -> None:
    """Inspect one approved artifact; never execute its source code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--extract-x4', action='store_true')
    args = parser.parse_args()
    if args.path.suffix.lower() == '.zip':
        inspect_zip(args.path, args.extract_x4)
    elif args.path.suffix.lower() == '.mat':
        print(json.dumps(inspect_mat(args.path), indent=2))
    else:
        print(json.dumps(inspect_torch(args.path), indent=2))


if __name__ == '__main__':
    main()
