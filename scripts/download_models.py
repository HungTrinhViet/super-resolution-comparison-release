"""Download one fixed, approved artifact with fail-closed byte validation."""
import argparse
from datetime import datetime, timezone
from email.message import Message
import hashlib
from pathlib import Path
import urllib.request

ZIP_SIGNATURE = b'PK\x03\x04'
LEGACY_SIGNATURE = bytes.fromhex('80 02 8A 0A 6C FC 9C 46 F9 20 6A A8 50 19')

ITEMS = {
    'srcnn': {
        'url': 'https://mmlab.ie.cuhk.edu.hk/projects/SRCNN/SRCNN_v1.zip',
        'size': 7558018, 'path': 'data/quarantine/srcnn/SRCNN_v1.zip',
        'prefix': None,
    },
    'edsr': {
        'url': ('https://drive.google.com/uc?export=download&'
                'id=1bCK6cFYU01uJudLgUUe-jgx-tZ3ikOWn'),
        'size': 6084475,
        'path': ('data/checkpoints/edsr/'
                 'EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth'),
        'prefix': '0c287733',
    },
    'esrgan': {
        'url': ('https://drive.google.com/uc?export=download&'
                'id=1b3_bWZTjNO3iL2js1yWkJfjZykcQgvzT'),
        'size': 66922040,
        'path': ('data/checkpoints/esrgan/'
                 'ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'),
        'prefix': 'ff704c30',
    },
}


def check_prefix(content: bytes, content_type: str) -> None:
    """Accept only full ZIP or legacy torch signatures, never HTML."""
    start = content.lstrip().lower()
    if ('text/html' in content_type.lower()
            or start.startswith((b'<html', b'<!doctype html'))):
        raise ValueError('HTML received instead of the approved artifact; stop.')
    if not content.startswith((ZIP_SIGNATURE, LEGACY_SIGNATURE)):
        raise ValueError('Artifact magic mismatch: expected ZIP or full '
                         '14-byte legacy torch signature; stop.')


def check_filename(disposition: str | None, expected: str) -> None:
    """Validate the publisher filename when Content-Disposition is present."""
    if disposition is None:
        return
    message = Message()
    message['Content-Disposition'] = disposition
    if message.get_filename() != expected:
        raise ValueError('Publisher filename differs from approved name; stop.')


def download(item: str, destination: Path) -> dict:
    """Stream into an exclusive .part file and promote only validated bytes."""
    specification = ITEMS[item]
    part = destination.with_name(destination.name + '.part')
    if destination.exists() or part.exists():
        raise FileExistsError(f'Refusing overwrite: {destination} or {part}')
    request = urllib.request.Request(
        specification['url'], headers={'User-Agent': 'SISR-artifact-audit'})
    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(request, timeout=60) as response:
        first = response.read(65536)
        check_prefix(first, response.headers.get('Content-Type', ''))
        if item == 'srcnn' and not first.startswith(ZIP_SIGNATURE):
            raise ValueError('SRCNN archive requires ZIP format; stop.')
        check_filename(response.headers.get('Content-Disposition'),
                       Path(specification['path']).name)
        expected_size = specification['size']
        content_length = response.headers.get('Content-Length')
        if content_length is not None and int(content_length) != expected_size:
            raise ValueError('Publisher response size differs from approved size.')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with part.open('xb') as stream:
            chunk = first
            while chunk:
                total += len(chunk)
                if total > expected_size:
                    raise ValueError('Artifact exceeds approved size; stop.')
                digest.update(chunk)
                stream.write(chunk)
                chunk = response.read(65536)
    if total != expected_size:
        raise ValueError(f'Artifact size mismatch: {total} != {expected_size}')
    sha256 = digest.hexdigest()
    prefix = specification['prefix']
    if prefix is not None and not sha256.startswith(prefix):
        raise ValueError(f'Partial filename hash mismatch: {sha256} vs {prefix}')
    # On Windows Path.rename refuses an already existing destination.
    if destination.exists():
        raise FileExistsError(f'Refusing overwrite: {destination}')
    part.rename(destination)
    return {'item': item, 'url': specification['url'],
            'path': destination.as_posix(), 'bytes': total,
            'sha256_local': sha256, 'publisher_full_sha256': None,
            'filename_prefix': prefix,
            'format': ('zip' if first.startswith(ZIP_SIGNATURE)
                       else 'torch_legacy_protocol_2'),
            'utc': datetime.now(timezone.utc).isoformat()}


def main() -> None:
    """Download exactly one requested gate item, never all artifacts at once."""
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--item', choices=ITEMS, required=True)
    args = parser.parse_args()
    record = download(args.item, Path(ITEMS[args.item]['path']))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
