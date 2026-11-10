"""Control whether missing real Set5 data skips or blocks acceptance."""
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the explicit real-data acceptance requirement."""
    parser.addoption('--require-set5', action='store_true', default=False,
                     help='Fail instead of skip if the Set5 data is missing.')


def check_set5(directory: Path, required: bool) -> None:
    """Require all ten paired images without downloading or changing files."""
    expected = [directory / f'img_{i:03d}_SRF_4_{kind}.png'
                for i in range(1, 6) for kind in ['HR', 'LR']]
    missing = [path.name for path in expected if not path.is_file()]
    if missing:
        reason = 'NOT VERIFIED: Set5 missing: ' + ', '.join(missing)
        if required:
            pytest.fail(reason)
        pytest.skip(reason)


@pytest.fixture
def set5_dir(request: pytest.FixtureRequest) -> Path:
    """Locate verified Set5 pairs in the project data directory."""
    directory = Path(__file__).resolve().parents[1] / 'data/set5/image_SRF_4'
    check_set5(directory, request.config.getoption('--require-set5'))
    return directory
