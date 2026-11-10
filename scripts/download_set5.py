"""Download only the ten approved Set5 RGB PNGs and verify source hashes."""
import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sr_compare.dataset import download_set5


def main() -> None:
    """Run the fixed-source downloader without overwriting data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=Path('data/set5/image_SRF_4'))
    parser.add_argument('--docs-dir', type=Path, default=Path('docs'))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(message)s')
    download_set5(args.data_dir, args.docs_dir)


if __name__ == '__main__':
    main()
