"""Run Bicubic x4 on supplied benchmark pairs or one real RGB HR image."""
import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sr_compare.experiment import run_experiment


def main() -> None:
    """Parse experiment settings; output must be a fresh directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--data-dir', type=Path)
    source.add_argument('--hr', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--backend', choices=['numpy', 'pillow'],
                        default='numpy')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(message)s')
    default_data = Path('data/set5/image_SRF_4')
    data_dir = None if args.hr else args.data_dir or default_data
    run_experiment(data_dir, args.output_dir, args.backend,
                   args.warmup, args.repeats, args.hr)


if __name__ == '__main__':
    main()
