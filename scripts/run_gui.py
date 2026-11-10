"""Launch the approved Super Resolution x4 desktop demo."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sr_compare.inference import configure_threads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-startup', action='store_true',
                        help='Report CPU thread settings without creating a window.')
    args = parser.parse_args()
    configure_threads()
    if args.check_startup:
        import torch
        print(f'CPU threads: {torch.get_num_threads()}/{torch.get_num_interop_threads()}')
        return
    import tkinter as tk
    from sr_compare.gui.app import DemoApp
    from sr_compare.gui.styles import enable_dpi_awareness
    enable_dpi_awareness()
    root = tk.Tk()
    DemoApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
