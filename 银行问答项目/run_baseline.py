"""Run the local baseline with Python 3.10+; no installation is required."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from icbc_baseline.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
