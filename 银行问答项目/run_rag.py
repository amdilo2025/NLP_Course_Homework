"""Entry point. Local dependencies are isolated from the baseline and system Python."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
if (ROOT/'.rag_packages').is_dir():sys.path.insert(0,str(ROOT/'.rag_packages'))
sys.path.insert(0,str(ROOT/'src'))
from icbc_rag.cli import main

if __name__=='__main__':raise SystemExit(main())
