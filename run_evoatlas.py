#!/usr/bin/env python3
"""EvoAtlas CLI entry point."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src import run_evoatlas

if __name__ == "__main__":
    results = run_evoatlas(
        input_source=None,
        n_seqs=80,
        out_dir="evoatlas_output",
        window_size=30,
        mi_max_sites=120,
        fast_mode=True,
    )
