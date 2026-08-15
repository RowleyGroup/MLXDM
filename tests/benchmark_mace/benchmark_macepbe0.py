"""Benchmark the MACEPBE0 (short-range only, no XDM dispersion) model.

Usage:
    python benchmark_macepbe0.py <max_n_pentane>
"""
import sys
import torch
from mace_benchmark_common import run_benchmark

MACEPBE0_CKPT = "/project/6007501/crowley/mace/h5-nibi/checkpoints/mace-pbe0_0_s2.model"


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_macepbe0.py <max_n_pentane>")
        sys.exit(1)

    max_n_pentane = int(sys.argv[1])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from mace.calculators import MACECalculator

    calc = MACECalculator(model_paths=MACEPBE0_CKPT, device=device)

    run_benchmark(calc, max_n_pentane, csv_filename="simulation_timings_macepbe0.csv")


if __name__ == "__main__":
    main()
