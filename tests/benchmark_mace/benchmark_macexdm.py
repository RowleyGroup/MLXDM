"""Benchmark the MACEXDM model (MACEPBE0 short-range + XDM dispersion).

Usage:
    python benchmark_macexdm.py <max_n_pentane>
"""
import sys
import torch
from mace_benchmark_common import run_benchmark

MACEPBE0_CKPT = "/project/6007501/crowley/mace/h5-nibi/checkpoints/mace-pbe0_0_s2.model"
ATOMICXDMMACE_CKPT = "/project/6007501/crowley/mace-xdm/mace-xdm-rocm/train-element/models_24/xdm_element.model"


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_macexdm.py <max_n_pentane>")
        sys.exit(1)

    max_n_pentane = int(sys.argv[1])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from mace.calculators import MACEXDMDispersionCalculator

    calc = MACEXDMDispersionCalculator(
        short_range_model_path=MACEPBE0_CKPT,
        xdm_model_path=ATOMICXDMMACE_CKPT,
        device=device,
    )

    run_benchmark(calc, max_n_pentane, csv_filename="simulation_timings_macexdm.csv")


if __name__ == "__main__":
    main()
