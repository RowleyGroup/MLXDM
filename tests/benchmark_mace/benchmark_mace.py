"""Benchmark MACEPBE0 or MACEXDM with a growing chain of pentane molecules.

Self-contained single-file version (only depends on ase, torch, and mace)
so it can be dropped onto a different cluster on its own.

Every 10 molecules, times a 10000-step Langevin MD run at 298.15 K and
writes [n_molecules, time_seconds] to a CSV file, flushing after each row.

Usage:
    python benchmark_mace.py macepbe0 100
    python benchmark_mace.py macexdm 100
    python benchmark_mace.py macexdm 100 --pbe0-checkpoint /path/to/pbe0.model \
        --xdm-checkpoint /path/to/xdm.model --xyz xyz/pentane.xyz --device cuda
"""
import argparse
import csv
import sys
import time

import ase.io
import torch
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# Defaults from the original cluster; override with --pbe0-checkpoint /
# --xdm-checkpoint when deploying elsewhere.
DEFAULT_PBE0_CKPT = "/project/6007501/crowley/mace/h5-nibi/checkpoints/mace-pbe0_0_s2.model"
DEFAULT_XDM_CKPT = "/project/6007501/crowley/mace-xdm/mace-xdm-rocm/train-element/models_24/xdm_element.model"

T = 298.15
MD_STEPS = 10000


def build_calculator(model, pbe0_checkpoint, xdm_checkpoint, device):
    if model == "macepbe0":
        from mace.calculators import MACECalculator

        return MACECalculator(model_paths=pbe0_checkpoint, device=device)

    from mace.calculators import MACEXDMDispersionCalculator

    return MACEXDMDispersionCalculator(
        short_range_model_path=pbe0_checkpoint,
        xdm_model_path=xdm_checkpoint,
        device=device,
    )


def run_benchmark(calc, max_n_pentane, csv_filename, xyz_path):
    try:
        base_mol = ase.io.read(xyz_path)
    except FileNotFoundError:
        print(f"Error: '{xyz_path}' not found.")
        sys.exit(1)

    coords_initial = base_mol.get_positions()
    atoms = base_mol.copy()
    atoms.calc = calc

    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_molecules", "time_seconds"])

        print(f"Starting simulation runs up to {max_n_pentane} molecules...")

        for i in range(1, max_n_pentane + 1):
            # Progressively translate new molecules so they don't overlap in space
            if i > 1:
                new_mol = base_mol.copy()
                new_mol.set_positions(coords_initial + [0, 0, 3 * i])
                atoms = atoms + new_mol
                atoms.calc = calc

            if i % 10 == 0:
                print(f"Running MD for {i} molecules...")
                time_initial = time.perf_counter()

                _ = atoms.get_potential_energy()

                MaxwellBoltzmannDistribution(atoms, temperature_K=T)
                dyn = Langevin(atoms, 1 * units.fs, T * units.kB, 0.002)
                dyn.run(MD_STEPS)

                time_final = time.perf_counter()
                elapsed_time = time_final - time_initial

                print(f"Completed {i} molecules in {elapsed_time:.2f} seconds.")

                writer.writerow([i, elapsed_time])
                f.flush()

    print(f"All timings successfully saved to {csv_filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["macepbe0", "macexdm"],
                         help="Which model to benchmark")
    parser.add_argument("max_n_pentane", type=int,
                         help="Grow the chain up to this many pentane molecules")
    parser.add_argument("--pbe0-checkpoint", default=DEFAULT_PBE0_CKPT,
                         help="Path to the MACEPBE0 checkpoint (used by both models)")
    parser.add_argument("--xdm-checkpoint", default=DEFAULT_XDM_CKPT,
                         help="Path to the XDM checkpoint (macexdm only)")
    parser.add_argument("--xyz", default="xyz/pentane.xyz",
                         help="Base pentane geometry (default: xyz/pentane.xyz)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                         help="torch device (default: cuda if available, else cpu)")
    parser.add_argument("--csv", default=None,
                         help="Output CSV path (default: simulation_timings_<model>.csv)")
    args = parser.parse_args()

    calc = build_calculator(args.model, args.pbe0_checkpoint, args.xdm_checkpoint, args.device)
    csv_filename = args.csv or f"simulation_timings_{args.model}.csv"

    run_benchmark(calc, args.max_n_pentane, csv_filename, args.xyz)


if __name__ == "__main__":
    main()
