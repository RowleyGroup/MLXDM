"""Benchmark MACEPBE0 or MACEXDM by running MD on a series of water balls.

Self-contained single-file version (only depends on ase, torch, and mace)
so it can be dropped onto a different cluster on its own.

Reads water-ball .xyz files from a directory (default:
/lustre06/project/6060902/crowley/timing/xyz), smallest first, and for each
one times a 10000-step Langevin MD run at 298.15 K. Results are written to a
CSV file, flushing after each row so partial results survive a job that
gets killed or times out partway through.

Usage:
    python benchmark_mace.py macepbe0
    python benchmark_mace.py macexdm --xyz-dir /path/to/xyz --limit 20
    python benchmark_mace.py macexdm --xyz-file /path/to/xyz/water_ball_0100.xyz
    python benchmark_mace.py macexdm --pbe0-checkpoint /path/to/pbe0.model \
        --xdm-checkpoint /path/to/xdm.model --device cuda
"""
import argparse
import csv
import glob
import os
import re
import sys
import time

import ase.io
import torch
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# Defaults from the original cluster; override with the matching flags when
# deploying elsewhere.
DEFAULT_PBE0_CKPT = "/project/6007501/crowley/mace/h5-nibi/checkpoints/mace-pbe0_0_s2.model"
DEFAULT_XDM_CKPT = "/project/6007501/crowley/mace-xdm/mace-xdm-rocm/train-element/models_24/xdm_element.model"
DEFAULT_XYZ_DIR = "/lustre06/project/6060902/crowley/timing/xyz"

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


def xyz_atom_count(path):
    with open(path) as fh:
        return int(fh.readline())


def find_xyz_files(xyz_dir, pattern):
    paths = glob.glob(os.path.join(xyz_dir, pattern))
    if not paths:
        print(f"Error: no files matching '{pattern}' found in '{xyz_dir}'.")
        sys.exit(1)
    return sorted(paths, key=xyz_atom_count)


def n_waters_from_filename(path):
    match = re.search(r"(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else None


def run_benchmark(calc, xyz_files, csv_filename):
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["xyz_file", "n_atoms", "n_waters", "time_seconds"])

        print(f"Starting benchmark over {len(xyz_files)} water-ball structures...")

        for path in xyz_files:
            atoms = ase.io.read(path)
            atoms.calc = calc
            n_atoms = len(atoms)
            n_waters = n_waters_from_filename(path)
            if n_waters is None:
                n_waters = n_atoms // 3

            name = os.path.basename(path)
            print(f"Running MD for {name} ({n_atoms} atoms)...")
            time_initial = time.perf_counter()

            # Get initial potential energy
            _ = atoms.get_potential_energy()

            # Setup and run Langevin dynamics
            MaxwellBoltzmannDistribution(atoms, temperature_K=T)
            dyn = Langevin(atoms, 1 * units.fs, T * units.kB, 0.002)
            dyn.run(MD_STEPS)

            time_final = time.perf_counter()
            elapsed_time = time_final - time_initial

            print(f"Completed {name} in {elapsed_time:.2f} seconds.")

            writer.writerow([name, n_atoms, n_waters, elapsed_time])
            f.flush()

    print(f"All timings successfully saved to {csv_filename}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", choices=["macepbe0", "macexdm"],
                         help="Which model to benchmark")
    parser.add_argument("--xyz-dir", default=DEFAULT_XYZ_DIR,
                         help="Directory of water-ball xyz files (default: %(default)s)")
    parser.add_argument("--pattern", default="water_ball_*.xyz",
                         help="Glob pattern for xyz files within --xyz-dir (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only benchmark the N smallest structures (default: all)")
    parser.add_argument("--xyz-file", default=None,
                         help="Benchmark only this single xyz file, instead of scanning --xyz-dir "
                              "(used to run one water-ball size per Slurm job)")
    parser.add_argument("--pbe0-checkpoint", default=DEFAULT_PBE0_CKPT,
                         help="Path to the MACEPBE0 checkpoint (used by both models)")
    parser.add_argument("--xdm-checkpoint", default=DEFAULT_XDM_CKPT,
                         help="Path to the XDM checkpoint (macexdm only)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                         help="torch device (default: cuda if available, else cpu)")
    parser.add_argument("--csv", default=None,
                         help="Output CSV path (default: simulation_timings_<model>.csv)")
    args = parser.parse_args()

    calc = build_calculator(args.model, args.pbe0_checkpoint, args.xdm_checkpoint, args.device)

    if args.xyz_file:
        if not os.path.isfile(args.xyz_file):
            print(f"Error: '{args.xyz_file}' not found.")
            sys.exit(1)
        xyz_files = [args.xyz_file]
        stem = os.path.splitext(os.path.basename(args.xyz_file))[0]
        default_csv = f"simulation_timings_{args.model}_{stem}.csv"
    else:
        xyz_files = find_xyz_files(args.xyz_dir, args.pattern)
        if args.limit is not None:
            xyz_files = xyz_files[:args.limit]
        default_csv = f"simulation_timings_{args.model}.csv"

    run_benchmark(calc, xyz_files, args.csv or default_csv)


if __name__ == "__main__":
    main()
