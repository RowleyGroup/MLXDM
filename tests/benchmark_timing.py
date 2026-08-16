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
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

DEFAULT_XYZ_DIR = "/lustre06/project/6060902/crowley/timing/xyz"

T = 298.15
MD_STEPS = 100000


def build_calculator(model, device):
    import torchanipbe0

    if model == "anipbe0":
        wrapped = torchanipbe0.models.ANIPBE0_2x()
    else:
        wrapped = torchanipbe0.models.ANIPBE0_2x_MLXDM_2x()

    calc = wrapped.ase()
    move_calculator_to_device(calc, device)
    return calc


def move_calculator_to_device(calc, device):
    """Move a torchanipbe0 ASE calculator's underlying model to `device`.

    torchanipbe0.ase.Calculator caches the device it was constructed on
    (read once from the first model parameter, at CPU by default) in
    `calc.device`, and calculate() uses that cached attribute -- not the
    model's live parameter device -- to build every input tensor. Calling
    calc.model.to(device) alone is not enough; `calc.device` has to be
    updated too, or every calculate() call silently keeps running on the
    original (CPU) device.
    """
    device = torch.device(device)
    calc.model.to(device)
    calc.device = device

    # torch.device("cuda") and torch.device("cuda:0") refer to the same
    # physical device but don't compare equal, so compare .type (and
    # .index only when the caller pinned a specific GPU).
    param_types = {p.device.type for p in calc.model.parameters()}
    if param_types != {device.type}:
        raise RuntimeError(
            f"Expected all model parameters on device type '{device.type}', "
            f"found {param_types}"
        )
    if device.index is not None:
        param_indices = {p.device.index for p in calc.model.parameters()}
        if param_indices != {device.index}:
            raise RuntimeError(
                f"Expected all model parameters on cuda:{device.index}, "
                f"found indices {param_indices}"
            )
    print(f"Confirmed model parameters and calculator on device: {device}")


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


def attach_timeseries_logger(dyn, atoms, csv_path, start_time, interval):
    """Log step, wall-clock time, energy and temperature via dyn.attach().

    Per-step (or per-interval) wall-clock timestamps let you see how the
    time per step evolves over the course of a run (e.g. warmup/JIT cost
    up front, throttling, memory growth) instead of only the total time,
    which hides all of that.
    """
    f = open(csv_path, mode="w", newline="")
    writer = csv.writer(f)
    writer.writerow(["step", "elapsed_seconds", "epot_eV", "temperature_K"])

    def log():
        step = dyn.get_number_of_steps()
        elapsed = time.perf_counter() - start_time
        epot = atoms.get_potential_energy()
        temp = atoms.get_temperature()
        writer.writerow([step, f"{elapsed:.4f}", f"{epot:.6f}", f"{temp:.2f}"])
        f.flush()

    dyn.attach(log, interval=interval)
    return f


def run_benchmark(calc, xyz_files, csv_filename, outdir, traj_interval, ts_interval):
    os.makedirs(outdir, exist_ok=True)

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
            stem = os.path.splitext(name)[0]
            print(f"Running MD for {name} ({n_atoms} atoms)...")
            time_initial = time.perf_counter()

            # Get initial potential energy
            _ = atoms.get_potential_energy()

            # Setup and run Langevin dynamics
            MaxwellBoltzmannDistribution(atoms, temperature_K=T)
            dyn = Langevin(atoms, 1 * units.fs, T * units.kB, 0.002)

            traj = Trajectory(os.path.join(outdir, f"{stem}.traj"), "w", atoms)
            dyn.attach(traj.write, interval=traj_interval)

            ts_file = attach_timeseries_logger(
                dyn,
                atoms,
                os.path.join(outdir, f"{stem}.timeseries.csv"),
                time_initial,
                ts_interval,
            )

            dyn.run(MD_STEPS)

            traj.close()
            ts_file.close()

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
    parser.add_argument("model", choices=["anipbe0", "mlxdm2"],
                         help="anipbe0: electronic-only ANIPBE0-2x. "
                              "mlxdm2: composite ANIPBE0-2x+MLXDM2 (electronic+dispersion)")
    parser.add_argument("--xyz-dir", default=DEFAULT_XYZ_DIR,
                         help="Directory of water-ball xyz files (default: %(default)s)")
    parser.add_argument("--pattern", default="water_ball_*.xyz",
                         help="Glob pattern for xyz files within --xyz-dir (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only benchmark the N smallest structures (default: all)")
    parser.add_argument("--xyz-file", default=None,
                         help="Benchmark only this single xyz file, instead of scanning --xyz-dir "
                              "(used to run one water-ball size per Slurm job)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                         help="torch device (default: cuda if available, else cpu)")
    parser.add_argument("--csv", default=None,
                         help="Output CSV path (default: simulation_timings_<model>.csv)")
    parser.add_argument("--outdir", default=".",
                         help="Directory to write per-structure trajectory/time-series "
                              "files to (default: %(default)s)")
    parser.add_argument("--traj-interval", type=int, default=1000,
                         help="Steps between trajectory frames (default: %(default)s)")
    parser.add_argument("--ts-interval", type=int, default=100,
                         help="Steps between time-series log rows (default: %(default)s)")
    args = parser.parse_args()

    calc = build_calculator(args.model, args.device)

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

    run_benchmark(
        calc,
        xyz_files,
        args.csv or default_csv,
        args.outdir,
        args.traj_interval,
        args.ts_interval,
    )


if __name__ == "__main__":
    main()
