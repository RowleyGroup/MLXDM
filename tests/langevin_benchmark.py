"""GPU-resident Langevin MD benchmark over a directory of water-ball xyz files.

Mirrors the CLI/output layout of the ASE-based benchmark (ase.md.langevin.Langevin
wrapping model.ase(), one host<->device round trip per MD step) but drives
torchanipbe0.md.Langevin directly: species/coordinates/velocities/masses/forces
stay as torch tensors on --device for the whole 100000-step run, and the only
host<->device syncs are the periodic time-series/trajectory logging points
(controlled by --ts-interval/--traj-interval), not every step. Comparing wall
times between the two scripts on the same xyz files/device isolates the cost
of the per-step ase.Atoms/Calculator round trip.
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
from ase import Atoms
from ase.io.trajectory import Trajectory

DEFAULT_XYZ_DIR = "/lustre06/project/6060902/crowley/timing/xyz"

T = 298.15
MD_STEPS = 100000
TIMESTEP_FS = 1.0
FRICTION = 0.002
WARMUP_STEPS = 10


def build_model(model_name, device):
    import torchanipbe0

    device = torch.device(device)
    if model_name == "anipbe0":
        model = torchanipbe0.models.ANIPBE0_2x(periodic_table_index=False).to(device)
    else:
        model = torchanipbe0.models.ANIPBE0_2x_MLXDM_2x(device)

    param_types = {p.device.type for p in model.parameters()}
    if param_types != {device.type}:
        raise RuntimeError(
            f"Expected all model parameters on device type '{device.type}', "
            f"found {param_types}"
        )
    print(f"Confirmed model parameters on device: {device}")
    return model, device


def species_tensor(model, symbols, device):
    """species_to_tensor lives on model.ani_model for the combined
    ANIDispersion model, and directly on model for the plain ANIPBE0_2x
    ensemble -- same distinction as tests/md_gpu.py.
    """
    to_tensor = model.ani_model.species_to_tensor if hasattr(model, "ani_model") \
        else model.species_to_tensor
    return to_tensor(symbols).unsqueeze(0).to(device)


def maxwell_boltzmann_velocities(masses, temperature_K, generator=None):
    """Sample per-atom velocities from the Maxwell-Boltzmann distribution,
    in the same (Angstrom, fs, AMU, Hartree) unit convention used internally
    by torchanipbe0.md.Langevin's own O-step noise (sigma = sqrt(kT/m)).
    """
    from torchanipbe0 import units

    kT = units.BOLTZMANN_CONSTANT * temperature_K
    sigma = torch.sqrt(kT / masses).view(1, -1, 1) / units.HARTREE_ANGSTROM_AMU_TIME_TO_FS
    noise = torch.randn(1, masses.shape[0], 3, device=masses.device, dtype=masses.dtype,
                         generator=generator)
    return sigma * noise


def kinetic_temperature(dyn, n_dof):
    from torchanipbe0 import units
    return 2.0 * dyn.kinetic_energy().item() / (n_dof * units.BOLTZMANN_CONSTANT)


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


def make_callback(dyn, n_dof, symbols, csv_path, traj_path, start_time,
                   ts_interval, traj_interval):
    """Build a single dyn.run() callback that logs a time-series row every
    ts_interval steps and a trajectory frame every traj_interval steps.
    Both operations call .item()/.cpu(), so they are the only points in the
    run that synchronize the GPU -- everything else stays device-resident.
    """
    from torchanipbe0 import units

    ts_file = open(csv_path, mode="w", newline="")
    writer = csv.writer(ts_file)
    writer.writerow(["step", "elapsed_seconds", "epot_eV", "temperature_K"])
    traj = Trajectory(traj_path, "w")

    def callback(step, coordinates, velocities, energy, forces):
        log_now = (step + 1) % ts_interval == 0
        write_frame = (step + 1) % traj_interval == 0
        if not (log_now or write_frame):
            return

        if log_now:
            elapsed = time.perf_counter() - start_time
            epot_eV = energy.item() * units.HARTREE_TO_EV
            temp = kinetic_temperature(dyn, n_dof)
            writer.writerow([step + 1, f"{elapsed:.4f}", f"{epot_eV:.6f}", f"{temp:.2f}"])
            ts_file.flush()

        if write_frame:
            atoms = Atoms(symbols=symbols, positions=coordinates[0].detach().cpu().numpy())
            traj.write(atoms)

    def close():
        ts_file.close()
        traj.close()

    return callback, close


def run_benchmark(model, device, xyz_files, csv_filename, outdir, traj_interval, ts_interval):
    import torchanipbe0
    from torchanipbe0 import units
    from torchanipbe0.md import Langevin

    os.makedirs(outdir, exist_ok=True)

    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["xyz_file", "n_atoms", "n_waters", "time_seconds"])

        print(f"Starting GPU-resident Langevin benchmark over {len(xyz_files)} "
              f"water-ball structures, {MD_STEPS} steps each...")

        for path in xyz_files:
            atoms = ase.io.read(path)
            symbols = atoms.get_chemical_symbols()
            n_atoms = len(atoms)
            n_waters = n_waters_from_filename(path)
            if n_waters is None:
                n_waters = n_atoms // 3

            name = os.path.basename(path)
            stem = os.path.splitext(name)[0]
            print(f"Running MD for {name} ({n_atoms} atoms)...")

            coordinates = torch.tensor(atoms.get_positions(), dtype=torch.float32,
                                        device=device).unsqueeze(0)
            masses = torch.tensor(atoms.get_masses(), dtype=torch.float32, device=device)
            species = species_tensor(model, symbols, device)

            generator = torch.Generator(device=device).manual_seed(0)
            velocities = maxwell_boltzmann_velocities(masses, T, generator=generator)

            n_dof = 3 * n_atoms
            dyn = Langevin(model, species, coordinates, velocities, masses,
                            timestep=TIMESTEP_FS, temperature=T, friction=FRICTION,
                            generator=generator)

            # Untimed warm-up steps: absorb one-time costs (CUDA context/kernel
            # JIT, cuDNN/cuBLAS autotuning for these tensor shapes) so they
            # don't get charged to the timed run below, especially on the
            # first structure of a fresh process.
            dyn.run(WARMUP_STEPS)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            time_initial = time.perf_counter()
            callback, close_logs = make_callback(
                dyn, n_dof, symbols,
                os.path.join(outdir, f"{stem}.timeseries.csv"),
                os.path.join(outdir, f"{stem}.traj"),
                time_initial, ts_interval, traj_interval,
            )

            dyn.run(MD_STEPS, callback=callback)
            close_logs()

            time_final = time.perf_counter()
            elapsed_time = time_final - time_initial

            print(f"Completed {name} in {elapsed_time:.2f} seconds "
                  f"({MD_STEPS / elapsed_time:.1f} steps/s).")

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
                         help="Output CSV path (default: simulation_timings_<model>_gpu.csv)")
    parser.add_argument("--outdir", default=".",
                         help="Directory to write per-structure trajectory/time-series "
                              "files to (default: %(default)s)")
    parser.add_argument("--traj-interval", type=int, default=1000,
                         help="Steps between trajectory frames (default: %(default)s)")
    parser.add_argument("--ts-interval", type=int, default=100,
                         help="Steps between time-series log rows (default: %(default)s)")
    args = parser.parse_args()

    model, device = build_model(args.model, args.device)

    if args.xyz_file:
        if not os.path.isfile(args.xyz_file):
            print(f"Error: '{args.xyz_file}' not found.")
            sys.exit(1)
        xyz_files = [args.xyz_file]
        stem = os.path.splitext(os.path.basename(args.xyz_file))[0]
        default_csv = f"simulation_timings_{args.model}_gpu_{stem}.csv"
    else:
        xyz_files = find_xyz_files(args.xyz_dir, args.pattern)
        if args.limit is not None:
            xyz_files = xyz_files[:args.limit]
        default_csv = f"simulation_timings_{args.model}_gpu.csv"

    run_benchmark(
        model,
        device,
        xyz_files,
        args.csv or default_csv,
        args.outdir,
        args.traj_interval,
        args.ts_interval,
    )


if __name__ == "__main__":
    main()
