"""Shared timing loop for the MACEPBE0 / MACEXDM benchmark scripts.

Grows a chain of pentane molecules and, every 10 molecules, times a short
Langevin MD run with the given ASE calculator already attached. Timings are
written to a CSV file incrementally so partial results survive a job that
gets killed or times out partway through.
"""
import sys
import time
import csv
import ase.io
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

T = 298.15
MD_STEPS = 10000


def run_benchmark(calc, max_n_pentane, csv_filename, xyz_path="xyz/pentane.xyz"):
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

                # Get initial potential energy
                _ = atoms.get_potential_energy()

                # Setup and run Langevin dynamics
                MaxwellBoltzmannDistribution(atoms, temperature_K=T)
                dyn = Langevin(atoms, 1 * units.fs, T * units.kB, 0.002)
                dyn.run(MD_STEPS)

                time_final = time.perf_counter()
                elapsed_time = time_final - time_initial

                print(f"Completed {i} molecules in {elapsed_time:.2f} seconds.")

                writer.writerow([i, elapsed_time])
                f.flush()

    print(f"All timings successfully saved to {csv_filename}")
