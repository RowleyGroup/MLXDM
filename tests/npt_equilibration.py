import os
import sys
import time
import numpy as np

import ase
import ase.io
from ase import units
from ase.io.trajectory import Trajectory

from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
)

import torchanipbe0


def main():

    # ---------------------------------------------------------
    # Parse CLI Arguments
    # ---------------------------------------------------------

    prefix = sys.argv[1] if len(sys.argv) > 1 else "ccl4"
    cycle = sys.argv[2] if len(sys.argv) > 2 else "0"

    target_temp = 298.15  # K (Room temperature)
    target_pressure = 1.01325 * units.bar  # 1 atm

    walltime_limit_sec = 2.75 * 3600

    # ---------------------------------------------------------
    # Find Most Recent Restart For This Replicate/Cycle
    # ---------------------------------------------------------
    #
    # Restart files are named "{prefix}.{cycle}.{restart_number}
    # .restart.json". For a given replicate/cycle, keep scanning
    # upward until the next restart_number is missing, so we
    # always resume from the highest-numbered json available.
    #
    # ---------------------------------------------------------

    restart_number = 0

    while os.path.exists(
        f"{prefix}.{cycle}.{restart_number}.restart.json"
    ):
        restart_number += 1

    if restart_number == 0:
        raise FileNotFoundError(
            f"Missing starting structure file: "
            f"{prefix}.{cycle}.0.restart.json"
        )

    json_path = (
        f"{prefix}.{cycle}.{restart_number - 1}.restart.json"
    )

    json_restart = (
        f"{prefix}.{cycle}.{restart_number}.restart.json"
    )

    out_prefix = (
        f"{prefix}.{cycle}.{restart_number}"
    )

    print()
    print("Input structure :", json_path)
    print("Output restart  :", json_restart)
    print()

    # ---------------------------------------------------------
    # Read Structure
    # ---------------------------------------------------------

    atoms = ase.io.read(json_path, format="json")

    velocities = atoms.get_velocities()

    if velocities is None or np.allclose(velocities, 0.0):

        print(
            "No velocities found. "
            "Initializing Maxwell-Boltzmann distribution."
        )

        MaxwellBoltzmannDistribution(
            atoms,
            temperature_K=target_temp,
        )

        Stationary(atoms)

    # ---------------------------------------------------------
    # Attach ANI Calculator
    # ---------------------------------------------------------

    model = torchanipbe0.models.ANIPBE0_2x_MLXDM_2x()

    atoms.calc = model.ase()

    # ---------------------------------------------------------
    # Print Initial State
    # ---------------------------------------------------------

    pressure_gpa = np.nan

    try:
        stress = atoms.get_stress()

        pressure = -(
            stress[0] +
            stress[1] +
            stress[2]
        ) / 3.0

        pressure_gpa = pressure / units.GPa

    except Exception as e:
        print("Could not evaluate initial pressure:", e)

    print(
        f"Initial temperature : "
        f"{atoms.get_temperature():.2f} K"
    )

    print(
        f"Initial volume      : "
        f"{atoms.get_volume():.2f} A^3"
    )

    print(
        f"Initial pressure    : "
        f"{pressure_gpa:.3f} GPa"
    )

    print()

    # ---------------------------------------------------------
    # NPT Berendsen Settings
    # ---------------------------------------------------------
    #
    # Compressibility for liquid CCl4 at 25C is ~1.05e-9 Pa^-1
    #
    # taup:
    # large value = slow volume adjustment
    #
    # taut:
    # gentle thermostat coupling
    #
    # ---------------------------------------------------------

    compressibility = 10.5e-10 / units.Pascal

    dyn = NPTBerendsen(
        atoms,
        timestep=0.5 * units.fs,
        temperature_K=target_temp,
        pressure_au=target_pressure,
        taut=100.0 * units.fs,
        taup=2000.0 * units.fs,
        compressibility_au=compressibility,
        logfile=f"{out_prefix}.log",
    )

    # ---------------------------------------------------------
    # Trajectory
    # ---------------------------------------------------------

    traj = Trajectory(
        f"{out_prefix}.traj",
        "w",
        atoms,
    )

    dyn.attach(traj.write, interval=1000)

    # ---------------------------------------------------------
    # Density Logging
    # ---------------------------------------------------------
    #
    # One CSV per replicate/cycle (not per restart segment), so
    # that resuming a cycle across multiple restarts appends to
    # the same continuous density history instead of starting a
    # new file each time.
    #
    # ---------------------------------------------------------

    density_file = f"{prefix}.{cycle}.density.csv"

    if not os.path.exists(density_file):
        with open(density_file, "w") as f:
            f.write(
                "Step,"
                "Volume_A3,"
                "Density_g_cm3,"
                "Temperature_K\n"
            )

    system_mass_g = (
        np.sum(atoms.get_masses())
        * 1.66053906660e-24
    )

    start_time = time.time()

    def monitor_and_log():

        volume_A3 = atoms.get_volume()

        density = (
            system_mass_g
            / (volume_A3 * 1.0e-24)
        )

        temp = atoms.get_temperature()

        step = dyn.get_number_of_steps()

        with open(density_file, "a") as f:
            f.write(
                f"{step},"
                f"{volume_A3:.4f},"
                f"{density:.6f},"
                f"{temp:.2f}\n"
            )

        if step % 100 == 0:

            try:

                stress = atoms.get_stress()

                pressure = -(
                    stress[0]
                    + stress[1]
                    + stress[2]
                ) / 3.0

                pressure_gpa = (
                    pressure / units.GPa
                )

                print(
                    f"Step {step:8d} | "
                    f"T = {temp:8.2f} K | "
                    f"P = {pressure_gpa:8.3f} GPa | "
                    f"rho = {density:8.4f} g/cm3"
                )

            except Exception:
                pass

        elapsed = time.time() - start_time

        if elapsed >= walltime_limit_sec:

            print()
            print(
                "Walltime limit reached. "
                "Stopping cleanly."
            )
            print()

            dyn.max_steps = dyn.nsteps

    dyn.attach(
        monitor_and_log,
        interval=100,
    )

    # ---------------------------------------------------------
    # Run
    # ---------------------------------------------------------

    print("Starting NPT-Berendsen equilibration")
    print()

    dyn.run(100_000_000)

    # ---------------------------------------------------------
    # Write Restart
    # ---------------------------------------------------------

    ase.io.write(
        json_restart,
        atoms,
        format="json",
    )

    print()
    print(
        f"Restart written to: "
        f"{json_restart}"
    )


if __name__ == "__main__":
    main()
