#!/usr/bin/env python3
"""
Load a water ball produced by build_tip3p_water_ball.py directly into
OpenMM, building the System (rigid constraints + force-field parameters) in
Python -- no psfgen/CHARMM parameter file required.

The TIP3P parameters used below (charges and Lennard-Jones) are the standard
CHARMM-modified TIP3P values, taken from OpenMM's own bundled
charmm36/water.xml (Jorgensen et al., J. Chem. Phys. 1983, 79, 926):
    q(OH2) = -0.834,  q(H1) = q(H2) = +0.417
    sigma(O) = 0.31506 nm, epsilon(O) = 0.6364 kJ/mol
    sigma(H) = 0.04000 nm, epsilon(H) = 0.1925 kJ/mol
    r(O-H) = 0.09572 nm, angle(H-O-H) = 104.52 deg -> r(H-H) = 0.15139 nm

Usage:
    python3 load_water_ball_openmm.py waterball_manifest.json --minimize out_min.pdb

This reads every chunk PDB listed in the manifest (as written by
build_tip3p_water_ball.py), builds an OpenMM Topology + System with rigid
water constraints, and (optionally) runs a short local energy minimization
as a smoke test before writing the result out.
"""

import argparse
import json
import math
from pathlib import Path

import openmm as mm
import openmm.app as app
import openmm.unit as unit

# --- standard CHARMM-modified TIP3P parameters (see module docstring) ---
Q_O, Q_H = -0.834, 0.417
SIGMA_O, EPSILON_O = 0.31506, 0.6364   # nm, kJ/mol
SIGMA_H, EPSILON_H = 0.04000, 0.1925   # nm, kJ/mol
R_OH_NM = 0.09572
ANGLE_HOH_RAD = math.radians(104.52)

MASS_O = 15.99940
MASS_H = 1.00800


def read_chunk_pdb(path: Path):
    """Parse a chunk PDB written by build_tip3p_water_ball.py. Returns a
    flat list of (x, y, z) positions in nm, in file order (OH2, H1, H2 per
    residue)."""
    coords = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            x = float(line[30:38]) / 10.0
            y = float(line[38:46]) / 10.0
            z = float(line[46:54]) / 10.0
            coords.append((x, y, z))
    return coords


def build_topology_and_positions(chunk_paths):
    topology = app.Topology()
    positions = []
    for path in chunk_paths:
        coords = read_chunk_pdb(path)
        assert len(coords) % 3 == 0, f"{path}: expected OH2/H1/H2 triplets"
        chain = topology.addChain()
        for i in range(0, len(coords), 3):
            residue = topology.addResidue("TIP3", chain)
            o = topology.addAtom("OH2", app.element.oxygen, residue)
            h1 = topology.addAtom("H1", app.element.hydrogen, residue)
            h2 = topology.addAtom("H2", app.element.hydrogen, residue)
            topology.addBond(o, h1)
            topology.addBond(o, h2)
            positions.extend(coords[i:i + 3])
    return topology, positions


def build_system(topology: app.Topology) -> mm.System:
    system = mm.System()
    # A water "ball"/droplet is a finite, non-periodic system (unlike a
    # periodic solvent box), so use a non-periodic cutoff rather than PME.
    # Waters far apart in a large droplet interact weakly enough that a
    # finite cutoff is the standard approximation here; switch to NoCutoff
    # (exact, O(N^2)) only for small systems where that cost is acceptable.
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.CutoffNonPeriodic)
    nonbonded.setCutoffDistance(1.2 * unit.nanometer)
    system.addForce(nonbonded)

    d_hh = 2 * R_OH_NM * math.sin(ANGLE_HOH_RAD / 2.0)

    for residue in topology.residues():
        o, h1, h2 = residue.atoms()
        o_idx = system.addParticle(MASS_O)
        h1_idx = system.addParticle(MASS_H)
        h2_idx = system.addParticle(MASS_H)

        nonbonded.addParticle(Q_O, SIGMA_O, EPSILON_O)
        nonbonded.addParticle(Q_H, SIGMA_H, EPSILON_H)
        nonbonded.addParticle(Q_H, SIGMA_H, EPSILON_H)

        system.addConstraint(o_idx, h1_idx, R_OH_NM)
        system.addConstraint(o_idx, h2_idx, R_OH_NM)
        system.addConstraint(h1_idx, h2_idx, d_hh)

        # Exclude all intramolecular nonbonded interactions (fully rigid molecule).
        nonbonded.addException(o_idx, h1_idx, 0.0, 1.0, 0.0)
        nonbonded.addException(o_idx, h2_idx, 0.0, 1.0, 0.0)
        nonbonded.addException(h1_idx, h2_idx, 0.0, 1.0, 0.0)

    return system


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", help="waterball_manifest.json from build_tip3p_water_ball.py")
    parser.add_argument("--minimize", metavar="OUT.pdb", default=None,
                         help="Run a short local energy minimization and write the result here.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    chunk_paths = [manifest_path.parent / name for name in manifest["chunk_files"]]

    print(f"Loading {manifest['n_waters']:,} waters from {len(chunk_paths)} chunk file(s)...")
    topology, positions = build_topology_and_positions(chunk_paths)
    system = build_system(topology)

    print(f"Built System: {system.getNumParticles():,} particles, "
          f"{system.getNumConstraints():,} constraints.")

    if args.minimize:
        integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                                   0.001 * unit.picoseconds)
        platform = mm.Platform.getPlatformByName("Reference")
        simulation = app.Simulation(topology, system, integrator, platform)
        simulation.context.setPositions(positions)
        print("Minimizing...")
        simulation.minimizeEnergy(maxIterations=200)
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        print(f"Potential energy after minimization: {state.getPotentialEnergy()}")
        with open(args.minimize, "w") as fh:
            app.PDBFile.writeFile(topology, state.getPositions(), fh)
        print(f"Wrote {args.minimize}")


if __name__ == "__main__":
    main()
