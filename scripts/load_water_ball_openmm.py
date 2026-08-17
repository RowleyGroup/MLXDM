#!/usr/bin/env python3
"""
Load a water ball produced by build_water_ball.py directly into OpenMM,
building the System (rigid constraints + force-field parameters) in Python
-- no psfgen/CHARMM parameter file required.

Reads the water model (tip3p / opc3 / tip3pfb) from the manifest and pulls
its parameters from water_models.py -- the same source of truth used to
generate the .rtf/.prm files -- so this is guaranteed consistent with
whatever model build_water_ball.py was run with.

Usage:
    python3 load_water_ball_openmm.py waterball_manifest.json --minimize out_min.pdb
"""

import argparse
import json
import math
from pathlib import Path

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from water_models import get_model


def read_chunk_pdb(path: Path):
    """Parse a chunk PDB written by build_water_ball.py. Returns a flat list
    of (x, y, z) positions in nm, in file order (O, H1, H2 per residue)."""
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


def build_topology_and_positions(chunk_paths, model):
    topology = app.Topology()
    positions = []
    for path in chunk_paths:
        coords = read_chunk_pdb(path)
        assert len(coords) % 3 == 0, f"{path}: expected O/H1/H2 triplets"
        chain = topology.addChain()
        for i in range(0, len(coords), 3):
            residue = topology.addResidue(model.resname, chain)
            o = topology.addAtom(model.atom_o, app.element.oxygen, residue)
            h1 = topology.addAtom("H1", app.element.hydrogen, residue)
            h2 = topology.addAtom("H2", app.element.hydrogen, residue)
            topology.addBond(o, h1)
            topology.addBond(o, h2)
            positions.extend(coords[i:i + 3])
    return topology, positions


def build_system(topology: app.Topology, model) -> mm.System:
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

    r_oh_nm = model.r_oh / 10.0
    d_hh_nm = model.d_hh / 10.0

    for residue in topology.residues():
        o, h1, h2 = residue.atoms()
        o_idx = system.addParticle(model.mass_o)
        h1_idx = system.addParticle(model.mass_h)
        h2_idx = system.addParticle(model.mass_h)

        nonbonded.addParticle(model.q_o, model.sigma_o_nm, model.epsilon_o_kjmol)
        nonbonded.addParticle(model.q_h, model.sigma_h_nm, model.epsilon_h_kjmol)
        nonbonded.addParticle(model.q_h, model.sigma_h_nm, model.epsilon_h_kjmol)

        system.addConstraint(o_idx, h1_idx, r_oh_nm)
        system.addConstraint(o_idx, h2_idx, r_oh_nm)
        system.addConstraint(h1_idx, h2_idx, d_hh_nm)

        # Exclude all intramolecular nonbonded interactions (fully rigid molecule).
        nonbonded.addException(o_idx, h1_idx, 0.0, 1.0, 0.0)
        nonbonded.addException(o_idx, h2_idx, 0.0, 1.0, 0.0)
        nonbonded.addException(h1_idx, h2_idx, 0.0, 1.0, 0.0)

    return system


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", help="waterball_manifest.json from build_water_ball.py")
    parser.add_argument("--minimize", metavar="OUT.pdb", default=None,
                         help="Run a short local energy minimization and write the result here.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    chunk_paths = [manifest_path.parent / name for name in manifest["chunk_files"]]
    model = get_model(manifest["water_model"])

    print(f"Loading {manifest['n_waters']:,} waters ({model.display_name}) "
          f"from {len(chunk_paths)} chunk file(s)...")
    topology, positions = build_topology_and_positions(chunk_paths, model)
    system = build_system(topology, model)

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
