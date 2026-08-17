#!/usr/bin/env python3
"""
Load a water ball produced by build_opc_water_ball.py directly into OpenMM,
building the M virtual site and OPC force-field parameters in Python -- no
psfgen/CHARMM LONEPAIR support required.

The published OPC parameters used below (charges, Lennard-Jones, and the
"average of 3 particles" virtual-site weights for the M site) are taken from
OpenMM's own bundled amber14/opc.xml, which implements Izadi, Anandakrishnan
& Onufriev, J. Phys. Chem. Lett. 2014, 5, 3863-3871:
    q(O) = 0.0,          q(H) = +0.679142,      q(M) = -1.358284
    sigma(O) = 0.3167 nm, epsilon(O) = 0.8904 kJ/mol   (H, M: epsilon = 0)
    r(O-H) = 0.087243313 nm, angle(H-O-H) = 1.808 rad (103.6 deg)
    M = 0.7046*O + 0.1477*H1 + 0.1477*H2   (ThreeParticleAverageSite)

Usage:
    python3 load_water_ball_openmm.py waterball_manifest.json --minimize out_min.pdb

This reads every chunk PDB listed in the manifest (as written by
build_opc_water_ball.py), builds an OpenMM Topology + System with rigid-water
constraints and the M virtual site, and (optionally) runs a short local
energy minimization as a smoke test before writing the result out.
"""

import argparse
import json
import math
from pathlib import Path

import openmm as mm
import openmm.app as app
import openmm.unit as unit

# --- published OPC parameters (see module docstring) ---
Q_O, Q_H, Q_M = 0.0, 0.679142, -1.358284
SIGMA_O, EPSILON_O = 0.3167, 0.8904          # nm, kJ/mol
SIGMA_HM, EPSILON_HM = 0.0891, 0.0           # nm, kJ/mol (H and M placeholders, eps=0)
R_OH_NM = 0.087243313
ANGLE_HOH_RAD = 1.808
VSITE_WEIGHTS = (0.7046, 0.1477, 0.1477)     # O, H1, H2

MASS_O = 15.99943
MASS_H = 1.008


def read_chunk_pdb(path: Path):
    """Parse a chunk PDB written by build_opc_water_ball.py. Returns
    positions (nm) for O, H1, H2 in file order (M is dropped -- OpenMM will
    rebuild it as a virtual site)."""
    coords = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            name = line[12:16].strip()
            if name == "M":
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
        assert len(coords) % 3 == 0, f"{path}: expected O/H1/H2 triplets"
        chain = topology.addChain()
        for i in range(0, len(coords), 3):
            residue = topology.addResidue("OPC", chain)
            o = topology.addAtom("O", app.element.oxygen, residue)
            h1 = topology.addAtom("H1", app.element.hydrogen, residue)
            h2 = topology.addAtom("H2", app.element.hydrogen, residue)
            m = topology.addAtom("M", None, residue)
            topology.addBond(o, h1)
            topology.addBond(o, h2)
            positions.extend(coords[i:i + 3])
            positions.append(coords[i])  # placeholder for M, replaced below
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
        atoms = list(residue.atoms())
        o, h1, h2, m = atoms  # addAtom order above: O, H1, H2, M
        o_idx = system.addParticle(MASS_O)
        h1_idx = system.addParticle(MASS_H)
        h2_idx = system.addParticle(MASS_H)
        m_idx = system.addParticle(0.0)

        nonbonded.addParticle(Q_O, SIGMA_O, EPSILON_O)
        nonbonded.addParticle(Q_H, SIGMA_HM, EPSILON_HM)
        nonbonded.addParticle(Q_H, SIGMA_HM, EPSILON_HM)
        nonbonded.addParticle(Q_M, SIGMA_HM, EPSILON_HM)

        system.addConstraint(o_idx, h1_idx, R_OH_NM)
        system.addConstraint(o_idx, h2_idx, R_OH_NM)
        system.addConstraint(h1_idx, h2_idx, d_hh)

        system.setVirtualSite(
            m_idx, mm.ThreeParticleAverageSite(o_idx, h1_idx, h2_idx, *VSITE_WEIGHTS)
        )

        # Exclude all intramolecular nonbonded interactions (rigid + virtual site).
        for a in (o_idx, h1_idx, h2_idx, m_idx):
            for b in (o_idx, h1_idx, h2_idx, m_idx):
                if a < b:
                    nonbonded.addException(a, b, 0.0, 1.0, 0.0)

    return system


def place_virtual_sites(system: mm.System, positions_nm):
    """Recompute the M-site positions from the (already correct) O/H1/H2
    positions using the same weights as the virtual site definition, so the
    initial coordinates are self-consistent."""
    positions = [mm.Vec3(*p) for p in positions_nm]
    for i in range(0, len(positions), 4):
        o, h1, h2 = positions[i], positions[i + 1], positions[i + 2]
        wo, w1, w2 = VSITE_WEIGHTS
        m = mm.Vec3(
            wo * o.x + w1 * h1.x + w2 * h2.x,
            wo * o.y + w1 * h1.y + w2 * h2.y,
            wo * o.z + w1 * h1.z + w2 * h2.z,
        )
        positions[i + 3] = m
    return positions


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", help="waterball_manifest.json from build_opc_water_ball.py")
    parser.add_argument("--minimize", metavar="OUT.pdb", default=None,
                         help="Run a short local energy minimization and write the result here.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    chunk_paths = [manifest_path.parent / name for name in manifest["chunk_files"]]

    print(f"Loading {manifest['n_waters']:,} waters from {len(chunk_paths)} chunk file(s)...")
    topology, positions_nm = build_topology_and_positions(chunk_paths)
    system = build_system(topology)
    positions = place_virtual_sites(system, positions_nm)

    print(f"Built System: {system.getNumParticles():,} particles, "
          f"{system.getNumConstraints():,} constraints, "
          f"{sum(1 for i in range(system.getNumParticles()) if system.isVirtualSite(i)):,} virtual sites.")

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
