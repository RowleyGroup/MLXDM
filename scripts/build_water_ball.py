#!/usr/bin/env python3
"""
Build a large spherical droplet ("water ball") of rigid 3-point water
molecules for use with NAMD (via psfgen) or OpenMM.

Why this exists
----------------
A plain PDB file cannot represent arbitrarily large systems: the ATOM serial
number field is 5 columns (max 99999) and the residue sequence number field
is 4 columns (max 9999). A large water ball easily needs more waters than
that. psfgen has the same 9999-residues-per-segment ceiling when it *builds*
a segment (each `segment { ... }` block in a psfgen script is limited by the
same 4-column resid field). This script sidesteps the problem the way NAMD/
psfgen tutorials recommend for big solvent boxes: the waters are split into
multiple segments/chunks, each holding at most `--max-per-segment` residues
(default 9999, i.e. one full 4-digit range), and each chunk is written to
its own conventional (non-extended) PDB file. psfgen is then driven, via a
generated Tcl script, to assemble all chunks into one PSF/PDB pair. Recent
psfgen/VMD (>=1.9.3) automatically switches the *combined* output to
hybrid-36 extended numbering if it still exceeds standard PDB limits, so the
final files stay readable by NAMD.

Water model
-----------
Choose with --model {tip3p, opc3, tip3pfb} (default: tip3p). All three are
rigid 3-point models (O, H1, H2 -- no virtual site), defined in
water_models.py with parameters taken from OpenMM's own bundled force field
files. See that module's docstring for exact sources/references.

What this script produces (in --outdir)
----------------------------------------
  <prefix>.rtf                 topology fragment (CHARMM RTF format), for psfgen
  <prefix>.prm                 CHARMM-format parameters (bonds/angles/nonbonded)
  <prefix>_chunk_000.pdb, ...  water coordinates, <= max-per-segment residues each
  <prefix>_build_psf.tcl       psfgen script that turns the chunks into a PSF+PDB
  <prefix>_manifest.json       summary (counts, segments, chunk files, model)

Requirements: Python 3.8+, numpy. No network access needed.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from water_models import get_model, rtf_fragment, prm_fragment, MODELS

WATER_MOLAR_MASS = 18.01528  # g/mol
AVOGADRO = 6.02214076e23     # 1/mol


def water_template(model) -> np.ndarray:
    """Return the 3 atom positions (O, H1, H2) of one rigid water with O at
    the origin and the H-O-H bisector along +x."""
    half_angle = math.radians(model.angle_hoh_deg) / 2.0
    h1 = (model.r_oh * math.cos(half_angle), model.r_oh * math.sin(half_angle), 0.0)
    h2 = (model.r_oh * math.cos(half_angle), -model.r_oh * math.sin(half_angle), 0.0)
    return np.array([(0.0, 0.0, 0.0), h1, h2])


def random_rotation_matrices(n: int, rng: np.random.Generator) -> np.ndarray:
    """n uniformly-distributed random 3x3 rotation matrices (Shoemake 1992)."""
    u1 = rng.random(n)
    u2 = rng.random(n)
    u3 = rng.random(n)
    x = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    y = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    z = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    w = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    r = np.empty((n, 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def number_density(density_g_cm3: float) -> float:
    """Bulk water number density in molecules / Angstrom^3."""
    n_per_cm3 = density_g_cm3 * AVOGADRO / WATER_MOLAR_MASS
    return n_per_cm3 / 1.0e24  # cm^3 -> Angstrom^3


def build_lattice_centers(radius: float, spacing: float, jitter: float,
                           rng: np.random.Generator) -> np.ndarray:
    """Simple-cubic lattice of oxygen positions inside a sphere of the given
    radius, with a small random jitter (fraction of spacing) to avoid a
    perfectly crystalline starting configuration."""
    n_half = int(math.ceil(radius / spacing)) + 1
    coords_1d = np.arange(-n_half, n_half + 1) * spacing
    gx, gy, gz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    if jitter > 0:
        grid = grid + rng.uniform(-jitter * spacing, jitter * spacing, size=grid.shape)

    dist = np.linalg.norm(grid, axis=1)
    return grid[dist <= radius]


def build_water_ball(model, diameter: float, density: float, jitter: float,
                      seed: int) -> np.ndarray:
    """Return array of shape (n_waters, 3, 3): O, H1, H2 per water."""
    radius = diameter / 2.0
    spacing = number_density(density) ** (-1.0 / 3.0)
    rng = np.random.default_rng(seed)

    centers = build_lattice_centers(radius, spacing, jitter, rng)
    n = centers.shape[0]
    if n == 0:
        raise ValueError("No waters generated -- diameter is too small for the lattice spacing.")

    template = water_template(model)
    rot = random_rotation_matrices(n, rng)
    # positions[i, a, :] = rot[i] @ template[a] + centers[i]
    positions = np.einsum("nij,aj->nai", rot, template) + centers[:, None, :]
    return positions


# ---------------------------------------------------------------------------
# PDB writing
# ---------------------------------------------------------------------------
def format_atom_line(serial: int, name: str, resname: str, chain: str,
                      resid: int, xyz, element: str, segid: str) -> str:
    # Column layout matches VMD's molfile pdbplugin write_raw_pdb_record
    # exactly ("%-6s%5s %4s%c%-4s%c%4s%c   %8.3f..."): a 4-character,
    # left-justified resName field -- 3 columns (the strict PDB spec width)
    # is too narrow for CHARMM residue names like "TIP3"/"OPC3"/"TP3F" and
    # silently shifts every field after it.
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d} {name:>4s}{'':1s}{resname:<4s}{chain:1s}{resid:4d}{'':1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}      {segid:<4s}{element:>2s}\n"
    )


def write_chunk_pdb(path: Path, positions: np.ndarray, model, segid: str, chain: str) -> None:
    atom_names = [model.atom_o, "H1", "H2"]
    elements = ["O", "H", "H"]
    with open(path, "w") as fh:
        serial = 1
        for resid, water in enumerate(positions, start=1):
            for name, element, xyz in zip(atom_names, elements, water):
                fh.write(format_atom_line(serial, name, model.resname, chain, resid, xyz, element, segid))
                serial += 1
        fh.write("END\n")


def segment_id(index: int) -> str:
    """4-character segname, e.g. W000, W001, ... (max 1000 segments)."""
    if index >= 1000:
        raise ValueError(
            "More than 1000 water segments requested; increase --max-per-segment "
            "or reduce the droplet diameter."
        )
    return f"W{index:03d}"


# ---------------------------------------------------------------------------
# psfgen Tcl script
# ---------------------------------------------------------------------------
TCL_HEADER = """\
# Auto-generated by build_water_ball.py (model: {model_name})
# Run with either:
#   vmd -dispdev text -e {tcl_name}
#   psfgen {tcl_name}          (standalone psfgen binary)
#
# Verify before trusting the output for production MD: once the combined
# system exceeds 9999 residues or 99999 atoms, writepsf/writepdb below need
# psfgen/VMD >= 1.9.3 to switch to hybrid-36 extended numbering automatically
# (there is no explicit "extended" keyword to pass -- writepsf only takes
# [charmm|x-plor] [cmap|nocmap] [nopatches]). Older builds will silently
# truncate/wrap numbers instead.
#
# NAMD "parameters" line should point at {prm_name} (bonds/angles/nonbonded
# for this model) in addition to any other parameter files your system needs.

package require psfgen
resetpsf
topology {rtf_name}

"""

TCL_SEGMENT = """\
segment {segid} {{
    auto none
    first none
    last none
    pdb {pdb_name}
}}
coordpdb {pdb_name} {segid}

"""

TCL_FOOTER = """\
writepsf x-plor {psf_name}
writepdb {pdb_name}
"""


def write_tcl_script(path: Path, model_name: str, rtf_name: str, prm_name: str,
                      chunk_files, psf_name: str, combined_pdb_name: str) -> None:
    with open(path, "w") as fh:
        fh.write(TCL_HEADER.format(model_name=model_name, tcl_name=path.name,
                                    rtf_name=rtf_name, prm_name=prm_name))
        for segid, pdb_name in chunk_files:
            fh.write(TCL_SEGMENT.format(segid=segid, pdb_name=pdb_name))
        fh.write(TCL_FOOTER.format(psf_name=psf_name, pdb_name=combined_pdb_name))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build a large spherical water droplet for NAMD/OpenMM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("diameter", type=float, help="Droplet diameter in Angstroms.")
    parser.add_argument("--model", choices=sorted(MODELS), default="tip3p",
                         help="Water model (default: tip3p). See water_models.py.")
    parser.add_argument("--density", type=float, default=0.997,
                         help="Target bulk density in g/cm^3 (default: 0.997).")
    parser.add_argument("--max-per-segment", type=int, default=9999,
                         help="Max waters per psfgen segment/chunk file (default: 9999, "
                              "the PDB 4-column resid limit).")
    parser.add_argument("--jitter", type=float, default=0.15,
                         help="Random per-water positional jitter, as a fraction of the "
                              "lattice spacing (default: 0.15; 0 disables jitter).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory.")
    parser.add_argument("--prefix", type=str, default="waterball", help="Output filename prefix.")
    args = parser.parse_args()

    if args.diameter <= 0:
        parser.error("diameter must be positive")
    if not (0 < args.max_per_segment <= 9999):
        parser.error("--max-per-segment must be in (0, 9999]")

    model = get_model(args.model)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Building {model.display_name} water ball: diameter={args.diameter:.1f} A, "
          f"density={args.density:.3f} g/cm^3, seed={args.seed}")
    positions = build_water_ball(model, args.diameter, args.density, args.jitter, args.seed)
    n_waters = positions.shape[0]
    n_atoms = n_waters * 3
    print(f"Generated {n_waters:,} waters ({n_atoms:,} atoms).")

    # --- RTF + PRM ---
    rtf_path = outdir / f"{args.prefix}.rtf"
    rtf_path.write_text(rtf_fragment(model))
    print(f"Wrote topology fragment: {rtf_path}")

    prm_path = outdir / f"{args.prefix}.prm"
    prm_path.write_text(prm_fragment(model))
    print(f"Wrote parameter file: {prm_path}")

    # --- chunked PDBs, one psfgen segment each ---
    max_per = args.max_per_segment
    chunk_files = []
    for i, start in enumerate(range(0, n_waters, max_per)):
        chunk = positions[start:start + max_per]
        segid = segment_id(i)
        chunk_name = f"{args.prefix}_chunk_{i:03d}.pdb"
        write_chunk_pdb(outdir / chunk_name, chunk, model, segid, chain="A")
        chunk_files.append((segid, chunk_name))
    print(f"Wrote {len(chunk_files)} chunk PDB file(s), <= {max_per} waters each "
          f"(largest segment: {min(max_per, n_waters):,} residues, "
          f"{min(max_per, n_waters) * 3:,} atoms -- both within standard PDB limits).")

    # --- psfgen Tcl script ---
    tcl_path = outdir / f"{args.prefix}_build_psf.tcl"
    psf_name = f"{args.prefix}.psf"
    combined_pdb_name = f"{args.prefix}.pdb"
    write_tcl_script(tcl_path, model.display_name, rtf_path.name, prm_path.name,
                      chunk_files, psf_name, combined_pdb_name)
    print(f"Wrote psfgen script: {tcl_path}")

    # --- manifest ---
    total_charge = n_waters * (model.q_o + 2 * model.q_h)
    manifest = {
        "water_model": model.key,
        "water_model_name": model.display_name,
        "diameter_angstrom": args.diameter,
        "density_g_cm3": args.density,
        "n_waters": n_waters,
        "n_atoms": n_atoms,
        "n_segments": len(chunk_files),
        "max_per_segment": max_per,
        "seed": args.seed,
        "jitter": args.jitter,
        "topology_file": rtf_path.name,
        "parameter_file": prm_path.name,
        "chunk_files": [name for _, name in chunk_files],
        "psfgen_tcl": tcl_path.name,
        "expected_psf": psf_name,
        "expected_pdb": combined_pdb_name,
        "total_charge": total_charge,
    }
    manifest_path = outdir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest: {manifest_path}")

    print("\nNext steps:")
    print(f"  1. vmd -dispdev text -e {tcl_path.name}   (or: psfgen {tcl_path.name})")
    print(f"     -> produces {psf_name} / {combined_pdb_name}")
    print(f"  2. Point NAMD's \"parameters\" config directive at {prm_path.name}.")
    print("  3. For OpenMM, scripts/load_water_ball_openmm.py can build a System directly "
          "from the chunk PDBs, using the same water_models.py parameters, without going "
          "through psfgen.")


if __name__ == "__main__":
    main()
