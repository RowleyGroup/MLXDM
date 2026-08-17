# TIP3P water ball builder

Builds a large spherical droplet of (CHARMM-modified) TIP3P water for NAMD
or OpenMM, without hitting the classic PDB numbering ceiling (5-digit atom
serial / 4-digit residue number).

## 1. Generate the water ball

```bash
python3 scripts/build_tip3p_water_ball.py <diameter_in_angstrom> \
    --outdir out --prefix waterball
```

Requires only `numpy`. Key options:

* `--density` (g/cm^3, default 0.997)
* `--max-per-segment` (default 9999 — the PDB/psfgen resid limit per segment)
* `--jitter` (default 0.15) — randomizes the starting lattice slightly
* `--seed`

This writes, into `out/`:

* `waterball.rtf` — the standard CHARMM TIP3P topology fragment (residue
  `TIP3`, atoms `OH2`/`H1`/`H2`), for psfgen
* `waterball_chunk_NNN.pdb` — coordinates, split so each chunk/segment stays
  under 9999 residues and 99999 atoms (this is what actually solves the
  "too many waters for PDB numbering" problem — psfgen's own 4-digit resid
  field is the real ceiling, not just the final PDB)
* `waterball_build_psf.tcl` — a psfgen script that assembles the chunks
* `waterball_manifest.json` — counts and file list

## 2a. NAMD route: build a PSF with psfgen

```bash
vmd -dispdev text -e out/waterball_build_psf.tcl
# or: psfgen out/waterball_build_psf.tcl
```

This produces `waterball.psf` / `waterball.pdb`. Two things to check before
using them for production MD:

1. **Numbering**: VMD/psfgen >= 1.9.3 automatically switches the combined
   output to hybrid-36 extended numbering once it exceeds 9999 residues /
   99999 atoms, and NAMD reads that format. Older builds will not — check
   your version if the combined system is large.
2. **Force field parameters**: the RTF fragment supplies atoms, bonds, and
   charges, but not Lennard-Jones parameters. Point NAMD at the standard
   CHARMM `par_water_ions.str` (part of any CHARMM36 toppar distribution) —
   it already defines `OT`/`HT` LJ parameters matching this topology, so no
   custom parameter file is needed.
3. Use NAMD's `rigidBonds water` (or `all`) with SHAKE — that's what the
   zero-force-constant H1-H2 bond in the RTF is for, matching standard
   CHARMM TIP3P setups.

## 2b. OpenMM route: build the System directly in Python

```bash
python3 scripts/load_water_ball_openmm.py out/waterball_manifest.json \
    --minimize out/waterball_minimized.pdb
```

This reads the chunk PDBs directly (no psfgen needed), and builds the
Topology/System itself:

* rigid water via distance `Constraint`s (O-H1, O-H2, H1-H2)
* charges and Lennard-Jones parameters taken from OpenMM's own bundled
  `charmm36/water.xml` (Jorgensen et al., *J. Chem. Phys.* 1983, 79, 926)
* a non-periodic cutoff (`CutoffNonPeriodic`) since a droplet, unlike a
  solvated box, isn't periodic — swap this out if you're instead going to
  embed the ball in a periodic box

## Notes / what this does *not* do

* No solute is placed in the ball — it's pure water. If you need a cavity for
  a solute, remove waters whose O is within some radius of the solute before
  running psfgen/OpenMM (not implemented here).
* No ions, no neutralization, no periodic box — this is a vacuum droplet,
  matching how spherical water balls are normally used (e.g. with a
  restraining boundary potential in NAMD, or just minimized/equilibrated as
  a standalone cluster).
* The initial lattice is a jittered simple-cubic packing at the requested
  bulk density, not a pre-equilibrated liquid structure — minimize (and
  ideally equilibrate at low temperature briefly) before production dynamics.
