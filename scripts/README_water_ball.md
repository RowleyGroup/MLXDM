# OPC water ball builder

Builds a large spherical droplet of OPC water for NAMD or OpenMM, without
hitting the classic PDB numbering ceiling (5-digit atom serial / 4-digit
residue number).

## 1. Generate the water ball

```bash
python3 scripts/build_opc_water_ball.py <diameter_in_angstrom> \
    --outdir out --prefix waterball
```

Requires only `numpy`. Key options:

* `--density` (g/cm^3, default 0.997)
* `--max-per-segment` (default 9999 — the PDB/psfgen resid limit per segment)
* `--jitter` (default 0.15) — randomizes the starting lattice slightly
* `--seed`

This writes, into `out/`:

* `waterball.rtf` — the OPC topology fragment (as supplied), for psfgen
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

This produces `waterball.psf` / `waterball.pdb`. Two things to verify before
using them for production MD, called out in the generated Tcl script too:

1. **Numbering**: VMD/psfgen >= 1.9.3 automatically switches the combined
   output to hybrid-36 extended numbering once it exceeds 9999 residues /
   99999 atoms, and NAMD reads that format. Older builds will not — check
   your version if the combined system is large.
2. **The M virtual site**: the supplied RTF fragment defines atoms and bonds
   only. It has no `LONEPAIR`/virtual-site reconstruction rule, and psfgen's
   support for reading/writing CHARMM `LONEPAIR` records is version-dependent
   (some builds fail with "FAILED TO RECOGNIZE LONEPAIR"). The M atom's
   *initial* coordinates in `waterball.pdb` are correct (built from OPC's
   rigid geometry), but confirm your psfgen actually emits a working
   lone-pair record before running dynamics — otherwise NAMD has no rule for
   moving M each step. If it doesn't, either patch it in with real CHARMM,
   or use the OpenMM route below, which needs no PSF-level lone-pair support
   at all.
3. **Force field parameters**: the RTF fragment has no charges/LJ beyond the
   placeholder ones in the fragment itself (and it has no charge/LJ line for
   M at all). You still need a real `.prm`/`.str` for OPC (e.g. from the
   CHARMM or AMBER water_ions toppar files) before running NAMD.

## 2b. OpenMM route: build the System directly in Python

```bash
python3 scripts/load_water_ball_openmm.py out/waterball_manifest.json \
    --minimize out/waterball_minimized.pdb
```

This reads the chunk PDBs directly (no psfgen needed), and builds the
Topology/System itself:

* rigid water via distance `Constraint`s (O-H1, O-H2, H1-H2)
* the M site as an OpenMM `ThreeParticleAverageSite` virtual site
* charges and Lennard-Jones parameters taken from OpenMM's own bundled
  `amber14/opc.xml` (Izadi, Anandakrishnan & Onufriev, *J. Phys. Chem. Lett.*
  2014, 5, 3863)
* a non-periodic cutoff (`CutoffNonPeriodic`) since a droplet, unlike a
  solvated box, isn't periodic — swap this out if you're instead going to
  embed the ball in a periodic box

This path is self-contained and doesn't depend on psfgen's lone-pair support
at all, which is why it's the more robust option for OPC specifically.

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
