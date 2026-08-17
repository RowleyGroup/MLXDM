# Water ball builder

Builds a large spherical droplet of rigid 3-point water for NAMD or OpenMM,
without hitting the classic PDB numbering ceiling (5-digit atom serial /
4-digit residue number).

## Choosing a water model

`water_models.py` is the single source of truth for all per-model numbers
(charges, geometry, Lennard-Jones, bond/angle force constants), taken
verbatim from OpenMM's own bundled force field files so there's one
checkable source rather than hand-copied values:

| `--model`  | Residue | Reference |
|------------|---------|-----------|
| `tip3p` (default) | `TIP3` | Jorgensen et al., *J. Chem. Phys.* 1983, 79, 926 (CHARMM-modified TIP3P, from `charmm36/water.xml`) |
| `opc3`     | `OPC3`  | Izadi & Onufriev, *J. Chem. Phys.* 2016, 145, 074501 (from `amber14/opc3.xml`) |
| `tip3pfb`  | `TP3F`  | Wang, Martinez & Pande, *J. Phys. Chem. Lett.* 2014, 5, 1885 (from `amber14/tip3pfb.xml`) |

All three are rigid, 3-site models (O, H1, H2 — no virtual site), so the
pipeline is identical regardless of which you pick.

## 1. Generate the water ball

```bash
python3 scripts/build_water_ball.py <diameter_in_angstrom> \
    --model opc3 --outdir out --prefix waterball
```

Requires only `numpy`. Key options:

* `--model` — `tip3p` (default), `opc3`, or `tip3pfb`
* `--density` (g/cm^3, default 0.997)
* `--max-per-segment` (default 9999 — the PDB/psfgen resid limit per segment)
* `--jitter` (default 0.15) — randomizes the starting lattice slightly
* `--seed`

This writes, into `out/`:

* `waterball.rtf` — CHARMM RTF topology fragment (atoms/bonds/angle/charges)
* `waterball.prm` — CHARMM-format parameters (BONDS/ANGLES/NONBONDED) for
  the chosen model — this is the piece that previously had to come from an
  external toppar distribution; now it's generated directly, for all three
  models, cross-checked against the real published CHARMM TIP3P parameter
  values (Rmin/2, epsilon, and — after fixing a unit-convention bug — the
  bond/angle force constants too)
* `waterball_chunk_NNN.pdb` — coordinates, split so each chunk/segment stays
  under 9999 residues and 99999 atoms (this is what actually solves the
  "too many waters for PDB numbering" problem — psfgen's own 4-digit resid
  field is the real ceiling, not just the final PDB)
* `waterball_build_psf.tcl` — a psfgen script that assembles the chunks
* `waterball_manifest.json` — counts, model, and file list

## 2a. NAMD route: build a PSF with psfgen

```bash
vmd -dispdev text -e out/waterball_build_psf.tcl
# or: psfgen out/waterball_build_psf.tcl
```

This produces `waterball.psf` / `waterball.pdb`. Point NAMD's `parameters`
config directive at `waterball.prm`. Two more things to check:

1. **Numbering**: VMD/psfgen >= 1.9.3 automatically switches the combined
   output to hybrid-36 extended numbering once it exceeds 9999 residues /
   99999 atoms, and NAMD reads that format. Older builds will not — check
   your version if the combined system is large.
2. Use NAMD's `rigidBonds water` (or `all`) with SHAKE — that's what the
   zero-force-constant H1-H2 bond in the RTF is for. The bond/angle force
   constants in the `.prm` only matter if you deliberately run flexible
   (non-rigid) water instead.

## 2b. OpenMM route: build the System directly in Python

```bash
python3 scripts/load_water_ball_openmm.py out/waterball_manifest.json \
    --minimize out/waterball_minimized.pdb
```

Reads the water model straight out of the manifest, so it automatically
matches whatever `--model` you built with — no separate flag needed. Builds
the Topology/System itself, with no psfgen dependency:

* rigid water via distance `Constraint`s (O-H1, O-H2, H1-H2)
* charges/Lennard-Jones from the same `water_models.py` table used to write
  the `.rtf`/`.prm` files
* a non-periodic cutoff (`CutoffNonPeriodic`) since a droplet, unlike a
  solvated box, isn't periodic — swap this out if you're instead going to
  embed the ball in a periodic box

All three models minimize to a sane, negative per-water potential energy
(tested at a 40 A / 1121-water droplet: TIP3P -41.5, OPC3 -40.1, TIP3P-FB
-46.6 kJ/mol per water) as a basic sanity check that the geometry/charges/LJ
are self-consistent.

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
