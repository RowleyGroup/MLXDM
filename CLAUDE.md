# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MLXDM is a modified fork of [TorchANI](https://github.com/aiqm/torchani) that adds a second neural
network predicting atomic dispersion coefficients (C6, C8, C10) via the exchange-hole dipole moment
(XDM) model, and combines them with an ANI-style short-range NNP (including a PBE0-trained variant)
to produce a potential energy surface with an explicit long-range London dispersion term. The
installable package is named `torchanipbe0` (see `setup.py`), even though the repo is called MLXDM.

## Commands

There is no build system, linter, or test runner configured (no CI config, no pytest/setup.cfg,
no flake8/lint config). Development is done by installing the package and running the scripts in
`tests/` manually.

```bash
# Install the package locally (editable is fine too: pip install -e .)
pip install .

# Runtime deps: torch, ase>=3.22, h5py>=3.6, requests>=2.26, lark
```

`tests/` is a set of demo/benchmark scripts, not an automated test suite — there is no pytest
integration and no assertions. Run them individually to sanity-check a change:

```bash
python tests/md.py           # BFGS geometry optimization demo, expects a formic-acid-dimer-like output
python tests/runtime.py      # Benchmarks forward-pass timing of several models on CPU/CUDA
python tests/simulation.py   # Unit-cell (graphite) optimization example
python tests/model_test.py   # Exercises many models.py factory functions end-to-end
```

Note: `tests/simulation.py` and parts of `tests/runtime.py`/`tests/model_test.py` call model
constructors (e.g. `ANIPBE0_Dispersion_Constant_Coef`, `MLXDM_simple`, `c6_simple_energy`) that do
not currently exist in `torchanipbe0/models.py`. Treat these as stale/aspirational examples rather
than a working contract — check `models.py` for the actual current factory function names before
relying on a test script.

Several scripts hardcode `torch.device('cuda')`; a CUDA GPU is expected to be available when
running them as-is.

## Architecture

### Two neural networks feeding one energy

The model is the composition of two independently trained torch modules, joined by `ANIDispersion`
(`torchanipbe0/dispersion/nn.py`):

1. **Short-range NNP** (`ani_model`) — a `BuiltinModel`/`BuiltinEnsemble` (or the `*2`/`*3` variants,
   see below) from `torchanipbe0/models.py`. This is the TorchANI-style pipeline:
   `SpeciesConverter -> AEVComputer -> ANIModel/Ensemble -> EnergyShifter`.
2. **Dispersion model** (`disp_model`) — a `DispersionLayer` (`torchanipbe0/dispersion/nn.py`) that
   reuses the *same* `AEVComputer` from the short-range model (AEVs are computed once conceptually,
   though currently each sub-model calls its own `aev_computer` instance with identical constants)
   and feeds them into four small per-element MLPs (`CoefficientLayer`, one each for "m1", "m2",
   "m3", "v") whose outputs are combined via the XDM formulas (`C6CombineLayer`, `C8CombineLayer`,
   `C10CombineLayer`) into pairwise C6/C8/C10 dispersion energies, damped by a Becke-Johnson-style
   van der Waals radius (`vanderWaalsLayer`) and summed (`EnergyLayer`).

`ANIDispersion.forward` just adds the two energies. `.ase()` wraps the combined module in
`torchanipbe0/ase.py`'s `Calculator` for use with ASE `Atoms.set_calculator(...)`.

### Model zoo (`torchanipbe0/models.py`)

All public models are built by factory functions (not classes to instantiate directly), e.g.
`ANI1x()`, `ANIPBE0()`, `ANIPBE0_2x()`, `MLXDM()`, `MLXDM_2x()`, and the combined
`ANIPBE0_MLXDM()`, `ANIPBE0_2x_MLXDM_2x()`, `ANI1x_MLXDM()`, `ANI1ccx_MLXDM()`. Each factory:
- Resolves resource paths relative to `Path(__file__).resolve().parent` (i.e. inside the installed
  package, under `torchanipbe0/resources/`), not the working directory.
- Loads short-range weights via `neurochem.parse_neurochem_resources(info_file)` and
  `neurochem.load_model*` (see below).
- For dispersion, loads per-element MLP weights + linear shift constants from
  `resources/dispersion/{m1,m2,m3,v}/best.pt` + `best.param` (4-element models: H, C, N, O) or
  `resources/dispersion_2x/{m1,m2,m3,v}/` (7-element: H, C, N, O, S, F, Cl) via
  `CoefficientLayer._from_file_2` / `_from_file_3`.
- `*_CC` variants (e.g. `XDM_CC`, `ANIPBE0_XDM_CC`) use fixed/constant per-element coefficients
  instead of the trained MLP (`CoefficientLayer._from_constants_2`), useful as an ablation baseline.
- `c6_energy`/`c8_energy`/`c10_energy` and `m1_coefficients`/`m2_coefficients`/`m3_coefficients`/
  `v_coefficients` (+ `_CC` and `_2x` variants) expose individual energy terms / raw coefficients
  for analysis rather than the full combined PES — useful when validating or debugging the
  dispersion NN in isolation. `pairwise_energy_extractor` / `pairwise_energy_CC_extractor` return
  per-pair (index, distance, C6, C8, C10) data instead of a scalar energy.
- `CoefficientExtractor`/`CoefficientExtractorCC` expose a `compute_from_ase(atoms)` convenience
  method that builds tensors directly from an ASE `Atoms` object (species/cell/pbc/positions) and
  returns a detached numpy array — this is the intended entry point for "just give me the
  coefficients for this structure" workflows, bypassing the ASE `Calculator` machinery.

**Why `*2`/`*3` model suffixes exist:** `BuiltinModel`/`BuiltinEnsemble` (original TorchANI) load
per-element networks from NeuroChem-format directories via `neurochem.load_model`/
`load_model_ensemble`. `BuiltinModel2`/`BuiltinEnsemble2` and `BuiltinModel3`/`BuiltinEnsemble3`
were added (see `models.py` "Modification from TorchANI start here") to instead load a single
`best.pt` state dict per ensemble member (`neurochem.load_model_2`/`load_model_3`), which is how
the PBE0-trained ANI networks (`ANIPBE0`, 4-element) and PBE0-2x networks (`ANIPBE0_2x`,
7-element) are distributed. When adding a new element set or architecture, this is the pattern to
follow: a new `BuiltinModelN`/`BuiltinEnsembleN` pair plus matching `neurochem.load_model_N`/
`load_model_ensemble_N` loader, mirroring `CoefficientLayer._create_model` vs `_create_model_2`
for the differing per-element MLP architectures (4 vs 7 elements).

### Resource layout (`torchanipbe0/resources/`)

- `ani-*_8x.info` files are pointers (const file / SAE file / ensemble prefix / ensemble size) read
  by `neurochem.parse_neurochem_resources`. If the referenced resource directory isn't present
  locally, TorchANI's original models (`ani-1x`, `ani-1ccx`, `ani-2x`) will attempt to download
  from the `aiqm/ani-model-zoo` GitHub repo into `~/.local/torchani/` as a fallback — the PBE0/MLXDM
  resources are not downloadable this way and must already exist under `resources/`.
- `resources/dispersion/` (4-element H,C,N,O) and `resources/dispersion_2x/` (7-element, adds
  S,F,Cl) each contain `m1/`, `m2/`, `m3/`, `v/` subfolders with a `best.pt` (MLP weights) and
  `best.param` (text file: number of element blocks, then `b0` shift list, `b1` scale list, then
  one line per element giving the MLP layer dimensions) — parsed by `CoefficientLayer._from_file_2`
  /`_3`.
- Model `.pt`/resource files are checked into git (see recent commit history — most commits are
  "updated ... pt files" for retrained weights). When updating trained weights, keep the
  `best.param` architecture description in sync with the `best.pt` state dict, and keep the 4- vs
  7-element (`_2`/`_3`) versions of `CoefficientLayer._create_model*` matching whichever resource
  directory you're pointing at.

### Units convention

Internally, distances passed into `AEVComputer`/dispersion layers are in Angstrom; energies coming
out of the short-range `EnergyShifter` and the dispersion layers are in Hartree (converted to eV by
the ASE `Calculator`, see `torchanipbe0/units.py`). `BOHR_TO_ANSTROM` in
`torchanipbe0/dispersion/utils.py` is used to convert the XDM dispersion formulas (which are
naturally in atomic units) into the Angstrom/Hartree convention used elsewhere.

### Periodic boundary conditions

Two code paths exist throughout the dispersion/AEV layers, selected by whether `cell`/`pbc` are
`None`:
- Non-periodic: `DistanceLayer` computes all pairwise distances via `torch.triu_indices` directly
  on Cartesian coordinates.
- Periodic: `DistanceNeighborList`/`neighbor_list` (`dispersion/utils.py`) computes minimum-image
  distances under lattice shifts (`compute_shifts`/`cell_info`), returning a variable-length
  `(distance, index)` pair instead of a fixed triangular matrix.

Both `DispersionLayer.forward` and the plain `AEVComputer` branch on this in the same way — keep
new dispersion/energy layers consistent with this branching if you add one.
