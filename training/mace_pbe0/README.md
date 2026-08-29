# PBE0 MACE training (Narval)

`train_pbe0.sh` trains a standard `MACE` model on the 7-element PBE0 dataset
(H, C, N, O, F, S, Cl) at `/project/6060902/crowley/hdf5/narval.hdf5`, using the
`mace-cueq` venv on Narval.

## Why this script exists

Two prior attempts, logged in `std_pbe0.out` and `macepbe0.log` from this same job:

1. **`mace_run_train: error: unrecognized arguments`** — the original command used flag
   names from an older MACE CLI version. The installed `mace_run_train`
   (`/home/crowley/venvs/mace-cueq/bin/mace_run_train`) renamed them:

   | old flag | current flag |
   |---|---|
   | `--train_files` | `--train_file` (singular) |
   | `--test_fraction` | `--valid_fraction` (no `--test_fraction`; a real held-out test set needs `--test_file`/`--test_dir`) |
   | `--num_bessel` | `--num_radial_basis` |
   | `--num_polynomial_cutoff` | `--num_cutoff_basis` |

2. **Loss instability** — once the flags were fixed and training ran (`macepbe0.log`),
   `train_loss` spiked from ~400 (epoch 1) to ~48,000,000 (epoch 2) and stayed elevated
   at epoch 3, with `valid_loss`/`rmse_e_per_atom` following the same pattern. That
   magnitude of spike is consistent with an outlier structure (e.g. an unconverged PBE0
   SCF point) dominating a batch's gradient, not ordinary optimizer noise.

## What changed here vs. the raw fixed command

The recipe (`model=MACE`, `hidden_irreps='128x0e + 128x1o'`, `r_max=5.0`,
`batch_size=10`, two-stage training via `--stage_two --start_stage_two=1200`, `--ema
--ema_decay=0.99`, `--amsgrad`, `--restart_latest`) is copied from a MACE run on a
different dataset (`MACE_big`) that trained cleanly to 1500 epochs — reusing settings
that are already known to work rather than guessing new ones.

On top of that, two additions specifically target the loss spike:

- `--loss=huber --huber_delta=0.01`: Huber loss is far less sensitive than the default
  MSE-style loss to a small number of badly-labeled configs, since the per-example
  contribution is bounded linearly instead of quadratically past `huber_delta`.
- `--clip_grad=10`: bounds the gradient norm per step so one bad batch can't push the
  optimizer state somewhere it doesn't recover from (matches the epoch-2 → epoch-3
  behavior seen in `macepbe0.log`, where the loss didn't come back down on its own).

`E0s` are the values `mace_run_train` itself computed from `narval.hdf5` in
`macepbe0.log`, passed explicitly here so a restart or resubmission doesn't have to
recompute them (and so runs are reproducible against exactly these reference energies).

## Before submitting

- Fill in `--account`, `--time`, and any `module load` lines to match your actual Narval
  allocation and the modules your `mace-cueq` venv was built against — those weren't
  fully visible from the logs and are left as a template.
- If training is still unstable with `--loss=huber` and `--clip_grad=10`, the next step
  is to inspect `narval.hdf5` directly for outlier forces/energies (e.g. max |force| per
  config, several sigma from the mean) rather than continuing to tune loss/clipping
  around a bad label.
- Consider `--enable_cueq=True` once the run is otherwise stable, to use the
  cuequivariance acceleration this venv is built for.
