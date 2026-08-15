# MACEPBE0 / MACEXDM benchmarks

Timing benchmarks for the MACE-based models, mirroring the existing
pentane-chain MD benchmark pattern used for the ANI/torchanipbe0 models.

* `benchmark_mace.py` — single self-contained script (only needs `ase`,
  `torch`, and `mace` installed). Grows a chain of pentane molecules and,
  every 10 molecules, times a 10000-step Langevin MD run at 298.15 K,
  writing `[n_molecules, time_seconds]` rows to a CSV file after every
  measurement. Takes the model (`macepbe0` or `macexdm`) as its first
  argument, so it drops onto any cluster as one file.
* `submit_macepbe0.sh`, `submit_macexdm.sh` — separate Slurm jobs, each
  calling `benchmark_mace.py` with a different model argument so the two
  models can be queued and compared independently.

Both scripts expect a base geometry at `xyz/pentane.xyz`, relative to the
submission directory (same as the existing ANI pentane benchmark) — pass
`--xyz` to point elsewhere.

## Usage

```bash
python benchmark_mace.py macepbe0 100
python benchmark_mace.py macexdm 100

# or via Slurm
sbatch submit_macepbe0.sh 100   # benchmark up to 100 pentane molecules
sbatch submit_macexdm.sh 100
```

The `max_n_pentane` argument is optional when submitting via Slurm and
defaults to 100.

## Deploying to a different cluster

The checkpoint paths, module/venv activation, and Slurm account are all
specific to the original cluster:

* `benchmark_mace.py` defaults `--pbe0-checkpoint` and `--xdm-checkpoint` to
  the original cluster's paths — override them on the command line, e.g.:
  ```bash
  python benchmark_mace.py macexdm 100 \
      --pbe0-checkpoint /path/to/mace-pbe0.model \
      --xdm-checkpoint /path/to/xdm_element.model
  ```
* Edit the `#SBATCH --account=...`, `module purge` / `source .../activate`,
  and `--gres=gpu:1` lines in the two `submit_*.sh` scripts to match the new
  cluster.

## Notes

* The Slurm template had a duplicated `--nodes=1` line and two conflicting
  `--output` directives; both submission scripts here keep a single, more
  specific `--output=slurm_%j_<job>.out` line instead.
