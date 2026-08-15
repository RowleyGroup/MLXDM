# MACEPBE0 / MACEXDM benchmarks

Timing benchmarks for the MACE-based models, run over a series of
progressively larger water-ball structures.

* `benchmark_mace.py` — single self-contained script (only needs `ase`,
  `torch`, and `mace` installed). Reads `water_ball_*.xyz` files from a
  directory (default: `/lustre06/project/6060902/crowley/timing/xyz`),
  smallest first, and for each one times a 10000-step Langevin MD run at
  298.15 K, writing `[xyz_file, n_atoms, n_waters, time_seconds]` rows to a
  CSV file after every measurement. Takes the model (`macepbe0` or
  `macexdm`) as its first argument, so it drops onto any cluster as one
  file.
* `submit_macepbe0.sh`, `submit_macexdm.sh` — separate Slurm jobs, each
  calling `benchmark_mace.py` with a different model argument so the two
  models can be queued and compared independently.

## Usage

```bash
python benchmark_mace.py macepbe0
python benchmark_mace.py macexdm

# or via Slurm
sbatch submit_macepbe0.sh
sbatch submit_macexdm.sh

# pass extra flags through to benchmark_mace.py, e.g. to cap how many
# structures run, or to point at a different xyz directory:
sbatch submit_macepbe0.sh --limit 20
sbatch submit_macexdm.sh --xyz-dir /path/to/xyz
```

## Deploying to a different cluster

The checkpoint paths, xyz directory, module/venv activation, and Slurm
account are all specific to the original cluster:

* `benchmark_mace.py` defaults `--pbe0-checkpoint`, `--xdm-checkpoint`, and
  `--xyz-dir` to the original cluster's paths — override them on the
  command line, e.g.:
  ```bash
  python benchmark_mace.py macexdm \
      --pbe0-checkpoint /path/to/mace-pbe0.model \
      --xdm-checkpoint /path/to/xdm_element.model \
      --xyz-dir /path/to/water_ball/xyz
  ```
* Edit the `#SBATCH --account=...`, `module purge` / `source .../activate`,
  and `--gres=gpu:1` lines in the two `submit_*.sh` scripts to match the new
  cluster.

## Notes

* Structures are sorted by atom count (read from each xyz file's first
  line), not by filename, so any naming scheme works.
* The number of waters reported in the CSV is parsed from the filename
  (first run of digits) when possible, falling back to `n_atoms // 3`.
* The Slurm template had a duplicated `--nodes=1` line and two conflicting
  `--output` directives; both submission scripts here keep a single, more
  specific `--output=slurm_%j_<job>.out` line instead.
