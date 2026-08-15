# MACEPBE0 / MACEXDM benchmarks

Timing benchmarks for the MACE-based models, run over a series of
progressively larger water-ball structures. Each water-ball size is
submitted as its own Slurm job.

* `benchmark_mace.py` — single self-contained script (only needs `ase`,
  `torch`, and `mace` installed). Given `--xyz-file`, times one water-ball
  xyz file; given `--xyz-dir` (default:
  `/lustre06/project/6060902/crowley/timing/xyz`), scans it and times every
  matching file, smallest first. Either way, for each structure it runs a
  10000-step Langevin MD at 298.15 K and appends
  `[xyz_file, n_atoms, n_waters, time_seconds]` to a CSV file, flushed after
  every measurement. Takes the model (`macepbe0` or `macexdm`) as its first
  argument.
* `submit_macepbe0.sh`, `submit_macexdm.sh` — per-job Slurm templates. Each
  takes one xyz file as its argument and times just that structure.
* `submit_all_macepbe0.sh`, `submit_all_macexdm.sh` — driver scripts (plain
  bash, run directly on the login node, not sbatch scripts themselves).
  They scan the xyz directory, and for every multiple of 10 water molecules
  (configurable via `--step`) that has a matching file, call
  `sbatch submit_mace*.sh <file>` — one Slurm job per size.

## Usage

```bash
./submit_all_macepbe0.sh
./submit_all_macexdm.sh

# every 20 waters instead of every 10, capped at 200 waters:
./submit_all_macepbe0.sh --step 20 --max 200

# a different xyz directory, or extra flags forwarded to benchmark_mace.py:
./submit_all_macexdm.sh --xyz-dir /path/to/xyz
./submit_all_macexdm.sh -- --xdm-checkpoint /path/to/xdm.model

# submit (or rerun) a single size directly:
sbatch submit_macepbe0.sh /path/to/xyz/water_ball_0100.xyz
```

Each Slurm job writes its own CSV (named after the xyz file, e.g.
`simulation_timings_macepbe0_water_ball_0100.csv`) so parallel jobs never
clobber each other's results — combine them afterwards if you want one
table.

## Deploying to a different cluster

The checkpoint paths, xyz directory, module/venv activation, and Slurm
account are all specific to the original cluster:

* `benchmark_mace.py` defaults `--pbe0-checkpoint`, `--xdm-checkpoint`, and
  `--xyz-dir` to the original cluster's paths — override them (the
  `submit_all_*.sh` drivers forward anything after `--` straight through to
  `benchmark_mace.py`, and `--xyz-dir` has its own driver flag).
* Edit the `#SBATCH --account=...`, `module purge` / `source .../activate`,
  and `--gres=gpu:1` lines in `submit_macepbe0.sh` / `submit_macexdm.sh` to
  match the new cluster.

## Notes

* Water-ball sizes are parsed from the xyz filename (first run of digits),
  falling back to `n_atoms // 3` if a filename has no digits.
* The Slurm template had a duplicated `--nodes=1` line and two conflicting
  `--output` directives; the per-job scripts keep a single, more specific
  `--output=slurm_%j_<job>.out` line instead.
