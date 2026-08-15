# MACEPBE0 / MACEXDM benchmarks

Timing benchmarks for the MACE-based models, mirroring the existing
pentane-chain MD benchmark pattern used for the ANI/torchanipbe0 models.
Each model runs as its own Slurm job so the two can be queued and compared
independently.

* `mace_benchmark_common.py` — shared timing loop: grows a chain of pentane
  molecules and, every 10 molecules, times a 10000-step Langevin MD run at
  298.15 K, writing `[n_molecules, time_seconds]` rows to a CSV file after
  every measurement.
* `benchmark_macepbe0.py` — benchmarks MACEPBE0 alone (short-range only, via
  `mace.calculators.MACECalculator`). Writes `simulation_timings_macepbe0.csv`.
* `benchmark_macexdm.py` — benchmarks MACEXDM (MACEPBE0 short-range + XDM
  dispersion, via `mace.calculators.MACEXDMDispersionCalculator`). Writes
  `simulation_timings_macexdm.csv`.
* `submit_macepbe0.sh`, `submit_macexdm.sh` — Slurm submission scripts.

Both scripts expect a base geometry at `xyz/pentane.xyz`, relative to the
submission directory (same as the existing ANI pentane benchmark).

## Usage

```bash
sbatch submit_macepbe0.sh 100   # benchmark up to 100 pentane molecules
sbatch submit_macexdm.sh 100
```

The `max_n_pentane` argument is optional and defaults to 100.

## Notes

* Checkpoint paths (`MACEPBE0_CKPT`, `ATOMICXDMMACE_CKPT`) are hardcoded at
  the top of each benchmark script — update them if the checkpoints move.
* The Slurm template had a duplicated `--nodes=1` line and two conflicting
  `--output` directives; both submission scripts here keep a single, more
  specific `--output=slurm_%j_<job>.out` line instead.
