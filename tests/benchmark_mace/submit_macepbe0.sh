#!/bin/bash
#SBATCH --job-name=macepbe0_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16000MB
#SBATCH --account=def-crowley-ab
#SBATCH --output=slurm_%j_macepbe0_benchmark.out

# Usage: sbatch submit_macepbe0.sh [extra benchmark_mace.py flags]
# e.g.:  sbatch submit_macepbe0.sh --limit 20
# e.g.:  sbatch submit_macepbe0.sh --xyz-dir /path/to/xyz

module purge
source /lustre06/project/6060902/crowley/macexdm/bin/activate

python benchmark_mace.py macepbe0 "$@"
