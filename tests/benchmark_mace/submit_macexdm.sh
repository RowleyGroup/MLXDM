#!/bin/bash
#SBATCH --job-name=macexdm_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16000MB
#SBATCH --account=def-crowley-ab
#SBATCH --output=slurm_%j_macexdm_benchmark.out

# Usage: sbatch submit_macexdm.sh [max_n_pentane]
MAX_N_PENTANE="${1:-100}"

module purge
source /lustre06/project/6060902/crowley/macexdm/bin/activate

python benchmark_macexdm.py "${MAX_N_PENTANE}"
