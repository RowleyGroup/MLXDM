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

# Times a single water-ball xyz file. Usually submitted by submit_all_macexdm.sh
# (one sbatch call per water-ball size), but can be called directly too:
# Usage: sbatch submit_macexdm.sh <xyz_file> [extra benchmark_mace.py flags]
XYZ_FILE="${1:?Usage: sbatch submit_macexdm.sh <xyz_file> [extra flags]}"
shift

module purge
source /lustre06/project/6060902/crowley/macexdm/bin/activate

python benchmark_mace.py macexdm --xyz-file "${XYZ_FILE}" "$@"
