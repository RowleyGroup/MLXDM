#!/bin/bash
# MACE training job for the 7-element PBE0 dataset (H, C, N, O, F, S, Cl) on Narval.
#
# Adapted from a known-working "MACE_big" recipe (see README.md in this directory for the
# full diff/rationale). Two things were changed relative to the run that produced
# std_pbe0.out / macepbe0.log:
#
#   1. Flag names updated for the installed mace_run_train version (mace-cueq venv):
#        --train_files      -> --train_file (singular)
#        --test_fraction     -> --valid_fraction (there is no --test_fraction; a held-out
#                                test set needs a separate --test_file/--test_dir, not used here)
#        --num_bessel        -> --num_radial_basis
#        --num_polynomial_cutoff -> --num_cutoff_basis
#
#   2. --clip_grad and --loss huber added: macepbe0.log showed train_loss spiking from
#      ~400 to ~48,000,000 at epoch 2 and staying high at epoch 3, which is a sign of an
#      outlier config (bad SCF convergence) or an unbounded gradient dominating a batch,
#      not normal optimizer noise. Huber loss + gradient clipping make training robust to
#      a handful of bad labels without having to first hunt them down in narval.hdf5.
#
# SLURM header below is a template — fill in --account/--time/--gres for your allocation
# before submitting (sbatch train_pbe0.sh).

#SBATCH --job-name=mace_pbe0
#SBATCH --account=<def-your-pi>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs_pbe0/%x-%j.out

set -euo pipefail

module purge
module load StdEnv/2023 gcc/12.3 cuda # add/adjust modules to match your mace-cueq build
source /home/crowley/venvs/mace-cueq/bin/activate

mkdir -p logs_pbe0

mace_run_train \
    --name="MACE_pbe0" \
    --train_file="/project/6060902/crowley/hdf5/narval.hdf5" \
    --valid_fraction=0.1 \
    --E0s='{1: -16.35824498, 6: -1035.67432207, 7: -1488.08434808, 8: -2045.43529646, 9: -2715.31530935, 16: -10831.85429748, 17: -12518.80754811}' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=5.0 \
    --num_radial_basis=8 \
    --num_cutoff_basis=5 \
    --batch_size=10 \
    --max_num_epochs=1500 \
    --stage_two \
    --start_stage_two=1200 \
    --loss=huber \
    --huber_delta=0.01 \
    --clip_grad=10 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda
