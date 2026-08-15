#!/bin/bash
# Submits one Slurm job per water-ball xyz file, at increments of --step
# water molecules (default 10), by calling `sbatch submit_macexdm.sh` once
# per matching file. Run this directly on the login node (it is not itself
# an sbatch script).
#
# Usage: ./submit_all_macexdm.sh [--xyz-dir DIR] [--pattern PATTERN] [--step N] [--max N] [-- extra benchmark_mace.py flags]
# e.g.:  ./submit_all_macexdm.sh
# e.g.:  ./submit_all_macexdm.sh --step 20 --max 200
# e.g.:  ./submit_all_macexdm.sh -- --xdm-checkpoint /path/to/xdm.model
set -euo pipefail

XYZ_DIR="/lustre06/project/6060902/crowley/timing/xyz"
PATTERN="water_ball_*.xyz"
STEP=10
MAX=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --xyz-dir) XYZ_DIR="$2"; shift 2 ;;
        --pattern) PATTERN="$2"; shift 2 ;;
        --step) STEP="$2"; shift 2 ;;
        --max) MAX="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

declare -A FILE_FOR_N
for f in "${XYZ_DIR}"/${PATTERN}; do
    [[ -e "$f" ]] || continue
    n_waters=$(basename "$f" | grep -oE '[0-9]+' | head -1)
    if [[ -z "$n_waters" ]]; then
        n_atoms=$(head -n 1 "$f" | tr -d '[:space:]')
        n_waters=$(( n_atoms / 3 ))
    fi
    n_waters=$((10#$n_waters))  # strip any leading zeros before arithmetic
    FILE_FOR_N[$n_waters]="$f"
done

if [[ ${#FILE_FOR_N[@]} -eq 0 ]]; then
    echo "No files matching '${PATTERN}' found in '${XYZ_DIR}'." >&2
    exit 1
fi

if [[ -z "$MAX" ]]; then
    MAX=0
    for n in "${!FILE_FOR_N[@]}"; do
        (( n > MAX )) && MAX=$n
    done
fi

echo "Submitting MACEXDM jobs at increments of ${STEP} waters (up to ${MAX})..."
for (( n = STEP; n <= MAX; n += STEP )); do
    if [[ -n "${FILE_FOR_N[$n]:-}" ]]; then
        echo "  ${n} waters -> ${FILE_FOR_N[$n]}"
        sbatch "${SCRIPT_DIR}/submit_macexdm.sh" "${FILE_FOR_N[$n]}" "${EXTRA_ARGS[@]}"
    else
        echo "  ${n} waters -> no matching xyz file, skipping"
    fi
done
