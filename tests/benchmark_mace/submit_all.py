#!/usr/bin/env python3
"""Submit one Slurm job per water-ball xyz file, at increments of --step
water molecules, by calling `sbatch` on the matching per-job template
(submit_macepbe0.sh or submit_macexdm.sh).

Usage:
    python submit_all.py macepbe0
    python submit_all.py macexdm --step 20 --max 200
    python submit_all.py macepbe0 --xyz-dir /path/to/xyz
    python submit_all.py both --dry-run
    python submit_all.py macexdm -- --xdm-checkpoint /path/to/xdm.model
"""
import argparse
import glob
import os
import re
import subprocess
import sys

DEFAULT_XYZ_DIR = "/lustre06/project/6060902/crowley/timing/xyz"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATES = {
    "macepbe0": os.path.join(SCRIPT_DIR, "submit_macepbe0.sh"),
    "macexdm": os.path.join(SCRIPT_DIR, "submit_macexdm.sh"),
}


def n_waters_from_file(path):
    """Water count from the filename's first run of digits, falling back to
    n_atoms // 3 read from the xyz header."""
    match = re.search(r"(\d+)", os.path.basename(path))
    if match:
        return int(match.group(1))
    with open(path) as fh:
        n_atoms = int(fh.readline())
    return n_atoms // 3


def find_files_by_size(xyz_dir, pattern):
    file_for_n = {}
    for path in glob.glob(os.path.join(xyz_dir, pattern)):
        file_for_n[n_waters_from_file(path)] = path
    return file_for_n


def submit_model(model, xyz_dir, pattern, step, max_n, extra_args, dry_run):
    template = TEMPLATES[model]
    file_for_n = find_files_by_size(xyz_dir, pattern)
    if not file_for_n:
        sys.exit(f"No files matching '{pattern}' found in '{xyz_dir}'.")

    if max_n is None:
        max_n = max(file_for_n)

    print(f"Submitting {model} jobs at increments of {step} waters (up to {max_n})...")
    for n in range(step, max_n + 1, step):
        path = file_for_n.get(n)
        if path is None:
            print(f"  {n} waters -> no matching xyz file, skipping")
            continue

        cmd = ["sbatch", template, path, *extra_args]
        print(f"  {n} waters -> {path}")
        if dry_run:
            print("    " + " ".join(cmd))
        else:
            subprocess.run(cmd, check=True)


def main():
    # Split off anything after a literal "--": those flags are forwarded
    # verbatim to benchmark_mace.py rather than parsed here.
    argv = sys.argv[1:]
    if "--" in argv:
        sep = argv.index("--")
        argv, extra_args = argv[:sep], argv[sep + 1:]
    else:
        extra_args = []

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", choices=["macepbe0", "macexdm", "both"],
                         help="Which model's jobs to submit")
    parser.add_argument("--xyz-dir", default=DEFAULT_XYZ_DIR,
                         help="Directory of water-ball xyz files (default: %(default)s)")
    parser.add_argument("--pattern", default="water_ball_*.xyz",
                         help="Glob pattern for xyz files (default: %(default)s)")
    parser.add_argument("--step", type=int, default=10,
                         help="Submit a job every N water molecules (default: 10)")
    parser.add_argument("--max", type=int, default=None,
                         help="Largest water count to submit (default: largest available)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the sbatch commands without running them")
    args = parser.parse_args(argv)

    models = ["macepbe0", "macexdm"] if args.model == "both" else [args.model]
    for model in models:
        submit_model(model, args.xyz_dir, args.pattern, args.step, args.max,
                     extra_args, args.dry_run)


if __name__ == "__main__":
    main()
