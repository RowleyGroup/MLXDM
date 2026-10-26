"""Count structures (conformers) in an HDF5 training set.

Usage:
    python scripts/count_structures.py data.h5 [--key coordinates]

Every dataset named --key (default: coordinates) is treated as a stack of
structures whose first axis is the conformer axis. Counts are reported per
top-level group (e.g. ani2x / mlxdm_2x / mlxdm_2x_extra / mlxdm_1x buckets),
along with the number of groups holding conformers (molecules, or molecule
pairs, in the TorchANI layout) and the grand total.
"""
import argparse
from collections import defaultdict

import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('h5file')
    ap.add_argument('--key', default='coordinates',
                    help='name of the per-group coordinate dataset')
    args = ap.parse_args()

    structures = defaultdict(int)
    groups = defaultdict(int)

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and name.split('/')[-1] == args.key:
            parts = name.split('/')
            top = parts[0] if len(parts) > 1 else '(root)'
            # A single (n_atoms, 3) array is one structure, not n_atoms of them.
            n = obj.shape[0] if obj.ndim == 3 else 1
            structures[top] += n
            groups[top] += 1

    with h5py.File(args.h5file, 'r') as f:
        f.visititems(visit)

    if not structures:
        print(f'No datasets named "{args.key}" found; try --key with another name.')
        return

    print(f'{"top-level group":30s} {"groups":>10s} {"structures":>12s}')
    for top in sorted(structures):
        print(f'{top:30s} {groups[top]:10d} {structures[top]:12d}')
    print(f'{"TOTAL":30s} {sum(groups.values()):10d} {sum(structures.values()):12d}')


if __name__ == '__main__':
    main()
