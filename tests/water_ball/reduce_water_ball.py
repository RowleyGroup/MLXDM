"""Shrink a spherical "ball" of water molecules read from a PDB file.

Water molecules are peeled off the outside of the ball, 10 at a time
(farthest-from-center first), and each intermediate structure is written out
as its own XYZ file. This produces a sequence of progressively smaller water
balls, e.g.:

    water_ball_0500.xyz   (full ball, 500 waters)
    water_ball_0490.xyz   (10 outermost waters removed)
    water_ball_0480.xyz
    ...
    water_ball_0010.xyz   (last full batch of 10)

Only the standard library is used, so no extra Python packages need to be
installed to run it.

Usage:
    python3 reduce_water_ball.py --input water_ball.pdb --outdir output
"""
import argparse
import math
import os

ATOMS_PER_WATER = 3


def guess_element(atom_name):
    """Infer an element symbol from a PDB atom name when columns 77-78 are
    missing or blank (e.g. 'OW' -> 'O', 'H1' -> 'H')."""
    name = atom_name.strip()
    for ch in name:
        if ch.isalpha():
            return ch.upper()
    return name[:1].upper() if name else "X"


def parse_pdb(path):
    """Parse ATOM/HETATM records and group them into molecules by
    (chain ID, residue sequence number, insertion code).

    Returns a list of molecules, each a list of (element, x, y, z) tuples,
    in the order first encountered in the file.
    """
    molecules = {}
    order = []
    with open(path) as fh:
        for line in fh:
            record = line[:6]
            if record not in ("ATOM  ", "HETATM"):
                continue
            # Standard PDB fixed-width columns (1-indexed in the spec).
            atom_name = line[12:16]
            res_name = line[17:20].strip()
            chain_id = line[21:22]
            res_seq = line[22:26].strip()
            icode = line[26:27]
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                # Fall back to whitespace splitting for non-standard files.
                fields = line.split()
                x, y, z = float(fields[-6]), float(fields[-5]), float(fields[-4])

            element = line[76:78].strip() if len(line) >= 78 else ""
            if not element:
                element = guess_element(atom_name)

            key = (chain_id, res_seq, icode)
            if key not in molecules:
                molecules[key] = {"res_name": res_name, "atoms": []}
                order.append(key)
            molecules[key]["atoms"].append((element, x, y, z))

    return [molecules[key]["atoms"] for key in order]


def molecule_center(atoms):
    n = len(atoms)
    sx = sum(a[1] for a in atoms)
    sy = sum(a[2] for a in atoms)
    sz = sum(a[3] for a in atoms)
    return (sx / n, sy / n, sz / n)


def ball_center(molecules):
    centers = [molecule_center(m) for m in molecules]
    n = len(centers)
    sx = sum(c[0] for c in centers)
    sy = sum(c[1] for c in centers)
    sz = sum(c[2] for c in centers)
    return (sx / n, sy / n, sz / n)


def distance(p, q):
    return math.sqrt(sum((pi - qi) ** 2 for pi, qi in zip(p, q)))


def write_xyz(path, molecules, comment):
    atoms = [atom for molecule in molecules for atom in molecule]
    with open(path, "w") as fh:
        fh.write("{}\n".format(len(atoms)))
        fh.write("{}\n".format(comment))
        for element, x, y, z in atoms:
            fh.write("{:<2s} {:14.6f} {:14.6f} {:14.6f}\n".format(element, x, y, z))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True,
                         help="Input PDB file containing the water ball")
    parser.add_argument("--outdir", "-o", default="output",
                         help="Directory to write the XYZ files into (default: output)")
    parser.add_argument("--step", "-n", type=int, default=10,
                         help="Number of water molecules removed per step (default: 10)")
    parser.add_argument("--prefix", default="water_ball",
                         help="Filename prefix for the XYZ files (default: water_ball)")
    args = parser.parse_args()

    molecules = parse_pdb(args.input)
    for atoms in molecules:
        if len(atoms) != ATOMS_PER_WATER:
            raise ValueError(
                "Expected every residue to have {} atoms (one water), got a "
                "residue with {} atoms. Check that the PDB groups each water "
                "molecule under its own residue sequence number.".format(
                    ATOMS_PER_WATER, len(atoms)))

    if not molecules:
        raise ValueError("No water molecules found in {}".format(args.input))

    os.makedirs(args.outdir, exist_ok=True)

    center = ball_center(molecules)
    # Order molecules from farthest to nearest so removal peels the ball
    # from the outside in.
    remaining = sorted(
        molecules, key=lambda m: distance(molecule_center(m), center), reverse=True
    )

    total = len(remaining)
    pad = len(str(total))

    def save(mols, tag):
        fname = "{}_{}.xyz".format(args.prefix, str(len(mols)).zfill(pad))
        path = os.path.join(args.outdir, fname)
        write_xyz(path, mols, "{} water molecules ({})".format(len(mols), tag))
        print("Wrote {} ({} waters)".format(path, len(mols)))

    save(remaining, "full ball")

    while len(remaining) > 0:
        n_remove = min(args.step, len(remaining))
        remaining = remaining[n_remove:]  # drop the farthest-out molecules (list is sorted far -> near)
        if not remaining:
            break
        save(remaining, "after removing {} outermost waters".format(n_remove))

    print("Done. {} total water molecules processed from {}.".format(total, args.input))


if __name__ == "__main__":
    main()
