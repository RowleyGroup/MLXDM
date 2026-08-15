"""Generate a spherical "ball" of water molecules and write it as a PDB file.

This is only a convenience script for producing a sample input structure for
``reduce_water_ball.py``. It packs water molecules (rigid, TIP3P-like
geometry) onto a cubic lattice, keeps only the lattice sites that fall inside
a sphere of the requested radius, and gives each molecule a random
orientation.

Usage:
    python3 generate_water_ball.py --radius 10.0 --output water_ball.pdb
"""
import argparse
import math
import random

O_H_BOND = 0.9572          # Angstrom
H_O_H_ANGLE = 104.52       # degrees


def water_geometry():
    """Return (O, H1, H2) coordinates for a water molecule centered at the
    origin with the oxygen atom at (0, 0, 0), before any rotation."""
    half_angle = math.radians(H_O_H_ANGLE) / 2.0
    h1 = (O_H_BOND * math.sin(half_angle), O_H_BOND * math.cos(half_angle), 0.0)
    h2 = (-O_H_BOND * math.sin(half_angle), O_H_BOND * math.cos(half_angle), 0.0)
    return (0.0, 0.0, 0.0), h1, h2


def random_rotation_matrix(rng):
    """Uniformly random 3x3 rotation matrix (Arvo's method)."""
    x1, x2, x3 = rng.random(), rng.random(), rng.random()
    theta = x1 * 2.0 * math.pi
    phi = x2 * 2.0 * math.pi
    z = x3 * 2.0

    r = math.sqrt(z)
    vx = math.sin(phi) * r
    vy = math.cos(phi) * r
    vz = math.sqrt(2.0 - z)

    st, ct = math.sin(theta), math.cos(theta)
    sx, sy = vx * ct - vy * st, vx * st + vy * ct

    rotation = [
        [vx * sx - ct, vx * sy - st, vx * vz],
        [vy * sx + st, vy * sy - ct, vy * vz],
        [vz * sx, vz * sy, 1.0 - z],
    ]
    return rotation


def apply_rotation(matrix, point, offset):
    x, y, z = point
    ox, oy, oz = offset
    rx = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z
    ry = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z
    rz = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z
    return (rx + ox, ry + oy, rz + oz)


def lattice_sites_in_sphere(radius, spacing):
    n = int(math.ceil(radius / spacing))
    sites = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            for k in range(-n, n + 1):
                x, y, z = i * spacing, j * spacing, k * spacing
                if math.sqrt(x * x + y * y + z * z) <= radius:
                    sites.append((x, y, z))
    return sites


def write_pdb(path, molecules):
    with open(path, "w") as fh:
        serial = 1
        for res_seq, (o, h1, h2) in enumerate(molecules, start=1):
            for atom_name, elem, pos in (
                (" O  ", "O", o),
                (" H1 ", "H", h1),
                (" H2 ", "H", h2),
            ):
                fh.write(
                    "ATOM  {serial:5d} {name:<4s}{res:<3s} {chain}{resseq:4d}    "
                    "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{temp:6.2f}          {elem:>2s}\n".format(
                        serial=serial % 100000,
                        name=atom_name,
                        res="HOH",
                        chain="W",
                        resseq=res_seq % 10000,
                        x=pos[0],
                        y=pos[1],
                        z=pos[2],
                        occ=1.00,
                        temp=0.00,
                        elem=elem,
                    )
                )
                serial += 1
        fh.write("END\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=10.0,
                         help="Radius of the water ball in Angstrom (default: 10.0)")
    parser.add_argument("--spacing", type=float, default=3.1,
                         help="Lattice spacing between water oxygens in Angstrom (default: 3.1)")
    parser.add_argument("--output", default="water_ball.pdb",
                         help="Output PDB file path (default: water_ball.pdb)")
    parser.add_argument("--seed", type=int, default=0,
                         help="Random seed for molecule orientations (default: 0)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    o_template, h1_template, h2_template = water_geometry()

    molecules = []
    for site in lattice_sites_in_sphere(args.radius, args.spacing):
        rot = random_rotation_matrix(rng)
        o = apply_rotation(rot, o_template, site)
        h1 = apply_rotation(rot, h1_template, site)
        h2 = apply_rotation(rot, h2_template, site)
        molecules.append((o, h1, h2))

    write_pdb(args.output, molecules)
    print("Wrote {} water molecules ({} atoms) to {}".format(
        len(molecules), len(molecules) * 3, args.output))


if __name__ == "__main__":
    main()
