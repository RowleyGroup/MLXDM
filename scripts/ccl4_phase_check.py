"""Liquid-vs-solid diagnostics for periodic CCl4 NPT trajectories.

Usage:
    python scripts/ccl4_phase_check.py traj1 [traj2 ...] [--stride N] [--dt-fs DT]

Any ASE-readable trajectory with a cell works (.traj, .extxyz, lammps dump, ...).
--dt-fs is the time between *stored* frames in fs (before --stride); if the
file carries no time information this is needed to report diffusion in cm^2/s.

For each trajectory it reports, over the last half of the frames:
  * density (g/cm^3)
  * MSD of molecular centres (C atoms), unwrapped frame-to-frame, and a
    diffusion coefficient from its slope. Liquid CCl4 at 298 K should give
    D on the order of 1e-5 cm^2/s (tens of A^2 per 100 ps); a crystal
    (ordered or plastic) gives an MSD that plateaus below ~1 A^2.
  * Lechner-Dellago averaged q6 of the C-atom centres (12 nearest centres).
    Disordered liquids give mean q6bar ~0.15-0.25; fcc/hcp-like lattices
    give ~0.4-0.55. Reported as the mean and the fraction of centres with
    q6bar > 0.35 ("crystal-like").
  * C-Cl bond-vector rotational autocorrelation <P1(u(0).u(t))> at the
    largest lag, which separates an orientationally frozen crystal from a
    plastic (rotator) phase in which centres are ordered but molecules spin.
  * C-C radial distribution function, written to <name>_gCC.dat, for a
    visual check of split second peaks / long-range oscillations.
"""
import argparse
import os
import sys

import numpy as np
from ase.io import read
from scipy.special import sph_harm_y

AMU_TO_G = 1.66053906660e-24


def mic(d, cell):
    """Minimum-image displacement vectors for a (possibly triclinic) cell."""
    frac = np.linalg.solve(cell.T, d.T).T
    frac -= np.round(frac)
    return frac @ cell


def molecules(atoms):
    """Return (C indices, [4 Cl indices per C]) by nearest-Cl assignment."""
    sym = np.array(atoms.get_chemical_symbols())
    c_idx = np.where(sym == 'C')[0]
    cl_idx = np.where(sym == 'Cl')[0]
    cell = atoms.cell.array
    pos = atoms.positions
    owner = []
    for j in cl_idx:
        d = np.linalg.norm(mic(pos[c_idx] - pos[j], cell), axis=1)
        owner.append(c_idx[np.argmin(d)])
    owner = np.array(owner)
    groups = [cl_idx[owner == c] for c in c_idx]
    bad = sum(len(g) != 4 for g in groups)
    if bad:
        print(f'  warning: {bad} C atoms do not have exactly 4 Cl neighbours '
              '(possible bond breaking)')
    return c_idx, groups


def q6bar(centres, cell, nn=12):
    n = len(centres)
    d = centres[None, :, :] - centres[:, None, :]
    d = mic(d.reshape(-1, 3), cell).reshape(n, n, 3)
    r = np.linalg.norm(d, axis=2)
    np.fill_diagonal(r, np.inf)
    nbr = np.argsort(r, axis=1)[:, :nn]
    qlm = np.zeros((n, 13), dtype=complex)
    for i in range(n):
        v = d[i, nbr[i]]
        rr = np.linalg.norm(v, axis=1)
        theta = np.arccos(np.clip(v[:, 2] / rr, -1, 1))
        phi = np.arctan2(v[:, 1], v[:, 0])
        for k, m in enumerate(range(-6, 7)):
            qlm[i, k] = sph_harm_y(6, m, theta, phi).mean()
    qbar = np.array([(qlm[i] + qlm[nbr[i]].sum(0)) / (nn + 1) for i in range(n)])
    return np.sqrt(4 * np.pi / 13 * (np.abs(qbar) ** 2).sum(1))


def gr(centres, cell, rmax, nbins=200):
    n = len(centres)
    d = centres[None, :, :] - centres[:, None, :]
    r = np.linalg.norm(mic(d.reshape(-1, 3), cell), axis=1)
    r = r[r > 1e-6]
    h, edges = np.histogram(r, bins=nbins, range=(0, rmax))
    vol = abs(np.linalg.det(cell))
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    return 0.5 * (edges[1:] + edges[:-1]), h / (n * (n / vol) * shell)


def analyse(path, stride, dt_fs):
    print(f'== {path}')
    frames = read(path, index=f'::{stride}')
    if not isinstance(frames, list):
        frames = [frames]
    nf = len(frames)
    if nf < 4:
        print('  too few frames')
        return
    half = frames[nf // 2:]
    c_idx, groups = molecules(half[0])
    nmol = len(c_idx)

    rho = [a.get_masses().sum() * AMU_TO_G / (a.get_volume() * 1e-24) for a in half]
    print(f'  frames used: {len(half)} of {nf} (stride {stride}), molecules: {nmol}')
    print(f'  density: {np.mean(rho):.3f} +/- {np.std(rho):.3f} g/cm^3')

    # Unwrapped centre trajectories and C->Cl unit vectors.
    centres = [half[0].positions[c_idx].copy()]
    uvecs = []
    for k, a in enumerate(half):
        cell = a.cell.array
        if k:
            step = mic(a.positions[c_idx] - half[k - 1].positions[c_idx], cell)
            centres.append(centres[-1] + step)
        u = mic(a.positions[[g[0] for g in groups]] - a.positions[c_idx], cell)
        uvecs.append(u / np.linalg.norm(u, axis=1)[:, None])
    centres = np.array(centres)
    uvecs = np.array(uvecs)

    # Subtract centre-of-mass drift of the whole box.
    centres -= centres.mean(1, keepdims=True) - centres[0].mean(0)
    lags = np.unique(np.linspace(1, len(half) - 1, 20).astype(int))
    msd = np.array([((centres[l:] - centres[:-l]) ** 2).sum(2).mean() for l in lags])
    p1 = np.array([(uvecs[l:] * uvecs[:-l]).sum(2).mean() for l in lags])

    print('  lag(frames)  MSD(A^2)   <P1>')
    for l, m, p in zip(lags[::4], msd[::4], p1[::4]):
        print(f'  {l:10d} {m:9.3f} {p:7.3f}')
    if dt_fs:
        t_ps = lags * stride * dt_fs / 1000
        sl = slice(len(lags) // 3, None)
        slope = np.polyfit(t_ps[sl], msd[sl], 1)[0]  # A^2/ps
        D = slope / 6 * 1e-4  # cm^2/s
        print(f'  window {t_ps[-1]:.1f} ps;  D = {D:.2e} cm^2/s')

    q = np.concatenate([q6bar(a.positions[c_idx], a.cell.array)
                        for a in half[:: max(1, len(half) // 10)]])
    print(f'  q6bar: mean {q.mean():.3f}, fraction > 0.35: {(q > 0.35).mean():.2f}')

    rmax = 0.5 * min(np.linalg.norm(half[-1].cell.array, axis=1))
    acc = None
    sample = half[:: max(1, len(half) // 50)]
    for a in sample:
        r, g = gr(a.positions[c_idx], a.cell.array, rmax)
        acc = g if acc is None else acc + g
    out = os.path.splitext(os.path.basename(path))[0] + '_gCC.dat'
    np.savetxt(out, np.c_[r, acc / len(sample)], header='r(A) g_CC(r)')
    print(f'  C-C g(r) written to {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trajs', nargs='+')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--dt-fs', type=float, default=None,
                    help='time between stored frames in fs')
    args = ap.parse_args()
    for p in args.trajs:
        analyse(p, args.stride, args.dt_fs)


if __name__ == '__main__':
    sys.exit(main())
