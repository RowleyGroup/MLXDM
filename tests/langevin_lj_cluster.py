# GPU LANGEVIN THERMOSTAT ON A LARGE LENNARD-JONES CLUSTER
#
# Stress-tests torchanipbe0.md.Langevin on a system much bigger than a single
# molecule: a several-hundred-atom Lennard-Jones (argon-like) cluster, using
# a minimal LJ "potential" that implements the same
# ``model((species, coordinates), cell=cell, pbc=pbc) -> SpeciesEnergies``
# interface as the real torchanipbe0 models, so it drops straight into
# optimize_geometry/Langevin with no other changes. Everything (coordinates,
# velocities, forces, the O(N^2) pairwise distance matrix) stays on the GPU
# for the whole run.
#
# With a large N, the relative fluctuation of the kinetic temperature about
# the target shrinks like sqrt(2 / n_dof), so this is also a much sharper
# check of the thermostat than the 5-atom methane demo.

import math
from types import SimpleNamespace

import torch

from torchanipbe0 import units
from torchanipbe0.md import optimize_geometry, Langevin, VelocityVerlet

device = torch.device('cuda')
dtype = torch.float32

# Argon-like LJ parameters, in the package's native units (Hartree, Angstrom, AMU).
EPSILON_EV = 0.0104
SIGMA = 3.4
MASS = 39.948
EPSILON = EPSILON_EV / units.HARTREE_TO_EV


class LJModel:
    """Minimal non-periodic Lennard-Jones potential, same call signature as
    a torchanipbe0 model: ``model((species, coordinates)) -> SpeciesEnergies``
    with an ``.energies`` attribute, computed via full O(N^2) pairwise
    distances (no neighbor list/cutoff -- fine for the cluster sizes here).
    """

    def __init__(self, epsilon: float, sigma: float):
        self.epsilon = epsilon
        self.sigma = sigma

    def __call__(self, species_coordinates, cell=None, pbc=None):
        species, coordinates = species_coordinates
        n_atoms = coordinates.shape[-2]
        i, j = torch.triu_indices(n_atoms, n_atoms, offset=1, device=coordinates.device)
        diff = coordinates[:, i, :] - coordinates[:, j, :]
        r = diff.norm(dim=-1)
        sr6 = (self.sigma / r) ** 6
        sr12 = sr6 * sr6
        pair_energies = 4.0 * self.epsilon * (sr12 - sr6)
        energies = pair_energies.sum(dim=-1)
        return SimpleNamespace(energies=energies)


model = LJModel(EPSILON, SIGMA)

# Build a simple-cubic lattice near the LJ pair minimum (r_min = 2^(1/6) sigma)
# as a starting point; L-BFGS will relax it into a bound cluster.
n_side = 8
n_atoms = n_side ** 3  # 512 atoms
spacing = 2.0 ** (1.0 / 6.0) * SIGMA
grid = torch.arange(n_side, dtype=dtype, device=device) * spacing
gx, gy, gz = torch.meshgrid(grid, grid, grid, indexing='ij')
coordinates = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
coordinates -= coordinates.mean(dim=0, keepdim=True)
coordinates = coordinates.unsqueeze(0)  # (1, n_atoms, 3)

species = torch.zeros(1, n_atoms, dtype=torch.long, device=device)
masses = torch.full((n_atoms,), MASS, dtype=dtype, device=device)

print(f'Lennard-Jones cluster: {n_atoms} atoms, epsilon = {EPSILON:.3e} Ha '
      f'({EPSILON_EV} eV), sigma = {SIGMA} A, mass = {MASS} amu')

opt_coordinates, energy, forces, converged = optimize_geometry(
    model, species, coordinates, fmax=1e-5, max_steps=500)
print(f'relaxed (converged={converged}): E = {energy.item(): .6f} Ha '
      f'({energy.item() * units.HARTREE_TO_EV:.3f} eV), '
      f'fmax = {forces.abs().max().item():.2e} Ha/A')

# Sanity check the potential/force pipeline with a short NVE run before
# trusting the thermostat: total energy should be conserved to a small drift.
velocities0 = torch.zeros_like(opt_coordinates)
nve = VelocityVerlet(model, species, opt_coordinates, velocities0, masses, timestep=2.0)
e0 = nve.total_energy().item()
nve.run(200)
e1 = nve.total_energy().item()
print(f'NVE sanity check: energy drift over 200 steps of 2.0 fs: {e1 - e0:.3e} Ha '
      f'(E0 = {e0:.6f} Ha)\n')

# Now the actual test: Langevin thermostat driving the cluster to a target
# bath temperature, starting from rest.
temperature = 40.0  # K -- well below Ar's bulk melting point (~84 K)
friction = 0.02      # fs^-1
timestep = 2.0        # fs
n_dof = 3 * n_atoms   # Langevin acts independently per atom/DOF, incl. COM

kB = units.BOLTZMANN_CONSTANT


def kinetic_temperature(integrator):
    return 2.0 * integrator.kinetic_energy().item() / (n_dof * kB)


velocities = torch.zeros_like(opt_coordinates)
generator = torch.Generator(device=device).manual_seed(0)
dyn = Langevin(model, species, opt_coordinates, velocities, masses,
               timestep=timestep, temperature=temperature, friction=friction,
               generator=generator)

print(f'target bath temperature: {temperature:.1f} K, n_dof = {n_dof}')

n_equilibration = 1000


def report(step, coordinates, velocities, energy, forces):
    if (step + 1) % 200 == 0:
        print(f'  step {step + 1:5d}: E_pot = {energy.item(): .6f} Ha, '
              f'T_kin = {kinetic_temperature(dyn):7.2f} K')


dyn.run(n_equilibration, callback=report)

n_sample = 3000
temps = []


def sample(step, coordinates, velocities, energy, forces):
    temps.append(kinetic_temperature(dyn))
    if (step + 1) % 500 == 0:
        print(f'  step {n_equilibration + step + 1:5d}: E_pot = {energy.item(): .6f} Ha, '
              f'T_kin = {kinetic_temperature(dyn):7.2f} K')


final_coordinates = dyn.run(n_sample, callback=sample)

mean_T = sum(temps) / len(temps)
var_T = sum((t - mean_T) ** 2 for t in temps) / len(temps)
expected_relative_fluctuation = math.sqrt(2.0 / n_dof)
print(f'\nmean kinetic temperature over last {n_sample} steps: {mean_T:.2f} K '
      f'(target {temperature:.1f} K)')
print(f'std/mean of T_kin: {math.sqrt(var_T) / mean_T:.4f} '
      f'(equipartition prediction sqrt(2/n_dof) = {expected_relative_fluctuation:.4f})')
print(f'final coordinates shape: {tuple(final_coordinates.shape)}, '
      f'finite: {torch.isfinite(final_coordinates).all().item()}')
print(f'radius of gyration (final): '
      f'{final_coordinates[0].std(dim=0).norm().item():.3f} A '
      f'(initial relaxed: {opt_coordinates[0].std(dim=0).norm().item():.3f} A)')
