# GPU LANGEVIN THERMOSTAT, NO ASE ROUND TRIP
#
# Demonstrates torchanipbe0.md.Langevin driving ANIPBE0_2x directly through
# tensors that stay on the GPU for the whole run. Checks that the kinetic
# temperature relaxes toward and then fluctuates around the requested bath
# temperature.

import torch

from torchanipbe0 import models, units
from torchanipbe0.md import optimize_geometry, Langevin

device = torch.device('cuda')

# A slightly stretched methane, given directly in code (Angstrom).
symbols = ['C', 'H', 'H', 'H', 'H']
coordinates = torch.tensor([[
    [0.0000, 0.0000, 0.0000],
    [0.7297, 0.6297, 0.6297],
    [-0.6297, -0.6297, 0.6297],
    [-0.6297, 0.6297, -0.6297],
    [0.6297, -0.6297, -0.6297],
]], dtype=torch.float32, device=device)

# Masses given directly in code (AMU): C, H, H, H, H.
masses = torch.tensor([12.011, 1.008, 1.008, 1.008, 1.008], dtype=torch.float32, device=device)

n_atoms = len(symbols)
n_dof = 3 * n_atoms - 3  # remove net translation
temperature = 300.0

torch.manual_seed(0)
model = models.ANIPBE0_2x(periodic_table_index=False).to(device)
species = model.species_to_tensor(symbols).unsqueeze(0).to(device)

opt_coordinates, energy, forces, converged = optimize_geometry(
    model, species, coordinates, fmax=1e-3, max_steps=200)
print(f'optimized (converged={converged}): E = {energy.item(): .6f} Ha, '
      f'fmax = {forces.abs().max().item():.2e} Ha/A')

velocities = torch.zeros_like(coordinates)  # start cold
generator = torch.Generator(device=device).manual_seed(0)

dyn = Langevin(model, species, opt_coordinates, velocities, masses,
               timestep=0.5, temperature=temperature, friction=0.01,
               generator=generator)

kB = units.BOLTZMANN_CONSTANT


def kinetic_temperature():
    return 2.0 * dyn.kinetic_energy().item() / (n_dof * kB)


print(f'target bath temperature: {temperature:.1f} K')


def report(step, coordinates, velocities, energy, forces):
    if (step + 1) % 200 == 0:
        print(f'  step {step + 1:5d}: E_pot = {energy.item(): .6f} Ha, '
              f'T_kin = {kinetic_temperature():7.1f} K')


n_equilibration = 1000
dyn.run(n_equilibration, callback=report)

n_sample = 4000
temps = []


def sample(step, coordinates, velocities, energy, forces):
    temps.append(kinetic_temperature())
    if (step + 1) % 500 == 0:
        print(f'  step {n_equilibration + step + 1:5d}: E_pot = {energy.item(): .6f} Ha, '
              f'T_kin = {kinetic_temperature():7.1f} K')


final_coordinates = dyn.run(n_sample, callback=sample)

mean_T = sum(temps) / len(temps)
print(f'\nmean kinetic temperature over last {n_sample} steps: {mean_T:.1f} K '
      f'(target {temperature:.1f} K)')
print(f'final coordinates shape: {tuple(final_coordinates.shape)}, '
      f'finite: {torch.isfinite(final_coordinates).all().item()}')
