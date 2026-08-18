# GPU LANGEVIN THERMOSTAT WITH THE REAL ANIPBE0_2x + MLXDM_2x POTENTIAL
#
# Runs torchanipbe0.md.Langevin against the actual combined short-range +
# dispersion neural-network potential (not a toy LJ model), on a cluster of
# water molecules, entirely through tensors kept on the GPU. Confirms the
# integrator works end-to-end with a real ANIDispersion model: gradients
# flow through both the ANI and MLXDM sub-networks every step, forces stay
# finite, and the kinetic temperature tracks the requested bath temperature.

import torch
from ase.build import molecule

from torchanipbe0 import models, units
from torchanipbe0.md import optimize_geometry, Langevin, VelocityVerlet

device = torch.device('cuda')
dtype = torch.float32

torch.manual_seed(0)
model = models.ANIPBE0_2x_MLXDM_2x(device)

# Build a cluster of water molecules on a cubic grid, each with a random
# rotation/jitter so the optimizer isn't handed a degenerate symmetric start.
water = molecule('H2O')
water_symbols = list(water.get_chemical_symbols())
water_positions = torch.tensor(water.get_positions(), dtype=dtype)

n_side = 3
spacing = 4.5  # Angstrom, enough to keep molecules from clashing pre-relaxation
symbols = []
coords_list = []
for ix in range(n_side):
    for iy in range(n_side):
        for iz in range(n_side):
            offset = torch.tensor([ix, iy, iz], dtype=dtype) * spacing
            # Random rotation via a random small-angle axis-angle perturbation
            # composed a few times, applied about the molecule's own centroid.
            pos = water_positions - water_positions.mean(dim=0, keepdim=True)
            angle = torch.rand(3) * 2 * torch.pi
            cx, cy, cz = torch.cos(angle)
            sx, sy, sz = torch.sin(angle)
            Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            pos = pos @ R.T
            jitter = (torch.rand(3) - 0.5) * 0.6
            pos = pos + offset + jitter
            coords_list.append(pos)
            symbols.extend(water_symbols)

n_mol = n_side ** 3
coordinates = torch.cat(coords_list, dim=0).unsqueeze(0).to(device)  # (1, n_atoms, 3)
n_atoms = coordinates.shape[1]

species = model.ani_model.species_to_tensor(symbols).unsqueeze(0).to(device)
masses_per_atom = {'H': 1.008, 'O': 15.999}
masses = torch.tensor([masses_per_atom[s] for s in symbols], dtype=dtype, device=device)

print(f'Water cluster: {n_mol} molecules, {n_atoms} atoms '
      f'(ANIPBE0_2x + MLXDM_2x, real neural-network potential)')

opt_coordinates, energy, forces, converged = optimize_geometry(
    model, species, coordinates, fmax=2e-3, max_steps=300)
print(f'relaxed (converged={converged}): E = {energy.item(): .6f} Ha, '
      f'fmax = {forces.abs().max().item():.2e} Ha/A')

# Short NVE sanity check before trusting the thermostat.
velocities0 = torch.zeros_like(opt_coordinates)
nve = VelocityVerlet(model, species, opt_coordinates, velocities0, masses, timestep=0.25)
e0 = nve.total_energy().item()
nve.run(100)
e1 = nve.total_energy().item()
print(f'NVE sanity check: energy drift over 100 steps of 0.25 fs: {e1 - e0:.3e} Ha '
      f'(E0 = {e0:.6f} Ha)\n')

# Langevin thermostat at room temperature.
temperature = 300.0
friction = 0.01
timestep = 0.25
n_dof = 3 * n_atoms

kB = units.BOLTZMANN_CONSTANT


def kinetic_temperature(integrator):
    return 2.0 * integrator.kinetic_energy().item() / (n_dof * kB)


velocities = torch.zeros_like(opt_coordinates)
generator = torch.Generator(device=device).manual_seed(0)
dyn = Langevin(model, species, opt_coordinates, velocities, masses,
               timestep=timestep, temperature=temperature, friction=friction,
               generator=generator)

print(f'target bath temperature: {temperature:.1f} K, n_dof = {n_dof}')


def report(step, coordinates, velocities, energy, forces):
    if (step + 1) % 200 == 0:
        finite = torch.isfinite(forces).all().item()
        print(f'  step {step + 1:4d}: E_pot = {energy.item(): .6f} Ha, '
              f'T_kin = {kinetic_temperature(dyn):7.1f} K, forces finite = {finite}')


n_equilibration = 2000
dyn.run(n_equilibration, callback=report)

n_sample = 1000
temps = []


def sample(step, coordinates, velocities, energy, forces):
    temps.append(kinetic_temperature(dyn))
    if (step + 1) % 200 == 0:
        print(f'  step {n_equilibration + step + 1:4d}: E_pot = {energy.item(): .6f} Ha, '
              f'T_kin = {kinetic_temperature(dyn):7.1f} K')


final_coordinates = dyn.run(n_sample, callback=sample)

mean_T = sum(temps) / len(temps)
print(f'\nmean kinetic temperature over last {n_sample} steps: {mean_T:.1f} K '
      f'(target {temperature:.1f} K)')
print(f'final coordinates shape: {tuple(final_coordinates.shape)}, '
      f'finite: {torch.isfinite(final_coordinates).all().item()}')
