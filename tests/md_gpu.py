# GPU GEOMETRY OPTIMIZATION AND NVE MOLECULAR DYNAMICS, NO ASE ROUND TRIP
#
# Demonstrates torchanipbe0.md.optimize_geometry and torchanipbe0.md.VelocityVerlet
# driving ANIPBE0_2x and ANIPBE0_2x + MLXDM_2x directly through tensors that stay
# on the GPU for the whole run (species/coordinates/masses/velocities/timestep/
# number of steps are all supplied by code, not read from a file).

import torch

from torchanipbe0 import models
from torchanipbe0.md import optimize_geometry, VelocityVerlet

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

# Small initial velocities given directly in code (Angstrom/fs).
torch.manual_seed(0)
velocities = 0.002 * torch.randn(1, 5, 3, dtype=torch.float32, device=device)
velocities -= velocities.mean(dim=1, keepdim=True)  # remove net linear momentum


def run_for(model, name):
    species = model.ani_model.species_to_tensor(symbols).unsqueeze(0).to(device) \
        if hasattr(model, 'ani_model') else model.species_to_tensor(symbols).unsqueeze(0).to(device)

    print(f'--- {name} ---')
    opt_coordinates, energy, forces, converged = optimize_geometry(
        model, species, coordinates, fmax=1e-3, max_steps=200)
    print(f'optimized (converged={converged}): E = {energy.item(): .6f} Ha, '
          f'fmax = {forces.abs().max().item():.2e} Ha/A')

    dyn = VelocityVerlet(model, species, opt_coordinates, velocities, masses, timestep=0.5)
    e0 = dyn.total_energy().item()

    def report(step, coordinates, velocities, energy, forces):
        if (step + 1) % 20 == 0:
            print(f'  step {step + 1:4d}: E_pot = {energy.item(): .6f} Ha, '
                  f'E_tot = {dyn.total_energy().item(): .6f} Ha')

    dyn.run(100, callback=report)
    e1 = dyn.total_energy().item()
    print(f'energy drift over 100 steps of 0.5 fs: {e1 - e0:.3e} Ha\n')


if __name__ == '__main__':
    run_for(models.ANIPBE0_2x(periodic_table_index=False).to(device), 'ANIPBE0_2x')
    run_for(models.ANIPBE0_2x_MLXDM_2x(device), 'ANIPBE0_2x + MLXDM_2x')
