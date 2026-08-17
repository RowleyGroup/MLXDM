# -*- coding: utf-8 -*-
"""Pure PyTorch geometry optimization and NVE molecular dynamics.

These routines drive any torchanipbe0 potential (anything exposing
``model((species, coordinates), cell=cell, pbc=pbc) -> SpeciesEnergies``,
e.g. :func:`torchanipbe0.models.ANIPBE0_2x` or
:func:`torchanipbe0.models.ANIPBE0_2x_MLXDM_2x`) directly through tensors
that stay on the model's device for the whole run.

Unlike ``model.ase()`` combined with an ``ase.optimize``/``ase.md`` driver,
there is no per-step round trip of coordinates/forces through
``ase.Atoms``/``ase.calculators`` (which forces a host<->device numpy copy
every step); coordinates, velocities, masses and forces are torch tensors
from start to finish, so geometry optimization and MD can run entirely on
the GPU.

Distances are in Angstrom, energies in Hartree and masses in AMU,
consistent with the rest of the package. Timesteps and velocities use
femtoseconds and Angstrom/fs (see :mod:`torchanipbe0.units`).
"""
from typing import Callable, NamedTuple, Optional, Tuple

import torch
from torch import Tensor

from . import units


class EnergyForces(NamedTuple):
    energy: Tensor
    forces: Tensor


def _energy_forces(model, species: Tensor, coordinates: Tensor,
                    cell: Optional[Tensor] = None, pbc: Optional[Tensor] = None) -> EnergyForces:
    """Single energy/force evaluation, entirely on ``coordinates.device``."""
    coordinates = coordinates.detach().requires_grad_(True)
    energy = model((species, coordinates), cell=cell, pbc=pbc).energies
    forces = -torch.autograd.grad(energy.sum(), coordinates)[0]
    return EnergyForces(energy.detach(), forces.detach())


def optimize_geometry(model, species: Tensor, coordinates: Tensor,
                       cell: Optional[Tensor] = None, pbc: Optional[Tensor] = None,
                       fmax: float = 4.5e-4, max_steps: int = 1000, lr: float = 1.0,
                       history_size: int = 100, max_iter: int = 20,
                       callback: Optional[Callable[[int, Tensor, Tensor, Tensor], None]] = None
                       ) -> Tuple[Tensor, Tensor, Tensor, bool]:
    """Relax ``coordinates`` under ``model`` with L-BFGS, on-device.

    Arguments:
        model: a torchanipbe0 potential, e.g. ``models.ANIPBE0_2x(...)`` or
            ``models.ANIPBE0_2x_MLXDM_2x(device)``.
        species: ``(n_batch, n_atoms)`` long tensor, already on the model's
            device and using whatever species indexing ``model`` expects.
        coordinates: ``(n_batch, n_atoms, 3)`` tensor in Angstrom, on the
            model's device. Not modified in place.
        cell, pbc: passed straight through to ``model``, for periodic
            systems.
        fmax: convergence threshold on the largest force component
            (Hartree/Angstrom).
        max_steps: maximum number of L-BFGS steps to take.
        lr, history_size, max_iter: passed to ``torch.optim.LBFGS``.
        callback: optional ``callback(step, energy, forces, coordinates)``
            called after every step.

    Returns:
        ``(coordinates, energy, forces, converged)``, all still on the
        model's device.
    """
    coordinates = coordinates.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS([coordinates], lr=lr, history_size=history_size,
                                   max_iter=max_iter, line_search_fn='strong_wolfe')

    energy = forces = None

    def closure():
        nonlocal energy, forces
        optimizer.zero_grad()
        energy = model((species, coordinates), cell=cell, pbc=pbc).energies
        energy.sum().backward()
        forces = -coordinates.grad.detach()
        return energy.sum()

    converged = False
    for step in range(max_steps):
        optimizer.step(closure)
        if callback is not None:
            callback(step, energy.detach(), forces, coordinates.detach())
        if forces.abs().max() <= fmax:
            converged = True
            break

    return coordinates.detach(), energy.detach(), forces, converged


class VelocityVerlet:
    """NVE molecular dynamics with the velocity Verlet algorithm, on-device.

    All state (coordinates, velocities, forces) is kept as torch tensors on
    ``coordinates.device`` for the whole trajectory; ``model`` is queried
    directly for energies/forces via autograd, with no ``ase.Atoms``/
    ``ase.Calculator`` in the loop.

    Arguments:
        model: a torchanipbe0 potential, e.g. ``models.ANIPBE0_2x(...)`` or
            ``models.ANIPBE0_2x_MLXDM_2x(device)``.
        species: ``(n_batch, n_atoms)`` long tensor.
        coordinates: ``(n_batch, n_atoms, 3)`` tensor, Angstrom.
        velocities: ``(n_batch, n_atoms, 3)`` tensor, Angstrom/fs.
        masses: ``(n_batch, n_atoms)`` or ``(n_atoms,)`` tensor, AMU.
        timestep: MD timestep, in femtoseconds.
        cell, pbc: passed straight through to ``model``, for periodic
            systems.
    """

    def __init__(self, model, species: Tensor, coordinates: Tensor, velocities: Tensor,
                 masses: Tensor, timestep: float,
                 cell: Optional[Tensor] = None, pbc: Optional[Tensor] = None):
        self.model = model
        self.species = species
        self.cell = cell
        self.pbc = pbc
        self.timestep = timestep

        device = coordinates.device
        dtype = coordinates.dtype
        self.coordinates = coordinates.clone().detach().to(device=device, dtype=dtype)
        self.velocities = velocities.clone().detach().to(device=device, dtype=dtype)

        masses = torch.as_tensor(masses, device=device, dtype=dtype)
        if masses.dim() == self.coordinates.dim() - 2:
            masses = masses.unsqueeze(0)
        self.masses = masses.unsqueeze(-1)  # broadcasts over x, y, z

        # Converts force/mass (Hartree / (Angstrom * AMU)) into
        # Angstrom/fs^2: see units.HARTREE_ANGSTROM_AMU_TIME_TO_FS.
        self._accel_factor = 1.0 / units.HARTREE_ANGSTROM_AMU_TIME_TO_FS ** 2

        self.energy, self.forces = _energy_forces(self.model, self.species, self.coordinates,
                                                    self.cell, self.pbc)
        self._accelerations = self.forces / self.masses * self._accel_factor

    def kinetic_energy(self) -> Tensor:
        """Kinetic energy per batch entry, in Hartree."""
        return 0.5 * (self.masses * self.velocities ** 2).sum(dim=(-2, -1)) \
            / self._accel_factor

    def total_energy(self) -> Tensor:
        """Potential + kinetic energy per batch entry, in Hartree."""
        return self.energy + self.kinetic_energy()

    def step(self) -> EnergyForces:
        dt = self.timestep
        self.coordinates = self.coordinates + self.velocities * dt + 0.5 * self._accelerations * dt ** 2
        self.energy, self.forces = _energy_forces(self.model, self.species, self.coordinates,
                                                    self.cell, self.pbc)
        new_accelerations = self.forces / self.masses * self._accel_factor
        self.velocities = self.velocities + 0.5 * (self._accelerations + new_accelerations) * dt
        self._accelerations = new_accelerations
        return EnergyForces(self.energy, self.forces)

    def run(self, n_steps: int,
            callback: Optional[Callable[[int, Tensor, Tensor, Tensor, Tensor], None]] = None
            ) -> Tuple[Tensor, Tensor]:
        """Run ``n_steps`` steps of velocity Verlet.

        ``callback``, if given, is called after every step as
        ``callback(step, coordinates, velocities, energy, forces)``.

        Returns the final ``(coordinates, velocities)``.
        """
        for step in range(n_steps):
            self.step()
            if callback is not None:
                callback(step, self.coordinates, self.velocities, self.energy, self.forces)
        return self.coordinates, self.velocities
