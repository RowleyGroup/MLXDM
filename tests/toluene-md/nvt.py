import ase
import ase.io
import ase.io.trajectory
import ase.eos

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory

import sys
import torchanipbe0
import torch
import ase.optimize
import ase.optimize.precon
import ase.units
import ase.constraints
import ase.build

import numpy

T=298.15

mol=ase.io.read('toluene_0.pdb')
mol.set_cell([(18.814466395, 0, 0), (0, 18.814466395, 0), (0, 0, 18.814466395)])
mol.set_pbc((True, True, True))
calculator = torchanipbe0.models.ANIPBE0_MLXDM().ase()

mol.set_calculator(calculator)


ei = mol.get_potential_energy()
cell = mol.get_cell()
print(cell)
print(mol.get_pbc())
opt = ase.optimize.FIRE(mol)
opt.run(fmax=0.1)

MaxwellBoltzmannDistribution(mol, temperature_K=298.15)

# We want to run MD with constant energy using the VelocityVerlet algorithm.
#dyn = VelocityVerlet(mol, 1 * ase.units.fs)  # 5 fs time step.
dyn = Langevin(mol, 1*ase.units.fs, T*ase.units.kB, 0.002)

def printenergy(a=mol):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * ase.units.kB), epot + ekin))


print('starting MD')
traj = Trajectory('toluene_md_0.traj', 'w', mol)
dyn.attach(traj.write, interval=100)

# Now run the dynamics
dyn.attach(printenergy, interval=10)

dyn.run(10000000)

ase.io.write('nvt.xyz', mol)

