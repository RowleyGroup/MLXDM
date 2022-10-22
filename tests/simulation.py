import ase
from ase.data.pubchem import pubchem_atoms_search
from ase.io.trajectory import Trajectory
from ase import units
from ase.md.langevin import Langevin
from torchanipbe0 import models

# System preparation
atoms1 = pubchem_atoms_search('HCOOH')
atoms2 = atoms1.copy()
atoms2.rotate(180, 'z')
atoms2.translate((0.2, 4.5, 0.0))
atoms = atoms1 + atoms2

# Setup simulation
model = models.ANIPBE0_MLXDM()
atoms.set_calculator(model.ase())
dyn = Langevin(atoms, timestep=1.0*units.fs, friction=1e-4, temperature_K=300)
traj = Trajectory('formic_dimer.traj', 'w', atoms)
dyn.attach(traj.write, interval=1)
dyn.run(10000)
traj.close()