# CHECK THE RUNTIME OF EXISTED MODELS
# Author: Tu Nguyen Thien Phuc
# Last update: 2021-09-30

import numpy as np
import time

    
import torchanipbe0
from torchanipbe0 import models
import torch
import numpy as np
import ase
from ase import io
from ase.md.verlet import VelocityVerlet
from ase.lattice.hexagonal import *
from ase import units

# Declare the system



def print_result(result):
    for i, r in enumerate(result):
        print(f'Model {i} :{r[0]} +/- {r[1]}')

def f(atoms = None, model = None):
    atoms.set_calculator(model.ase())
    atoms.get_potential_energy()

def g(atoms = None, model = None):
    atoms.set_calculator(model.ase())
    atoms.get_forces()

def h(atoms = None, model = None):
    atoms.set_calculator(model.ase())
    dyn = VelocityVerlet(atoms, timestep = 1.0 * units.fs)
    dyn.run(1000)

if __name__ == '__main__':
    # atoms = io.read('xyz/S22by7_1_0.7_dimer.xyz')
    atoms = io.read('xyz/S22by7_6_1.2_dimer.xyz')
    # atoms = Graphite(symbol = 'C',
    #                  latticeconstant={'a': 2.461,'c': 6.708},
    #                  # size = (3,3,2),
    #                  pbc=True)
    device = torch.device('cuda')

    model1 = models.ANI2x()
    model2 = models.ANI2x().to(device)
    model3 = models.ANIPBE0()
    model4 = models.ANIPBE0().to(device)
    model5 = models.ANIPBE0_MLXDM(exact_combination=True)
    model6 = models.ANIPBE0_MLXDM(exact_combination=True).to(device)
    model_list = [model1, model2, model3, model4, model5, model6]
    result = []
    for model in model_list:
        result.append(run_time(f, 100, 5, atoms = atoms, model = model))
    print_result(result)
    # result = []
    # for model in model_list:
    #     result.append(run_time(g, 100, 7, atoms = atoms, model = model))
    # print_result(result)
    # result = []
    # for model in model_list:
    #     result.append(run_time(h, 1, 3, atoms = atoms, model = model))
    # print_result(result)
        








