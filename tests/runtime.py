# CHECK THE RUNTIME OF EXISTED MODELS
# Author: Tu Nguyen Thien Phuc
# Last update: 2021-09-30

import numpy as np
import time

def run_time(func, loop = 1, repeat = 5, **kwargs):
    lst = []
    s = 0.0
    for i in range(repeat):
        start_time = time.time()
        for j in range(loop):
            func(**kwargs)
        end_time = time.time()
        t = (end_time - start_time) / loop
        s += t
        lst.append(t)
    s /= repeat
    var = 0.0
    for i in range(repeat):
        var += (lst[i] - s)**2
    var /= repeat
    var = var ** 0.5
    return s, var
    
import torchanipbe0
from torchanipbe0 import models
import torch
import numpy as np
import ase
from ase import io
from ase.md.verlet import VelocityVerlet
from ase.lattice.hexagonal import *
from ase import units

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
    # atoms = io.read('xyz/S22by7_6_1.2_dimer.xyz')
    atoms = Graphite(symbol = 'C',
                     latticeconstant={'a': 2.461,'c': 6.708},
                     # size = (3,3,2),
                     pbc=True)
    device = torch.device('cuda')

    model1 = models.ANI2x()
    model2 = models.ANI2x().to(device)
    model3 = models.ANIPBE0()
    model4 = models.ANIPBE0().to(device)
    model5 = models.ANIPBE0_MLXDM()
    model6 = models.ANIPBE0_MLXDM().to(device)
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
        








