# OPTIMIZATION OF THE UNIT CELL OF GRAPHITE
# Author: Tu Nguyen Thien Phuc
# Last update: 2021-12-16

from ase.lattice.hexagonal import *
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
from ase.constraints import UnitCellFilter

import torchanipbe0
from torchanipbe0 import models

model = models.ANIPBE0_Dispersion_Constant_Coef(exact_combination=True)
graphite = Graphite(symbol = 'C',
                 latticeconstant={'a': 2.461,'c': 6.708},
                 pbc=True)
graphite.set_calculator(model.ase(overwrite = True))
def report(x = graphite):
    print(f'Cell     : {x.get_cell()}')
cell = UnitCellFilter(graphite, mask = [True, True, True, False, False, True])
dyn = BFGS(cell)
dyn.attach(report, interval = 500)
dyn.run(fmax = 0.1, steps = 5000)
