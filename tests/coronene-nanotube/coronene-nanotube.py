import torch
import torchanipbe0
import ase
import ase.build
import ase.io
import ase.md
import ase.optimize
import ase.units
import ase.io.trajectory
import sys
import numpy as np
from scipy.spatial.transform import Rotation

def structure_rotate(first):
    # 2 rotation and 1 translation vector
    r1 = Rotation.from_euler('xyz', 
                        [0, 77.0/360.0, 0]).as_matrix()
    coord_1 = np.matmul(first, r1)
    return coord_1

total_z=63.9
n_coro=int(sys.argv[1])
coronene_spacing=total_z/n_coro

coronene=ase.io.read('coronene.xyz')

coronene_initial_crd=structure_rotate(coronene.get_positions())
print(coronene_initial_crd)

model=torchanipbe0.models.ANIPBE0_MLXDM()
calculator = torchanipbe0.models.ANIPBE0_MLXDM().ase()
coronene.set_calculator(calculator)

atoms = ase.build.nanotube(19, 0, length=15)
print(atoms)

for i in range(n_coro):
    coronene.set_positions(coronene_initial_crd + [0,0,coronene_spacing*i])
    atoms+=coronene

ase.io.write('coronene-nt-initial_' + sys.argv[1] + '.xyz', atoms)
atoms.set_calculator(calculator)

atoms.set_cell( [[32, 0, 0], [0, 32, 0], [0, 0, 63.9]])
atoms.set_pbc((True, True, True))

model=torchanipbe0.models.ANIPBE0_MLXDM()

calculator = torchanipbe0.models.ANIPBE0_MLXDM().ase()

atoms.set_calculator(calculator)

print("Begin minimizing...")
opt = ase.optimize.BFGS(atoms)
opt.run(fmax=0.001)
ase.io.write('coronene_rotate_mlxdm_opt_' + sys.argv[1] + '.xyz', atoms)
en=atoms.get_potential_energy()

fhout=open('coronene_rotate_mlxdm_opt_' + sys.argv[1] + '.txt', 'w')
fhout.write(sys.argv[1] + ' ' + str(en) + '\n')
fhout.close()
print()

