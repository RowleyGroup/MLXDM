# Split the dispersion energy
import torchanipbe0
from torchanipbe0 import models
import ase
from ase import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


traj_file_name = 'toluene_md.xyz'
index_file = 'index.txt'
output_file_name = 'dispersion_data.csv'

# General
atoms_list = io.read(traj_file_name, index=':')
n = len(atoms_list)
# atoms.set_pbc([False, False, False])
model_list = [models.ANIPBE0_C6().ase(), 
              models.ANIPBE0_C8().ase(), 
              models.ANIPBE0_C10().ase(), 
              models.ANIPBE0_MLXDM(dispersion_only=True).ase()]
temp = np.zeros((n, 8))
data = pd.DataFrame(temp, columns = ['C6_energy', 'C8_energy', 'C10_energy', 'dispersion_energy',
                                     'C6_intra', 'C8_intra', 'C10_intra', 'dispersion_intra'])

# Index handle
f = open(index_file, 'r')
index_list = f.readlines()[0].split()
f.close()
index_list = [int(x) for x in index_list]
index_list = np.array(index_list)
index_unique = np.unique(index_list)

for i, atoms in enumerate(atoms_list):
    atoms.set_pbc([False, False, False])
    for j in range(4): # 4 models
    # for name, model in model_list.items():
        # print((i,j))
        name = ['C6_energy', 'C8_energy', 'C10_energy', 'dispersion_energy'][j]
        name_2 = ['C6_intra', 'C8_intra', 'C10_intra', 'dispersion_intra'][j]
        atoms.set_calculator(model_list[j])
        data.loc[i, name] = atoms.get_potential_energy()*23.0261
        data.loc[i, name_2] = 0.0
        for ind in index_unique:
            sub_atom = atoms[index_list==ind]
            sub_atom.set_calculator(model_list[j])
            data.loc[i, name_2] += sub_atom.get_potential_energy()*23.0261

data.to_csv(output_file_name)

# Plotting
# All
plt.plot(data.index, -data['C6_energy'], label='C6')
plt.plot(data.index, -data['C8_energy'], label='C8')
plt.plot(data.index, -data['C10_energy'], label='C10')
plt.plot(data.index, -data['dispersion_energy'], label='total')
plt.legend()
plt.axhline(y=0, alpha=0.0)
plt.xlabel('Iteration')
plt.ylabel('- Energy (kcal/mol)')
plt.savefig('dispersion_all.png', dpi=300)
# Just intermolecular
plt.plot(data.index, -data['C6_energy']+data['C6_intra'], label='C6')
plt.plot(data.index, -data['C8_energy']+data['C8_intra'], label='C8')
plt.plot(data.index, -data['C10_energy']+data['C10_intra'], label='C10')
plt.plot(data.index, -data['dispersion_energy']+data['dispersion_intra'], label='total')
plt.legend()
plt.axhline(y=0, alpha=0.0)
plt.xlabel('Iteration')
plt.ylabel('- Energy (kcal/mol)')
plt.savefig('dispersion_inter.png', dpi=300)

