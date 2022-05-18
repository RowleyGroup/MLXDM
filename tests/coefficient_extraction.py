# TEST DIMER INTERACTION ENERGY
# Author: Tu Nguyen Thien Phuc
# Last update: 2021-09-30

import ase.io as io
from ase import units

import torch
import numpy as np
import pandas as pd
import torchanipbe0
from torchanipbe0 import models, coefficient
import matplotlib.pyplot as plt

model_list = [coefficient.c6_coefficient_model(),
              coefficient.c8_coefficient_model(),
              coefficient.c10_coefficient_model()]

def read_postg(file_name):
    with open(file_name,'r') as f:
        lines = f.readlines()
        total = 0
        i = 0
        c6_list = []
        c8_list = []
        c10_list = []
        while i < len(lines):
            line = lines[i].split()
            if not line:
                i += 1
                continue
            if line[0] == 'natoms':
                total = int(line[1])
                i += 1
                continue
            if line[0] == 'coefficients':
                i += 2
                for j in range(total):
                    for k in range(j, total):
                        if j == k:
                            line = lines[i].split()
                            c6_list.append(float(line[3]))
                            c8_list.append(float(line[4]))
                            c10_list.append(float(line[5]))
                        i += 1
                continue
            i += 1
    return c6_list, c8_list, c10_list

if __name__ == '__main__':
    result = {'C6':[], 'C8':[], 'C10':[]}
    ref = {'C6':[], 'C8':[], 'C10':[]}
    for r in [0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]:
        xyz_file = f'xyz/S22by7_1_{r}_dimer.xyz'
        postg_file = f'postg_output/s22_1_{r}-postg.out'
        atoms = io.read(xyz_file)
        coef_list = read_postg(postg_file)
        # print(f'******************System: {r}********************')
        for name, model, coef in zip(['C6','C8','C10'], model_list, coef_list):
            # print(f'  Coefficient: {name}')
            outp = model.compute_from_ase(atoms)
            result[name] = result[name] + list(outp)
            ref[name] = ref[name] + coef
    fig = plt.figure()
    fig.set_size_inches(10,5)
    for i, name in enumerate(['C6', 'C8', 'C10']):
        ax = fig.add_subplot(1, 3, i+1)
        ax.scatter(ref[name], result[name])
        ax.set_title(name)
        ax.set_aspect('equal')
        ax.set_xlabel('postg coefficient')
        ax.set_ylabel('ML coefficient')
    fig.tight_layout()
    plt.savefig('coef.png', dpi = 400)     



