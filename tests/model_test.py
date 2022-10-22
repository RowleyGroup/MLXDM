import torchanipbe0
from torchanipbe0 import models

import ase
from ase.build import molecule

import torch

atoms = molecule('H2O')
device = torch.device('cuda')

# Test direct potential

# ANI1x

print('ANI1x')
try:
    model = models.ANI1x()
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

print('ANI1x CUDA')
try:
    model = model.to(device)
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

# ANIPBE0

print('ANIPBE0')
try:
    model = models.ANIPBE0()
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

print('ANIPBE0 CUDA')
try:
    model = model.to(device)
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

# MLXDM
print('MLXDM')
try:
    model = models.MLXDM()
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

print('MLXDM CUDA')
try:
    model = models.MLXDM(device)
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

# XDM_CC
print('XDM_CC')
try:
    model = models.XDM_CC()
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

print('XDM_CC CUDA')
try:
    model = models.XDM_CC(device)
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

# MLXDM simple
print('MLXDM_simple')
try:
    model = models.MLXDM_simple()
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

print('MLXDM_simple CUDA')
try:
    model = models.MLXDM_simple(device)
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

# XDM_CC simple
print('XDM_CC_simple')
try:
    model = models.XDM_CC_simple()
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

print('XDM_CC_simple CUDA')
try:
    model = models.XDM_CC_simple(device)
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

# ANIPBE0_MLXDM
print('ANIPBE0_MLXDM')
try:
    model = models.ANIPBE0_MLXDM()
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

print('ANIPBE0_MLXDM CUDA')
try:
    model = models.ANIPBE0_MLXDM(device)
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

# ANIPBE0_XDM_CC
print('ANIPBE0_XDM_CC')
try:
    model = models.ANIPBE0_XDM_CC()
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

print('ANIPBE0_XDM_CC CUDA')
try:
    model = models.ANIPBE0_XDM_CC(device)
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

# Get individual energy

# C6 energy
print('ANIPBE0_MLXDM C6 energy')
try:
    model = models.c6_energy()
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

print('ANIPBE0_MLXDM C6 energy CUDA')
try:
    model = models.c6_energy(device)
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

# C8 energy
print('ANIPBE0_MLXDM C8 energy')
try:
    model = models.c8_energy()
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

print('ANIPBE0_MLXDM C8 energy CUDA')
try:
    model = models.c8_energy(device)
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

# C10 energy
print('ANIPBE0_MLXDM C10 energy')
try:
    model = models.c10_energy()
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

print('ANIPBE0_MLXDM C10 energy CUDA')
try:
    model = models.c10_energy(device)
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

# C6 energy simple
print('ANIPBE0_MLXDM_simple C6 energy')
try:
    model = models.c6_simple_energy()
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

print('ANIPBE0_MLXDM_simple C6 energy CUDA')
try:
    model = models.c6_simple_energy(device)
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

# C8 energy simple
print('ANIPBE0_MLXDM_simple C8 energy')
try:
    model = models.c8_simple_energy()
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

print('ANIPBE0_MLXDM_simple C8 energy CUDA')
try:
    model = models.c8_simple_energy(device)
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

# C10 energy simple
print('ANIPBE0_MLXDM_simple C10 energy')
try:
    model = models.c10_simple_energy()
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

print('ANIPBE0_MLXDM_simple C10 energy CUDA')
try:
    model = models.c10_simple_energy(device)
    atoms.set_calculator(model.ase())
    print(atoms.get_potential_energy())
    print('Success')
except:
    print('ERROR')

# Get individual coefficients

# M1
print('M1 coefficients')
try:
    model = models.m1_coefficients()
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

print('M1 coefficients CUDA')
try:
    model = models.m1_coefficients(device)
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

# M2
print('M2 coefficients')
try:
    model = models.m2_coefficients()
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

print('M2 coefficients CUDA')
try:
    model = models.m2_coefficients(device)
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

# M3
print('M3 coefficients')
try:
    model = models.m3_coefficients()
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

print('M3 coefficients CUDA')
try:
    model = models.m3_coefficients(device)
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

# V
print('V coefficients')
try:
    model = models.v_coefficients()
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

print('V coefficients CUDA')
try:
    model = models.v_coefficients(device)
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

# M1 CC
print('M1 coefficients CC')
try:
    model = models.m1_coefficients_CC()
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

print('M1 coefficients CC CUDA')
try:
    model = models.m1_coefficients_CC(device)
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

# M2 CC
print('M2 coefficients CC')
try:
    model = models.m2_coefficients_CC()
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

print('M2 coefficients CC CUDA')
try:
    model = models.m2_coefficients_CC(device)
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

# M3
print('M3 coefficients CC')
try:
    model = models.m3_coefficients_CC()
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

print('M3 coefficients CC CUDA')
try:
    model = models.m3_coefficients_CC(device)
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

# V
print('V coefficients CC')
try:
    model = models.v_coefficients_CC()
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

print('V coefficients CC CUDA')
try:
    model = models.v_coefficients_CC(device)
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')


# atomic C6
print('C6 coefficients')
try:
    model = models.c6_coefficients()
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

print('C6 coefficients CUDA')
try:
    model = models.c6_coefficients(device)
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

# atomic C8
print('C8 coefficients')
try:
    model = models.c8_coefficients()
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

print('C8 coefficients CUDA')
try:
    model = models.c8_coefficients(device)
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

# atomic C10
print('C10 coefficients')
try:
    model = models.c10_coefficients()
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

print('C10 coefficients CUDA')
try:
    model = models.c10_coefficients(device)
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

# Pairwise energy
print('Pairwise')
try:
    model = models.pairwise_energy_extractor()
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

print('Pairwise CUDA')
try:
    model = models.pairwise_energy_extractor(device)
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')
    
print('Pairwise CC')
try:
    model = models.pairwise_energy_CC_extractor()
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')

print('Pairwise CC CUDA')
try:
    model = models.pairwise_energy_CC_extractor(device)
    print(model.compute_from_ase(atoms))
    print('Success')
except:
    print('ERROR')