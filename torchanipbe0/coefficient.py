# DISPERSION COEFFICIENTS EXTRACTOR
# Author: Tu Nguyen Thien Phuc
# Date: 2022-05-04

import numpy as np
import torch
import ase
import os
from torch import nn
from .dispersion import CoefficientLayer
from . import utils, models
from pathlib import Path

class CoefficientExtractor(nn.Module):
    '''
    '''
    def __init__(self, info_file_path, dimensions, b0_list, b1_list, 
                 aev_computer, species_to_tensor, dtype, device):
        super().__init__()
        self.model = CoefficientLayer(b0_list, b1_list)
        self.model._from_file_2(info_file_path, dimensions)
        self.model = self.model.to(device)
        self.aev_computer = aev_computer
        self.dtype = dtype
        self.device = device
        self.species_to_tensor = species_to_tensor
    
    def forward(self, species_coordinates, cell = None, pbc = None):
        species_aev = self.aev_computer(species_coordinates, cell, pbc)
        return self.model(species_aev)
        
    def compute_from_ase(self, atoms):
        species = self.species_to_tensor(atoms.get_chemical_symbols()).to(self.device)
        species = species.unsqueeze(0)
        cell = torch.tensor(atoms.get_cell(complete = True),
                            dtype = self.dtype, device = self.device)
        pbc = torch.tensor(atoms.get_pbc(), dtype = torch.bool,
                           device = self.device)
        pbc_enabled = pbc.any().item()
        coordinates = torch.tensor(atoms.get_positions())
        coordinates = coordinates.to(self.device).to(self.dtype).unsqueeze(0)
        if pbc_enabled:
            coordinates = utils.map2central(cell, coordinates, pbc)
        species_coordinates = (species, coordinates)
        output = self(species_coordinates, cell, pbc)
        return output.squeeze().detach().cpu().numpy()
        
def c6_coefficient_model(device = None):
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = models.ANIPBE0()
    path = os.path.join(torchani_dir, 'resources/dispersion/c6/')
    return CoefficientExtractor(path, [[384, 160, 128, 96],
                                       [384, 144, 112, 96],
                                       [384, 128, 112, 96],
                                       [384, 128, 112, 96]],
                                [2.556908763590551, 22.32366797778062, 16.847532407791093, 12.347053484315223],
                                [0.17447624496857395, 1.5532671126558564, 1.814900793085204, 1.2263004291581763],
                                ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)
                                
def c8_coefficient_model(device = None):
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = models.ANIPBE0()
    path = os.path.join(torchani_dir, 'resources/dispersion/c8/')
    return CoefficientExtractor(path, [[384, 160, 128, 96],
                                       [384, 144, 112, 96],
                                       [384, 128, 112, 96],
                                       [384, 128, 112, 96]],
                                [60.11874278372575, 841.0373444172435, 486.7241394964031, 283.6348571895923],
                                [4.963231507675922, 61.53690552359383, 58.36628032450961, 32.365514287773536],
                                ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)
                                
def c10_coefficient_model(device = None):
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = models.ANIPBE0()
    path = os.path.join(torchani_dir, 'resources/dispersion/c10/')
    return CoefficientExtractor(path, [[384, 160, 128, 96],
                                       [384, 144, 112, 96],
                                       [384, 128, 112, 96],
                                       [384, 128, 112, 96]],
                                [1956.8679019547583, 34276.10440769646, 14842.800050465172, 6846.580871919675],
                                [213.31848185100714, 2994.7423466387086, 1934.0812271421078, 842.8678839801049],
                                ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)
    
