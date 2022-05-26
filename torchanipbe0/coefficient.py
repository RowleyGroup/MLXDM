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
                                [2.5053229253887226, 22.462900764773437, 17.05021785931179, 12.57621125300233],
                                [0.20894301395856424, 1.4590016624541282, 1.8738488642622464, 1.224105161636984],
                                ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)
                                
def c8_coefficient_model(device = None):
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = models.ANIPBE0()
    path = os.path.join(torchani_dir, 'resources/dispersion/c8/')
    return CoefficientExtractor(path, [[384, 160, 128, 96],
                                       [384, 144, 112, 96],
                                       [384, 128, 112, 96],
                                       [384, 128, 112, 96]],
                                [60.48611719044784, 858.2971024155167, 498.17411788889984, 287.71613578539626],
                                [6.0737109984570985, 62.200168506367056, 60.73042587155412, 33.13683091496654],
                                ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)
                                
def c10_coefficient_model(device = None):
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = models.ANIPBE0()
    path = os.path.join(torchani_dir, 'resources/dispersion/c10/')
    return CoefficientExtractor(path, [[384, 160, 128, 96],
                                       [384, 144, 112, 96],
                                       [384, 128, 112, 96],
                                       [384, 128, 112, 96]],
                                [2018.003824706439, 35704.907820392655, 15418.428201717945, 6977.32928015642],
                                [257.49041066441407, 3266.5763911897166, 2009.8906257122967, 863.8261589061798],
                                ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)
    
