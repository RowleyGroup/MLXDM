# DISPERSION COEFFICIENTS EXTRACTOR
# Author: Tu Nguyen Thien Phuc
# Date: 2022-05-04

import torch
import os
from torch import nn
from .dispersion import CoefficientLayer
from .dispersion import C6Layer, C8Layer, C10Layer, DistanceLayer, PolarizabilityLayer
from .dispersion import vanderWaalsLayer, DistanceNeighborList
from .dispersion import BOHR_TO_ANSTROM, cutoff_function
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
    
    def forward(self, species_coordinates, cell=None, pbc=None):
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

# For energy extractor, need to change the energy layer to return the individual energy
class EnergyLayer(nn.Module):
    '''
    The dispersion energy layer
    Taking the coefficient and vanderWaals radius to produce the energy
    distance: [n_batch, n_interaction]
    coef    : [n_batch, n_interaction]
    rvdw    : [n_batch, n_interaction]
    out     : [n_batch]
    '''
    def __init__(self, n, cutoff = None):
        super().__init__()
        self.n = n
        self.cutoff = cutoff

    def forward(self, distance, coef, rvdw):
        x = -coef  / (distance**self.n + rvdw**self.n)
        if self.cutoff != None:
            x = x * cutoff_function(distance, self.cutoff)
        # x = x.sum(dim = 1) # Don't sum up at the end for the purpose of extraction
        x = x * BOHR_TO_ANSTROM ** self.n # Unit conversion for the distance
        return x

class EnergyExtractor(nn.Module):
    '''
    '''
    def __init__(self, aev_computer, m1_net, m2_net, m3_net, v_net, a0_vdw, a1_vdw,
                 polar_list, volume_list, cutoff, species_to_tensor, dtype, device):
        # m1 net, m2 net, m3 net and v net should be used the CoefficientLayer to
        # ensure the flow of the code
        super().__init__()
        self.cutoff = cutoff
        self.dtype = dtype
        self.device = device
        self.aev_computer = aev_computer
        self.m1_net = m1_net.to(device)
        self.m2_net = m2_net.to(device)
        self.m3_net = m3_net.to(device)
        self.v_net = v_net.to(device)
        self.c6_layer = C6Layer()
        self.c8_layer = C8Layer()
        self.c10_layer = C10Layer()
        self.distance_layer = DistanceLayer()
        self.polar_layer = PolarizabilityLayer(polar_list, volume_list)
        self.vdw_layer = vanderWaalsLayer(a1_vdw, a0_vdw) # The order in the code is reversed
        self.c6_e_layer = EnergyLayer(6, cutoff)
        self.c8_e_layer = EnergyLayer(8, cutoff)
        self.c10_e_layer = EnergyLayer(10, cutoff)
        self.distance_layer_neighbor = DistanceNeighborList(cutoff)
        self.species_to_tensor = species_to_tensor

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
        return [result.squeeze().detach().cpu().numpy() for result in output]

    def forward(self, species_coordinates, cell=None, pbc=None):
        species_aev = self.aev_computer(species_coordinates, cell, pbc)
        m1 = self.m1_net(species_aev)
        m2 = self.m2_net(species_aev)
        m3 = self.m3_net(species_aev)
        v = self.v_net(species_aev)
        polar = self.polar_layer((species_aev[0], v))
        if cell is None or pbc is None:
            n_atom = m1.shape[1]
            distance = self.distance_layer(species_coordinates[1])
            index = torch.triu_indices(n_atom, n_atom, 1)
        else:
            distance, index = self.distance_layer_neighbor(species_coordinates[1], cell, pbc)
        c6_pair = self.c6_layer(m1, polar, index)
        c8_pair = self.c8_layer(m1, m2, polar, index)
        c10_pair = self.c10_layer(m1, m2, m3, polar, index)
        rvdw = self.vdw_layer(c6_pair,c8_pair,c10_pair)
        c6_energy = self.c6_e_layer(distance, c6_pair, rvdw)
        c8_energy = self.c8_e_layer(distance, c8_pair, rvdw)
        c10_energy = self.c10_e_layer(distance, c10_pair, rvdw)
        return index, distance, c6_energy, c8_energy, c10_energy



def c6_coefficient_model(device=None):
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = models.ANIPBE0()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion/c6/')
    return CoefficientExtractor(path, [[384, 160, 128, 96],
                                       [384, 144, 112, 96],
                                       [384, 128, 112, 96],
                                       [384, 128, 112, 96]],
                                [2.5053229253887226, 22.462900764773437, 17.05021785931179, 12.57621125300233],
                                [0.20894301395856424, 1.4590016624541282, 1.8738488642622464, 1.224105161636984],
                                ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)
                                
def c8_coefficient_model(device=None):
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = models.ANIPBE0()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion/c8/')
    return CoefficientExtractor(path, [[384, 160, 128, 96],
                                       [384, 144, 112, 96],
                                       [384, 128, 112, 96],
                                       [384, 128, 112, 96]],
                                [60.48611719044784, 858.2971024155167, 498.17411788889984, 287.71613578539626],
                                [6.0737109984570985, 62.200168506367056, 60.73042587155412, 33.13683091496654],
                                ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)
                                
def c10_coefficient_model(device=None):
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = models.ANIPBE0()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion/c10/')
    return CoefficientExtractor(path, [[384, 160, 128, 96],
                                       [384, 144, 112, 96],
                                       [384, 128, 112, 96],
                                       [384, 128, 112, 96]],
                                [2018.003824706439, 35704.907820392655, 15418.428201717945, 6977.32928015642],
                                [257.49041066441407, 3266.5763911897166, 2009.8906257122967, 863.8261589061798],
                                ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)
    
def m1_model(device=None):
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = models.ANIPBE0()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion/m1/')
    return CoefficientExtractor(path, [[384, 160, 128, 96],
                                       [384, 144, 112, 96],
                                       [384, 128, 112, 96],
                                       [384, 128, 112, 96]], 
                                [1.5487023355096374, 4.33832763491479, 4.689239793170584, 4.835634998519475],
                                [0.12092114874978759, 0.4171882700357045, 0.5586210781338053, 0.4070627085041909],
                                ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)

def m2_model(device=None):
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = models.ANIPBE0()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion/m2/')
    return CoefficientExtractor(path, [[384, 160, 128, 96],
                                       [384, 144, 112, 96],
                                       [384, 128, 112, 96],
                                       [384, 128, 112, 96]], 
                                [12.5683581996112, 55.10387928710763, 45.607226558095036, 37.35790106431391],
                                [1.1739220921349471, 5.030540928979481, 5.739017414160557, 3.827982573531133],
                                ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)

def m3_model(device=None):
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = models.ANIPBE0()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion/m3/')
    return CoefficientExtractor(path, [[384, 160, 128, 96],
                                       [384, 144, 112, 96],
                                       [384, 128, 112, 96],
                                       [384, 128, 112, 96]], 
                                [204.21782032761914, 971.7168397173592, 592.7362211972969, 372.60697888431537],
                                [17.42101897973452, 75.12440414708934, 58.82559008729809, 34.5463306934543],
                                ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)

def v_model(device=None):
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = models.ANIPBE0()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion/v/')
    return CoefficientExtractor(path, [[384, 160, 128, 96],
                                       [384, 144, 112, 96],
                                       [384, 128, 112, 96],
                                       [384, 128, 112, 96]], 
                                [6.027731295335626, 31.54828240955614, 26.242213297618285, 21.793723234843963],
                                [0.38563987151949186, 1.0527722226285448, 1.0606388518419783, 0.8187177116508805],
                                ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)

def pairwise_energy_extractor(device=None):
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = models.ANIPBE0()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion/')
    m1_net = CoefficientLayer([1.5487023355096374, 4.33832763491479, 4.689239793170584, 4.835634998519475],
                              [0.12092114874978759, 0.4171882700357045, 0.5586210781338053, 0.4070627085041909])
    m2_net = CoefficientLayer([12.5683581996112, 55.10387928710763, 45.607226558095036, 37.35790106431391],
                              [1.1739220921349471, 5.030540928979481, 5.739017414160557, 3.827982573531133])
    m3_net = CoefficientLayer([204.21782032761914, 971.7168397173592, 592.7362211972969, 372.60697888431537],
                              [17.42101897973452, 75.12440414708934, 58.82559008729809, 34.5463306934543])
    v_net = CoefficientLayer([6.027731295335626, 31.54828240955614, 26.242213297618285, 21.793723234843963],
                              [0.38563987151949186, 1.0527722226285448, 1.0606388518419783, 0.8187177116508805])
    m1_net._from_file_2(f'{path}m1/', [[384, 160, 128, 96],
                                       [384, 144, 112, 96],
                                       [384, 128, 112, 96],
                                       [384, 128, 112, 96]])
    m2_net._from_file_2(f'{path}m2/', [[384, 160, 128, 96],
                                       [384, 144, 112, 96],
                                       [384, 128, 112, 96],
                                       [384, 128, 112, 96]])
    m3_net._from_file_2(f'{path}m3/', [[384, 160, 128, 96],
                                       [384, 144, 112, 96],
                                       [384, 128, 112, 96],
                                       [384, 128, 112, 96]])
    v_net._from_file_2(f'{path}v/', [[384, 160, 128, 96],
                                       [384, 144, 112, 96],
                                       [384, 128, 112, 96],
                                       [384, 128, 112, 96]])
    return EnergyExtractor(ani_model.aev_computer, m1_net, m2_net, m3_net, v_net,
                            0.4186, 2.6791, [4.4997895, 11.87706886, 7.423168043, 5.412164335], 
                            [8.2794385587, 35.403450375, 26.774856263, 22.577665436], 
                            14.0, ani_model.species_to_tensor, torch.float32, device)

def pairwise_energy_extractor_const(device=None):
    ani_model = models.ANIPBE0()
    ani_model = ani_model.to(device)
    m1_net = CoefficientLayer([0.0, 0.0, 0.0, 0.0],
                              [1.0, 1.0, 1.0, 1.0])
    m2_net = CoefficientLayer([0.0, 0.0, 0.0, 0.0],
                              [1.0, 1.0, 1.0, 1.0])
    m3_net = CoefficientLayer([0.0, 0.0, 0.0, 0.0],
                              [1.0, 1.0, 1.0, 1.0])
    v_net = CoefficientLayer([0.0, 0.0, 0.0, 0.0],
                              [1.0, 1.0, 1.0, 1.0])
    m1_net._from_test([1.5487023355096374, 4.33832763491479, 4.689239793170584, 4.835634998519475])
    m2_net._from_test([12.5683581996112, 55.10387928710763, 45.607226558095036, 37.35790106431391])
    m3_net._from_test([204.21782032761914, 971.7168397173592, 592.7362211972969, 372.60697888431537])
    v_net._from_test([6.027731295335626, 31.54828240955614, 26.242213297618285, 21.793723234843963])
    return EnergyExtractor(ani_model.aev_computer, m1_net, m2_net, m3_net, v_net,
                            0.4186, 2.6791, [4.4997895, 11.87706886, 7.423168043, 5.412164335], 
                            [8.2794385587, 35.403450375, 26.774856263, 22.577665436], 
                            14.0, ani_model.species_to_tensor, torch.float32, device)
