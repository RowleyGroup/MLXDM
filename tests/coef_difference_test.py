import os
import torch
from torch import nn
import torchanipbe0
from torchanipbe0 import models
from torchanipbe0.dispersion.nn import DispersionModel, CoefficientConvert, DistanceLayer, \
    vanderWaalsLayer, DistanceNeighborList, cutoff_function, BOHR_TO_ANSTROM
import pandas as pd
import numpy as np


# Load the model

class Shifter(nn.Module):
    '''
    The class to shift the data according to linear relation
    y = b0 + b1 * x
    '''
    def __init__(self, b0_list, b1_list):
        super().__init__()
        self.b0_list = b0_list
        self.b1_list = b1_list
        assert len(b0_list) == len(b1_list)

    def forward(self, x):
        species = x[0]
        coef = x[1]
        n = len(self.b0_list)
        result = torch.zeros(coef.size(), dtype = coef.dtype, device = coef.device)
        for i in range(n):
            mask = (species == i)
            result[mask] = self.b0_list[i] + self.b1_list[i] * coef[mask]
        return (species, result)

def create_model():
    '''
    Create the model with various architecture for each type
    '''
    H_network = torch.nn.Sequential(
        torch.nn.Linear(384, 160),
        torch.nn.CELU(0.1),
        torch.nn.Linear(160, 128),
        torch.nn.CELU(0.1),
        torch.nn.Linear(128, 96),
        torch.nn.CELU(0.1),
        torch.nn.Linear(96, 1)
    )

    C_network = torch.nn.Sequential(
        torch.nn.Linear(384, 144),
        torch.nn.CELU(0.1),
        torch.nn.Linear(144, 112),
        torch.nn.CELU(0.1),
        torch.nn.Linear(112, 96),
        torch.nn.CELU(0.1),
        torch.nn.Linear(96, 1)
    )

    N_network = torch.nn.Sequential(
        torch.nn.Linear(384, 128),
        torch.nn.CELU(0.1),
        torch.nn.Linear(128, 112),
        torch.nn.CELU(0.1),
        torch.nn.Linear(112, 96),
        torch.nn.CELU(0.1),
        torch.nn.Linear(96, 1)
    )

    O_network = torch.nn.Sequential(
        torch.nn.Linear(384, 128),
        torch.nn.CELU(0.1),
        torch.nn.Linear(128, 112),
        torch.nn.CELU(0.1),
        torch.nn.Linear(112, 96),
        torch.nn.CELU(0.1),
        torch.nn.Linear(96, 1)
    )
    return DispersionModel([H_network, C_network, N_network, O_network])

def from_test(data, dtype=None, device=None):
    '''
    The model that return constant dispersion coefficients
    '''
    class SampleLayer(nn.Module):
        '''
        Input: [n_batch, n_atom, aev_dim]
        Output:[n_batch, n_atom]
        '''
        def __init__(self, param, dtype = None, device = None):
            super().__init__()
            self.param = torch.tensor(param, dtype = dtype, device = device)
        def forward(self, x):
            return self.param.repeat(x.size()[0], x.size()[1])

    return DispersionModel([SampleLayer(data[0], dtype, device),
                            SampleLayer(data[1], dtype, device),
                            SampleLayer(data[2], dtype, device),
                            SampleLayer(data[3], dtype, device),
                            SampleLayer(0.0, dtype, device),
                            SampleLayer(0.0, dtype, device),
                            SampleLayer(0.0, dtype, device)])

def from_file_2(path):
    '''
    Create the network with self-defined dimension, and load it from "best.pt" file in path
    '''
    neural_networks = create_model()
    checkpoint = torch.load(os.path.join(path,'best.pt'), map_location=torch.device('cpu'))
    neural_networks.load_state_dict(checkpoint)
    return neural_networks

def feed_forward(species_aev, neural_networks, shifter=None):
    x = neural_networks(species_aev)
    if shifter is not None:
        x = shifter(x)
    return x

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
        # x = x.sum(dim = 1)
        x = x * BOHR_TO_ANSTROM ** self.n # Unit conversion for the distance
        return x

def forward(species_coordinates, cell, pbc, aev_computer, c6_net, c8_net, c10_net, a0_vdw, a1_vdw,
            cutoff, shift6=None, shift8=None, shift10=None):
    coef_convert = CoefficientConvert()
    distance_layer = DistanceLayer()
    vdw_layer = vanderWaalsLayer(a1_vdw, a0_vdw)
    c6_layer = EnergyLayer(6, cutoff)
    c8_layer = EnergyLayer(8, cutoff)
    c10_layer = EnergyLayer(10, cutoff)
    distance_layer_neighbor = DistanceNeighborList(cutoff)
    # Real calculation
    species_aev = aev_computer(species_coordinates, cell, pbc)
    c6 = feed_forward(species_aev, c6_net, shift6)
    c8 = feed_forward(species_aev, c8_net, shift8)
    c10 = feed_forward(species_aev, c10_net, shift10)
    if cell == None or pbc == None:
        # Non-periodic system
        distance = distance_layer(species_coordinates[1])
        c6_pair = coef_convert(c6[1])
        c8_pair = coef_convert(c8[1])
        c10_pair = coef_convert(c10[1])
    else:
        # Periodic system
        assert cell != None and pbc != None
        distance, index = distance_layer_neighbor(species_coordinates[1], cell, pbc)
        c6_pair = coef_convert(c6[1], index)
        c8_pair = coef_convert(c8[1], index)
        c10_pair = coef_convert(c10[1], index)
    rvdw = vdw_layer(c6_pair,c8_pair,c10_pair)
    c6_energy = c6_layer(distance, c6_pair, rvdw)
    c8_energy = c8_layer(distance, c8_pair, rvdw)
    c10_energy = c10_layer(distance, c10_pair, rvdw)
    energy = (c6_energy + c8_energy + c10_energy).sum(axis=1)
    return (c6[1].detach().cpu().numpy(), c8[1].detach().cpu().numpy(), c10[1].detach().cpu().numpy(),
            c6_pair.detach().cpu().numpy(), c8_pair.detach().cpu().numpy(), c10_pair.detach().cpu().numpy(),
            rvdw.detach().cpu().numpy(), c6_energy.detach().cpu().numpy(), c8_energy.detach().cpu().numpy(), 
            c10_energy.detach().cpu().numpy(), energy.detach().cpu().numpy())

# case one
aev_computer = models.ANIPBE0().aev_computer
species_to_tensor = torchanipbe0.utils.ChemicalSymbolsToInts(['H','C','N','O'])
path = '/home/zeldery/Documents/GitHub/MLXDM/torchanipbe0/resources/dispersion/' # Change this to your local file


c6_net = from_file_2(f'{path}c6/')
c8_net = from_file_2(f'{path}c8/')
c10_net = from_file_2(f'{path}c10/')

shift_6 = Shifter([2.5053229253887226, 22.462900764773437, 17.05021785931179, 12.57621125300233],
                  [0.20894301395856424, 1.4590016624541282, 1.8738488642622464, 1.224105161636984])
shift_8 = Shifter([60.48611719044784, 858.2971024155167, 498.17411788889984, 287.71613578539626],
                  [6.0737109984570985, 62.200168506367056, 60.73042587155412, 33.13683091496654])
shift_10 = Shifter([2018.003824706439, 35704.907820392655, 15418.428201717945, 6977.32928015642],
                   [257.49041066441407, 3266.5763911897166, 2009.8906257122967, 863.8261589061798])

c6_const = from_test([2.5053229253887226, 22.462900764773437, 17.05021785931179, 12.57621125300233])
c8_const = from_test([60.48611719044784, 858.2971024155167, 498.17411788889984, 287.71613578539626])
c10_const = from_test([2018.003824706439, 35704.907820392655, 15418.428201717945, 6977.32928015642])



# Load the molecular system
data = pd.read_csv('deshaw_370K.csv')
index = 359419 # Choose the index here
coord = list(map(float, data.loc[index, 'xyz'].split()))
coord = torch.tensor(np.array(coord).reshape((-1,3)), dtype=torch.float32)
element = data.loc[index, 'elements'].split()
element = np.array(element)
species = species_to_tensor(element)
species = species.unsqueeze(0)
coord = coord.unsqueeze(0)
#print(species)
#print(coord)
a = forward((species, coord), None, None, aev_computer, c6_net, c8_net,\
    c10_net, 0.4186, 2.6791, 14.0, shift_6, shift_8, shift_10)
b = forward((species, coord), None, None, aev_computer, c6_const, c8_const,\
    c10_const, 0.4186, 2.6791, 14.0)

# Print data

print('Atomic quantity: C6 C6_const C8 C8_const C10 C10_const')
n = a[0].shape[1]
for i in range(n):
    print(f'{a[0][0,i]:9.2f} {b[0][0,i]:9.2f} {a[1][0,i]:9.2f} {b[1][0,i]:9.2f} {a[2][0,i]:9.2f} {b[2][0,i]:9.2f} ')

print('Pair data c6 c6_const c8 c8_const c10 c10_const rvdw rvdw_const')
n = a[3].shape[1]
for i in range(n):
    print(f'{a[3][0,i]:9.2f} {b[3][0,i]:9.2f} {a[4][0,i]:9.2f} {b[4][0,i]:9.2f} {a[5][0,i]:9.2f} {b[5][0,i]:9.2f} {a[6][0,i]:9.4f} {b[6][0,i]:9.4f} ')

print('Energy c6 c6_const c8 c8_const c10 c10_const')
n = a[7].shape[1]
for i in range(n):
    print(f'{a[7][0,i]:9.8f} {b[7][0,i]:9.8f} {a[8][0,i]:9.8f} {b[8][0,i]:9.8f} {a[9][0,i]:9.8f} {b[9][0,i]:9.8f} ')
print('Total energy')
print(f'{a[10][0]:9.5f} {b[10][0]:9.5f}')

