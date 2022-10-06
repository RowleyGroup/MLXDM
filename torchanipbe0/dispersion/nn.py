'''
The neural network dispersion unit
Author: Tu Nguyen Thien Phuc
Last update: 2021-09-30
'''

import os
import torch
from torch import nn
from .utils import CAtomic, BOHR_TO_ANSTROM, cutoff_function, neighbor_list
from ..ase import Calculator as TorchANICalculator
from ..nn import SpeciesEnergies, ANIModel
from typing import Tuple, Optional
from torch import Tensor


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
        x = x.sum(dim = 1)
        x = x * BOHR_TO_ANSTROM ** self.n # Unit conversion for the distance
        return x

class vanderWaalsLayer(nn.Module):
    '''
    The van der Waals layers
    Calculate the van der Waals of the interaction for damp function
    Taking c6, c8 and c10 in
    Shape: [n_batch, n_interaction]
    Out:   [n_batch, n_interaction]
    '''
    def __init__(self, b0, b1):
        # y = b0 + b1 * x
        super().__init__()
        self.b0 = b0
        self.b1 = b1

    def forward(self, c6, c8, c10):
        r_critical = ((c8/c6)**0.5 + (c10/c6)**0.25 + (c10/c8)**0.5) / 3
        # Default unit for b0 is Angstrom, where b1 is unitless
        return self.b0 + self.b1 * r_critical * BOHR_TO_ANSTROM

class DistanceLayer(nn.Module):
    '''
    Convert the coordinate to distance
    Work with non-periodic system
    Input:   x    : [n_batch, n_atom, 3]
    Output:  distance : [n_batch, n_interaction]
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n_atom = x.shape[1]
        indices = torch.triu_indices(n_atom, n_atom, 1)
        return (x[:,indices[1,:],:] - x[:,indices[0,:],:]).norm(2,-1)

class CoefficientConvert(nn.Module):
    '''
    Convert atomic coefficient to interaction interaction
    Work with both non-periodic system (without index) and periodic system
    x        :  [n_batch, n_atom]
    indices  :  [2, n_interaction]
    Output   :  [n_batch, n_interaction]
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x, indices = None):
        n_atom = x.shape[1]
        if indices == None:
            # Non-periodic system
            ind = torch.triu_indices(n_atom, n_atom, 1)
            return (x[:,ind[0,:]] * x[:,ind[1,:]])**0.5
        else:
            # Periodic system
            return (x[:,indices[0,:]] * x[:,indices[1,:]])**0.5


class DistanceNeighborList(nn.Module):
    '''
    Distance layer with neighborlist method
    Read util.py for more information
    x        : [n_batch, n_atom, 3]
    cell     : [3, 3]
    pbc      : [3]
    output 1 : [n_batch, n_interaction]
    output 2 : [2, n_interaction]
    '''
    def __init__(self, cut_off):
        super().__init__()
        self.cut_off = cut_off

    def forward(self, x, cell, pbc):
        return neighbor_list(x, cell, pbc, self.cut_off)


class DispersionModel(ANIModel):
    # Change the ANIModel to make it don't sum up at the end of forward
    def forward(self, species_aev: Tuple[Tensor, Tensor],  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]

        atomic_energies = self._atomic_energies((species, aev))
        return CAtomic(species, atomic_energies)

class ConstantLayer(nn.Module):
    '''
    Input: [n_batch, n_atom, aev_dim]
    Output:[n_batch, n_atom]
    '''
    def __init__(self, param, dtype = None, device = None):
        super().__init__()
        self.param = torch.tensor(param, dtype = dtype, device = device)
    def forward(self, x):
        return self.param.repeat(x.size()[0], x.size()[1])

class CoefficientLayer(nn.Module):
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
            return result

    def __init__(self, b0_list, b1_list, dtype = None, device = None):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.shifter = self.Shifter(b0_list, b1_list)
        self.neural_networks = None

    @staticmethod
    def _create_model(aev_dim):
        '''
        Create the model with various architecture for each type
        '''
        H_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1)
        )

        C_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 144),
            torch.nn.CELU(0.1),
            torch.nn.Linear(144, 112),
            torch.nn.CELU(0.1),
            torch.nn.Linear(112, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1)
        )

        N_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 112),
            torch.nn.CELU(0.1),
            torch.nn.Linear(112, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1)
        )

        O_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 112),
            torch.nn.CELU(0.1),
            torch.nn.Linear(112, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1)
        )
        return DispersionModel([H_network, C_network, N_network, O_network])

    @staticmethod
    def _create_model_2(dimensions):
        '''
        Create a model with various architecture for different atomic type
        Taking the user-defined dimensions
        dimensions:   [4, 4]
        '''
        H_network = torch.nn.Sequential(
            torch.nn.Linear(dimensions[0][0], dimensions[0][1]),
            torch.nn.CELU(0.1),
            torch.nn.Linear(dimensions[0][1], dimensions[0][2]),
            torch.nn.CELU(0.1),
            torch.nn.Linear(dimensions[0][2], dimensions[0][3]),
            torch.nn.CELU(0.1),
            torch.nn.Linear(dimensions[0][3], 1)
        )

        C_network = torch.nn.Sequential(
            torch.nn.Linear(dimensions[1][0], dimensions[1][1]),
            torch.nn.CELU(0.1),
            torch.nn.Linear(dimensions[1][1], dimensions[1][2]),
            torch.nn.CELU(0.1),
            torch.nn.Linear(dimensions[1][2], dimensions[1][3]),
            torch.nn.CELU(0.1),
            torch.nn.Linear(dimensions[1][3], 1)
        )

        N_network = torch.nn.Sequential(
            torch.nn.Linear(dimensions[2][0], dimensions[2][1]),
            torch.nn.CELU(0.1),
            torch.nn.Linear(dimensions[2][1], dimensions[2][2]),
            torch.nn.CELU(0.1),
            torch.nn.Linear(dimensions[2][2], dimensions[2][3]),
            torch.nn.CELU(0.1),
            torch.nn.Linear(dimensions[2][3], 1)
        )

        O_network = torch.nn.Sequential(
            torch.nn.Linear(dimensions[3][0], dimensions[3][1]),
            torch.nn.CELU(0.1),
            torch.nn.Linear(dimensions[3][1], dimensions[3][2]),
            torch.nn.CELU(0.1),
            torch.nn.Linear(dimensions[3][2], dimensions[3][3]),
            torch.nn.CELU(0.1),
            torch.nn.Linear(dimensions[3][3], 1)
        )
        return DispersionModel([H_network, C_network, N_network, O_network])

    def _from_file(self, path, aev_dim):
        '''
        Create the network and load it from "best.pt" file in path
        '''
        self.neural_networks = CoefficientLayer._create_model(aev_dim)
        checkpoint = torch.load(os.path.join(path,'best.pt'))
        self.neural_networks.load_state_dict(checkpoint)

    def _from_file_2(self, path, dimensions):
        '''
        Create the network with self-defined dimension, and load it from "best.pt" file in path
        '''
        self.neural_networks = CoefficientLayer._create_model_2(dimensions)
        checkpoint = torch.load(os.path.join(path,'best.pt'), map_location=torch.device('cpu'))
        self.neural_networks.load_state_dict(checkpoint)

    def _from_test(self, data):
        '''
        The model that return constant dispersion coefficients
        '''
        self.neural_networks = DispersionModel([ConstantLayer(data[0], self.dtype, self.device),
                                                ConstantLayer(data[1], self.dtype, self.device),
                                                ConstantLayer(data[2], self.dtype, self.device),
                                                ConstantLayer(data[3], self.dtype, self.device),
                                                ConstantLayer(0.0, self.dtype, self.device),
                                                ConstantLayer(0.0, self.dtype, self.device),
                                                ConstantLayer(0.0, self.dtype, self.device)])

    def forward(self, species_aev):
        x = self.neural_networks(species_aev)
        x = self.shifter(x)
        return x


class DispersionLayer(nn.Module):
    '''
    MAIN LAYER OF DISPERSION
    Given the aev_computer from the ANI model, all of the coefficients network,
                b0, b1 for vdw calculation, and the device necessary
    Take in the species_coordinate from the torchani
    Shape: [n_batch, n_atom]
    Out : [n_batch] of the energy in SpeciesEnergy cover
    '''
    def __init__(self, aev_computer, c6_net, c8_net, c10_net, a0_vdw, a1_vdw,
                 cutoff, dtype, device, species = None):
        super().__init__()
        self.cutoff = cutoff
        self.dtype = dtype
        self.device = device
        self.aev_computer = aev_computer
        self.c6_net = c6_net
        self.c8_net = c8_net
        self.c10_net = c10_net
        self.coef_convert = CoefficientConvert()
        self.distance_layer = DistanceLayer()
        self.vdw_layer = vanderWaalsLayer(a1_vdw, a0_vdw) # The order in the code is reversed
        self.c6_layer = EnergyLayer(6, cutoff)
        self.c8_layer = EnergyLayer(8, cutoff)
        self.c10_layer = EnergyLayer(10, cutoff)
        self.distance_layer_neighbor = DistanceNeighborList(cutoff)
        self.species = species

    def forward(self, species_coordinates, cell = None, pbc = None):
        species_aev = self.aev_computer(species_coordinates, cell, pbc)
        c6 = self.c6_net(species_aev)
        c8 = self.c8_net(species_aev)
        c10 = self.c10_net(species_aev)
        if cell == None or pbc == None:
            # Non-periodic system
            distance = self.distance_layer(species_coordinates[1])
            c6_pair = self.coef_convert(c6)
            c8_pair = self.coef_convert(c8)
            c10_pair = self.coef_convert(c10)
        else:
            # Periodic system
            assert cell != None and pbc != None
            distance, index = self.distance_layer_neighbor(species_coordinates[1], cell, pbc)
            c6_pair = self.coef_convert(c6, index)
            c8_pair = self.coef_convert(c8, index)
            c10_pair = self.coef_convert(c10, index)
        rvdw = self.vdw_layer(c6_pair,c8_pair,c10_pair)
        c6_energy = self.c6_layer(distance, c6_pair, rvdw)
        c8_energy = self.c8_layer(distance, c8_pair, rvdw)
        c10_energy = self.c10_layer(distance, c10_pair, rvdw)
        return SpeciesEnergies(species_aev[0], c6_energy + c8_energy + c10_energy)

    def ase(self, species = None, **kwargs):
        # Need the species from the ANI model
        if species == None:
            return TorchANICalculator(self.species, self, **kwargs)
        else:
            return TorchANICalculator(species, self, **kwargs)

# New update for correct combination rules

class PolarizabilityLayer(nn.Module):
    def __init__(self, polar_free, volume_free):
        super().__init__()
        assert len(polar_free) == len(volume_free)
        self.polar_free = polar_free
        self.volume_free = volume_free

    def forward(self, species_volume):
        species = species_volume[0]
        volume = species_volume[1]
        result = torch.zeros_like(volume)
        n = len(self.polar_free)
        for i in range(n):
            mask = (species==i)
            result[mask] = volume[mask] * self.polar_free[i] / self.volume_free[i]
        return result

class C6Layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, m, polar, indices):
        # Change here
        ind1 = indices[0,:]
        ind2 = indices[1,:]
        m1 = m[:,ind1]
        m2 = m[:,ind2]
        polar1 = polar[:,ind1]
        polar2 = polar[:,ind2]
        return m1 * m2 / (m1 / polar1 + m2 / polar2)

class C8Layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, m1, m2, polar, indices):
        ind1 = indices[0,:]
        ind2 = indices[1,:]
        m1_1 = m1[:,ind1]
        m1_2 = m1[:,ind2]
        m2_1 = m2[:,ind1]
        m2_2 = m2[:,ind2]
        polar1 = polar[:,ind1]
        polar2 = polar[:,ind2]
        return 1.5 * (m1_1*m2_2 + m1_2*m2_1) / (m1_1/polar1 + m1_2/polar2)

class C10Layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, m1, m2, m3, polar, indices):
        ind1 = indices[0,:]
        ind2 = indices[1,:]
        m1_1 = m1[:,ind1]
        m1_2 = m1[:,ind2]
        m2_1 = m2[:,ind1]
        m2_2 = m2[:,ind2]
        m3_1 = m3[:,ind1]
        m3_2 = m3[:,ind2]
        polar1 = polar[:,ind1]
        polar2 = polar[:,ind2]
        return 2*(m1_1*m3_2 + m3_1*m1_2 + 2.1*m2_1*m2_2)/(m1_1/polar1 + m1_2/polar2)

# Use CoefficientLayer to cover the m1, m2, m3 and v net

class DispersionLayer2(nn.Module):
    '''
    The second generation Dispersion module
    Not using the combination rules, but original relation from eXchange-hole Dispersion Model
    Shape: [n_batch, n_atom]
    Output: [n_batch]
    '''
    def __init__(self, aev_computer, m1_net, m2_net, m3_net, v_net, a0_vdw, a1_vdw,
                 polar_list, volume_list, cutoff, dtype, device, species = None):
        # m1 net, m2 net, m3 net and v net should be used the CoefficientLayer to
        # ensure the flow of the code
        super().__init__()
        self.cutoff = cutoff
        self.dtype = dtype
        self.device = device
        self.aev_computer = aev_computer
        self.m1_net = m1_net
        self.m2_net = m2_net
        self.m3_net = m3_net
        self.v_net = v_net
        # self.coef_convert = CoefficientConvert()
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
        self.species = species

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
        return SpeciesEnergies(species_aev[0], c6_energy + c8_energy + c10_energy)

    def ase(self, species=None, **kwargs):
        if species is None:
            return TorchANICalculator(self.species, self, **kwargs)
        else:
            return TorchANICalculator(species, self, **kwargs)

# Get the energy for separate split
class Dispersion_C6(DispersionLayer):
    def forward(self, species_coordinates, cell = None, pbc = None):
        species_aev = self.aev_computer(species_coordinates, cell, pbc)
        c6 = self.c6_net(species_aev)
        c8 = self.c8_net(species_aev)
        c10 = self.c10_net(species_aev)
        if cell == None or pbc == None:
            # Non-periodic system
            distance = self.distance_layer(species_coordinates[1])
            c6_pair = self.coef_convert(c6)
            c8_pair = self.coef_convert(c8)
            c10_pair = self.coef_convert(c10)
        else:
            # Periodic system
            assert cell != None and pbc != None
            distance, index = self.distance_layer_neightbor(species_coordinates[1], cell, pbc)
            c6_pair = self.coef_convert(c6, index)
            c8_pair = self.coef_convert(c8, index)
            c10_pair = self.coef_convert(c10, index)
        rvdw = self.vdw_layer(c6_pair,c8_pair,c10_pair)
        c6_energy = self.c6_layer(distance, c6_pair, rvdw)
        return SpeciesEnergies(species_aev[0], c6_energy)

class Dispersion_C8(DispersionLayer):
    def forward(self, species_coordinates, cell = None, pbc = None):
        species_aev = self.aev_computer(species_coordinates, cell, pbc)
        c6 = self.c6_net(species_aev)
        c8 = self.c8_net(species_aev)
        c10 = self.c10_net(species_aev)
        if cell == None or pbc == None:
            # Non-periodic system
            distance = self.distance_layer(species_coordinates[1])
            c6_pair = self.coef_convert(c6)
            c8_pair = self.coef_convert(c8)
            c10_pair = self.coef_convert(c10)
        else:
            # Periodic system
            assert cell != None and pbc != None
            distance, index = self.distance_layer_neightbor(species_coordinates[1], cell, pbc)
            c6_pair = self.coef_convert(c6, index)
            c8_pair = self.coef_convert(c8, index)
            c10_pair = self.coef_convert(c10, index)
        rvdw = self.vdw_layer(c6_pair,c8_pair,c10_pair)
        c8_energy = self.c8_layer(distance, c8_pair, rvdw)
        return SpeciesEnergies(species_aev[0], c8_energy)

class Dispersion_C10(DispersionLayer):
    def forward(self, species_coordinates, cell = None, pbc = None):
        species_aev = self.aev_computer(species_coordinates, cell, pbc)
        c6 = self.c6_net(species_aev)
        c8 = self.c8_net(species_aev)
        c10 = self.c10_net(species_aev)
        if cell == None or pbc == None:
            # Non-periodic system
            distance = self.distance_layer(species_coordinates[1])
            c6_pair = self.coef_convert(c6)
            c8_pair = self.coef_convert(c8)
            c10_pair = self.coef_convert(c10)
        else:
            # Periodic system
            assert cell != None and pbc != None
            distance, index = self.distance_layer_neightbor(species_coordinates[1], cell, pbc)
            c6_pair = self.coef_convert(c6, index)
            c8_pair = self.coef_convert(c8, index)
            c10_pair = self.coef_convert(c10, index)
        rvdw = self.vdw_layer(c6_pair,c8_pair,c10_pair)
        c10_energy = self.c10_layer(distance, c10_pair, rvdw)
        return SpeciesEnergies(species_aev[0], c10_energy)

class Dispersion_C6_2(DispersionLayer2):
    def forward(self, species_coordinates, cell = None, pbc = None):
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
        # c8_energy = self.c8_e_layer(distance, c8_pair, rvdw)
        # c10_energy = self.c10_e_layer(distance, c10_pair, rvdw)
        return SpeciesEnergies(species_aev[0], c6_energy)

class Dispersion_C8_2(DispersionLayer2):
    def forward(self, species_coordinates, cell = None, pbc = None):
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
        # c6_energy = self.c6_e_layer(distance, c6_pair, rvdw)
        c8_energy = self.c8_e_layer(distance, c8_pair, rvdw)
        # c10_energy = self.c10_e_layer(distance, c10_pair, rvdw)
        return SpeciesEnergies(species_aev[0], c8_energy)

class Dispersion_C10_2(DispersionLayer2):
    def forward(self, species_coordinates, cell = None, pbc = None):
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
        # c6_energy = self.c6_e_layer(distance, c6_pair, rvdw)
        # c8_energy = self.c8_e_layer(distance, c8_pair, rvdw)
        c10_energy = self.c10_e_layer(distance, c10_pair, rvdw)
        return SpeciesEnergies(species_aev[0], c10_energy)


class ANIDispersion(nn.Module):
    '''
    Combine two potential for the final result
    Take in the species_coordinate from the torchani
    Shape: [n_batch, n_atom]
    Out : [n_batch] of the energy
    Giving the ase to return torchani style calculator
    '''
    def __init__(self, ani_model, disp_model):
        super().__init__()
        self.ani_model = ani_model
        self.disp_model = disp_model
    def forward(self, x, cell = None, pbc = None):
        species_energy = self.ani_model(x, cell, pbc)
        energy = species_energy[1] + self.disp_model(x, cell, pbc)[1]
        return SpeciesEnergies(species_energy[0], energy)
    def ase(self, **kwargs):
        return TorchANICalculator(self.ani_model.species, self, **kwargs)
