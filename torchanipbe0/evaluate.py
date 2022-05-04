'''
Code to express the mean and standard deviation of ensemble simulation
Author: Tu Nguyen Thien Phuc
Last update: 2022-05-04
'''

import torch
import ase
from . import neurochem, utils
from .nn import SpeciesConverter
from .aev import AEVComputer

class EnsembleEvaluater:
    def __init__(self, info_file_path, dtype, device, unit = None):
        const_file, sae_file, ensemble_prefix, ensemble_size = neurochem.parse_neurochem_resources(info_file_path)
        consts = neurochem.Constants(const_file)
        self.species_converter = SpeciesConverter(consts.species).to(device)
        self.aev_computer = AEVComputer(**consts).to(device)
        energy_shifter, sae_dict = neurochem.load_sae(sae_file, return_dict=True)
        self.energy_shifter = energy_shifter.to(device)
        self.species_to_tensor = consts.species_to_tensor
        self.species = consts.species
        aev_dim = self.aev_computer.aev_length
        self.dtype = dtype
        self.device = device
        if unit == 'eV':
            self.convert_factor = ase.units.Hartree
        else:
            self.convert_factor = 1.0
        self.neural_networks = []
        for i in range(ensemble_size):
            network_dir = f'{ensemble_prefix}{i}'
            if info_file_path == 'ani-pbe0_8x.info':
                network_dir = f'{ensemble_prefix}{i}'
                self.neural_networks.append(neurochem.load_model_2(network_dir, aev_dim).to(device))
            else:
                network_dir = f'{ensemble_prefix}{i}/networks'
                self.neural_networks.append(neurochem.load_model(self.species, network_dir).to(device))
    
    def evaluate(self, species_coordinates, cell = None, pbc = None):
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        output = []
        for model in self.neural_networks:
            species_energies = model(species_aevs)
            energy = self.energy_shifter(species_energies).energies
            output.append(energy.detach().cpu().numpy() * self.convert_factor)
        mean = 0.0
        for out in output:
            mean += out
        mean /= len(output)
        sd = 0.0
        for out in output:
            sd += (out - mean) ** 2
        return mean, (sd / len(output) )**0.5
    
    def evaluate_from_ase(self, atoms):
        species = self.species_to_tensor(atoms.get_chemical_symbols()).to(self.device)
        species = species.unsqueeze(0)
        cell = torch.tensor(atoms.get_cell(complete=True),
                            dtype=self.dtype, device=self.device)
        pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool,
                           device=self.device)
        pbc_enabled = pbc.any().item()
        if pbc_enabled:
            coordinates = utils.map2central(cell, coordinates, pbc)
        coordinates = torch.tensor(atoms.get_positions())
        
        coordinates = coordinates.to(self.device).to(self.dtype).unsqueeze(0)
        species_coordinates = (species, coordinates)
        return self.evaluate(species_coordinates, cell, pbc)
        
def ani_pbe0_evaluater(device = torch.device('cpu'), unit = None):
    return EnsembleEvaluater('ani-pbe0_8x.info', torch.float32, device, unit)
    
def ani_1x_evaluater(device = torch.device('cpu'), unit = None):
    return EnsembleEvaluater('ani-1x_8x.info', torch.float32, device, unit)
    
def ani_2x_evaluater(device = torch.device('cpu'), unit = None):
    return EnsembleEvaluater('ani-2x_8x.info', torch.float32, device, unit)
    
