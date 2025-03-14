# -*- coding: utf-8 -*-
"""The ANI model zoo that stores public ANI models.

Currently the model zoo has three models: ANI-1x, ANI-1ccx, and ANI-2x.
The parameters of these models are stored in `ani-model-zoo`_ repository and
will be automatically downloaded the first time any of these models are
instantiated. The classes of these models are :class:`ANI1x`, :class:`ANI1ccx`,
and :class:`ANI2x` these are subclasses of :class:`torch.nn.Module`.
To use the models just instantiate them and either
directly calculate energies or get an ASE calculator. For example:

.. _ani-model-zoo:
    https://github.com/aiqm/ani-model-zoo

.. code-block:: python

    ani1x = torchani.models.ANI1x()
    # compute energy using ANI-1x model ensemble
    _, energies = ani1x((species, coordinates))
    ani1x.ase()  # get ASE Calculator using this ensemble
    # convert atom species from string to long tensor
    ani1x.species_to_tensor(['C', 'H', 'H', 'H', 'H'])

    model0 = ani1x[0]  # get the first model in the ensemble
    # compute energy using the first model in the ANI-1x model ensemble
    _, energies = model0((species, coordinates))
    model0.ase()  # get ASE Calculator using this model
    # convert atom species from string to long tensor
    model0.species_to_tensor(['C', 'H', 'H', 'H', 'H'])
"""
import os
import torch
from torch import Tensor
from typing import Tuple, Optional, NamedTuple
from .nn import SpeciesConverter, SpeciesEnergies
from .aev import AEVComputer
from .dispersion.nn import ANIDispersion, CoefficientExtractorCC, DispersionLayer, CoefficientLayer
from .dispersion.nn import C6DispersionLayer, C8DispersionLayer, C10DispersionLayer
from .dispersion.nn import EnergyExtractor, CoefficientExtractor, CoefficientExtractorCC
from pathlib import Path

class SpeciesEnergiesQBC(NamedTuple):
    species: Tensor
    energies: Tensor
    qbcs: Tensor

class BuiltinModel(torch.nn.Module):
    r"""Private template for the builtin ANI models """

    def __init__(self, species_converter, aev_computer, neural_networks, energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index):
        super().__init__()
        self.species_converter = species_converter
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.energy_shifter = energy_shifter
        self._species_to_tensor = species_to_tensor
        self.species = consts.species
        self.periodic_table_index = periodic_table_index

        # a bit useless maybe
        self.consts = consts
        self.sae_dict = sae_dict


    @classmethod
    def _from_neurochem_resources(cls, info_file_path, periodic_table_index=False, model_index=0):
        from . import neurochem
        # this is used to load only 1 model (by default model 0)
        # const_file, sae_file, ensemble_prefix, ensemble_size = cls._parse_neurochem_resources(info_file_path)
        const_file, sae_file, ensemble_prefix, ensemble_size = neurochem.parse_neurochem_resources(info_file_path)
        if (model_index >= ensemble_size):
            raise ValueError("The ensemble size is only {}, model {} can't be loaded".format(ensemble_size, model_index))

        consts = neurochem.Constants(const_file)
        species_converter = SpeciesConverter(consts.species)
        aev_computer = AEVComputer(**consts)
        energy_shifter, sae_dict = neurochem.load_sae(sae_file, return_dict=True)
        species_to_tensor = consts.species_to_tensor

        network_dir = os.path.join('{}{}'.format(ensemble_prefix, model_index), 'networks')
        neural_networks = neurochem.load_model(consts.species, network_dir)

        return cls(species_converter, aev_computer, neural_networks,
                   energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index)


    def forward(self, species_coordinates: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        """Calculates predicted properties for minibatch of configurations

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_energies: energies for the given configurations

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            will be in Hartree.
        """
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)

        # check if unknown species are included
        if species_coordinates[0].ge(self.aev_computer.num_species).any():
            raise ValueError(f'Unknown species found in {species_coordinates[0]}')
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        species_energies = self.neural_networks(species_aevs)
        return self.energy_shifter(species_energies)

    @torch.jit.export
    def atomic_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                        cell: Optional[Tensor] = None,
                        pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        """Calculates predicted atomic energies of all atoms in a molecule

        ..warning::
            Since this function does not call ``__call__`` directly,
            hooks are not registered and profiling is not done correctly by
            pytorch on it. It is meant as a convenience function for analysis
             and active learning.

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            will be in Hartree.

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_atomic_energies: species and energies for the given configurations
                note that the shape of species is (C, A), where C is
                the number of configurations and A the number of atoms, and
                the shape of energies is (C, A) for a BuiltinModel.
        """
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        species, aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        atomic_energies = self.neural_networks._atomic_energies((species, aevs))
        self_energies = self.energy_shifter.self_energies.clone().to(species.device)
        self_energies = self_energies[species]
        self_energies[species == torch.tensor(-1, device=species.device)] = torch.tensor(0, device=species.device, dtype=torch.double)
        # shift all atomic energies individually
        assert self_energies.shape == atomic_energies.shape
        atomic_energies += self_energies
        return SpeciesEnergies(species, atomic_energies)

    @torch.jit.export
    def _recast_long_buffers(self):
        self.species_converter.conv_tensor = self.species_converter.conv_tensor.to(dtype=torch.long)
        self.aev_computer.triu_index = self.aev_computer.triu_index.to(dtype=torch.long)

    def species_to_tensor(self, *args, **kwargs):
        """Convert species from strings to tensor.

        See also :method:`torchani.neurochem.Constant.species_to_tensor`

        Arguments:
            species (:class:`str`): A string of chemical symbols

        Returns:
            tensor (:class:`torch.Tensor`): A 1D tensor of integers
        """
        # The only difference between this and the "raw" private version
        # _species_to_tensor is that this sends the final tensor to the model
        # device
        return self._species_to_tensor(*args, **kwargs) \
            .to(self.aev_computer.ShfR.device)

    def ase(self, **kwargs):
        """Get an ASE Calculator using this ANI model

        Arguments:
            kwargs: ase.Calculator kwargs

        Returns:
            calculator (:class:`int`): A calculator to be used with ASE
        """
        from . import ase
        return ase.Calculator(self.species, self, **kwargs)


class BuiltinEnsemble(BuiltinModel):
    """Private template for the builtin ANI ensemble models.

    ANI ensemble models form the ANI models zoo are instances of this class.
    This class is a torch module that sequentially calculates
    AEVs, then energies from a torchani.Ensemble and then uses EnergyShifter
    to shift those energies. It is essentially a sequential

    'AEVComputer -> Ensemble -> EnergyShifter'

    (periodic_table_index=False), or a sequential

    'SpeciesConverter -> AEVComputer -> Ensemble -> EnergyShifter'

    (periodic_table_index=True).

    .. note::
        This class is for internal use only, avoid relying on anything from it
        except the public methods, always use ANI1x, ANI1ccx, etc to instance
        the models.
        Also, don't confuse this class with torchani.Ensemble, which is only a
        container for many ANIModel instances and shouldn't be used directly
        for calculations.

    Attributes:
        species_converter (:class:`torchani.nn.SpeciesConverter`): Converts periodic table index to
            internal indices. Only present if periodic_table_index is `True`.
        aev_computer (:class:`torchani.AEVComputer`): AEV computer with
            builtin constants
        energy_shifter (:class:`torchani.EnergyShifter`): Energy shifter with
            builtin Self Atomic Energies.
        periodic_table_index (bool): Whether to use element number in periodic table
            to index species. If set to `False`, then indices must be `0, 1, 2, ..., N - 1`
            where `N` is the number of parametrized species.
    """

    def __init__(self, species_converter, aev_computer, neural_networks,
                 energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index):

        super().__init__(species_converter, aev_computer, neural_networks,
                         energy_shifter, species_to_tensor, consts, sae_dict,
                         periodic_table_index)

    @torch.jit.export
    def atomic_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                        cell: Optional[Tensor] = None,
                        pbc: Optional[Tensor] = None, average: bool = True) -> SpeciesEnergies:
        """Calculates predicted atomic energies of all atoms in a molecule

        see `:method:torchani.BuiltinModel.atomic_energies`

        If average is True (the default) it returns the average over all models
        (shape (C, A)), otherwise it returns one atomic energy per model (shape
        (M, C, A))
        """
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        species, aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        members_list = []
        for nnp in self.neural_networks:
            members_list.append(nnp._atomic_energies((species, aevs)).unsqueeze(0))
        member_atomic_energies = torch.cat(members_list, dim=0)

        self_energies = self.energy_shifter.self_energies.clone().to(species.device)
        self_energies = self_energies[species]
        self_energies[species == torch.tensor(-1, device=species.device)] = torch.tensor(0, device=species.device, dtype=torch.double)
        # shift all atomic energies individually
        assert self_energies.shape == member_atomic_energies.shape[1:]
        member_atomic_energies += self_energies
        if average:
            return SpeciesEnergies(species, member_atomic_energies.mean(dim=0))
        return SpeciesEnergies(species, member_atomic_energies)



    @classmethod
    def _from_neurochem_resources(cls, info_file_path, periodic_table_index=False):
        from . import neurochem
        # this is used to load only 1 model (by default model 0)
        # consts, sae_file, ensemble_prefix, ensemble_size = cls._parse_neurochem_resources(info_file_path)

        const_file, sae_file, ensemble_prefix, ensemble_size = neurochem.parse_neurochem_resources(info_file_path)
        consts = neurochem.Constants(const_file)

        species_converter = SpeciesConverter(consts.species)
        aev_computer = AEVComputer(**consts)
        energy_shifter, sae_dict = neurochem.load_sae(sae_file, return_dict=True)
        species_to_tensor = consts.species_to_tensor
        neural_networks = neurochem.load_model_ensemble(consts.species,
                                                        ensemble_prefix, ensemble_size)

        return cls(species_converter, aev_computer, neural_networks,
                   energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index)

    def __getitem__(self, index):
        """Get a single 'AEVComputer -> ANIModel -> EnergyShifter' sequential model

        Get a single 'AEVComputer -> ANIModel -> EnergyShifter' sequential model
        or
        Indexing allows access to a single model inside the ensemble
        that can be used directly for calculations. The model consists
        of a sequence AEVComputer -> ANIModel -> EnergyShifter
        and can return an ase calculator and convert species to tensor.

        Args:
            index (:class:`int`): Index of the model

        Returns:
            ret: (:class:`torchani.models.BuiltinModel`) Model ready for
                calculations
        """
        ret = BuiltinModel(self.species_converter, self.aev_computer,
                           self.neural_networks[index], self.energy_shifter,
                           self._species_to_tensor, self.consts, self.sae_dict,
                           self.periodic_table_index)
        return ret

    @torch.jit.export
    def members_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                         cell: Optional[Tensor] = None,
                         pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        """Calculates predicted energies of all member modules

        ..warning::
            Since this function does not call ``__call__`` directly,
            hooks are not registered and profiling is not done correctly by
            pytorch on it. It is meant as a convenience function for analysis
             and active learning.

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            will be in Hartree.

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_energies: species and energies for the given configurations
                note that the shape of species is (C, A), where C is
                the number of configurations and A the number of atoms, and
                the shape of energies is (M, C), where M is the number
                of modules in the ensemble

        """
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        species, aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        member_outputs = []
        for nnp in self.neural_networks:
            unshifted_energies = nnp((species, aevs)).energies
            shifted_energies = self.energy_shifter((species, unshifted_energies)).energies
            member_outputs.append(shifted_energies.unsqueeze(0))
        return SpeciesEnergies(species, torch.cat(member_outputs, dim=0))

    @torch.jit.export
    def energies_qbcs(self, species_coordinates: Tuple[Tensor, Tensor],
                      cell: Optional[Tensor] = None,
                      pbc: Optional[Tensor] = None, unbiased: bool = True) -> SpeciesEnergiesQBC:
        """Calculates predicted predicted energies and qbc factors

        QBC factors are used for query-by-committee (QBC) based active learning
        (as described in the ANI-1x paper `less-is-more`_ ).

        .. _less-is-more:
            https://aip.scitation.org/doi/10.1063/1.5023802

        ..warning::
            Since this function does not call ``__call__`` directly,
            hooks are not registered and profiling is not done correctly by
            pytorch on it. It is meant as a convenience function for analysis
             and active learning.

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            and qbc factors will be in Hartree.

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not
                enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set
                to None if PBC is not enabled
            unbiased: if `True` then Bessel's correction is applied to the
                standard deviation over the ensemble member's. If `False` Bessel's
                correction is not applied, True by default.

        Returns:
            species_energies_qbcs: species, energies and qbc factors for the
                given configurations note that the shape of species is (C, A),
                where C is the number of configurations and A the number of
                atoms, the shape of energies is (C,) and the shape of qbc
                factors is also (C,).
        """
        species, energies = self.members_energies(species_coordinates, cell, pbc)

        # standard deviation is taken across ensemble members
        qbc_factors = energies.std(0, unbiased=unbiased)

        # rho's (qbc factors) are weighted by dividing by the square root of
        # the number of atoms in each molecule
        num_atoms = (species >= 0).sum(dim=1, dtype=energies.dtype)
        qbc_factors = qbc_factors / num_atoms.sqrt()
        energies = energies.mean(dim=0)
        assert qbc_factors.shape == energies.shape
        return SpeciesEnergiesQBC(species, energies, qbc_factors)
    def __len__(self):
        """Get the number of networks in the ensemble

        Returns:
            length (:class:`int`): Number of networks in the ensemble
        """
        return len(self.neural_networks)


def ANI1x(periodic_table_index=False, model_index=None):
    """The ANI-1x model as in `ani-1x_8x on GitHub`_ and `Active Learning Paper`_.

    The ANI-1x model is an ensemble of 8 networks that was trained using
    active learning on the ANI-1x dataset, the target level of theory is
    wB97X/6-31G(d). It predicts energies on HCNO elements exclusively, it
    shouldn't be used with other atom types.

    .. _ani-1x_8x on GitHub:
        https://github.com/isayev/ASE_ANI/tree/master/ani_models/ani-1x_8x

    .. _Active Learning Paper:
        https://aip.scitation.org/doi/abs/10.1063/1.5023802
    """
    info_file = 'ani-1x_8x.info'
    if model_index is None:
        return BuiltinEnsemble._from_neurochem_resources(info_file, periodic_table_index)
    return BuiltinModel._from_neurochem_resources(info_file, periodic_table_index, model_index)


def ANI1ccx(periodic_table_index=False, model_index=None):
    """The ANI-1ccx model as in `ani-1ccx_8x on GitHub`_ and `Transfer Learning Paper`_.

    The ANI-1ccx model is an ensemble of 8 networks that was trained
    on the ANI-1ccx dataset, using transfer learning. The target accuracy
    is CCSD(T)*/CBS (CCSD(T) using the DPLNO-CCSD(T) method). It predicts
    energies on HCNO elements exclusively, it shouldn't be used with other
    atom types.

    .. _ani-1ccx_8x on GitHub:
        https://github.com/isayev/ASE_ANI/tree/master/ani_models/ani-1ccx_8x

    .. _Transfer Learning Paper:
        https://doi.org/10.26434/chemrxiv.6744440.v1
    """
    info_file = 'ani-1ccx_8x.info'
    if model_index is None:
        return BuiltinEnsemble._from_neurochem_resources(info_file, periodic_table_index)
    return BuiltinModel._from_neurochem_resources(info_file, periodic_table_index, model_index)


def ANI2x(periodic_table_index=False, model_index=None):
    """The ANI-2x model as in `ANI2x Paper`_ and `ANI2x Results on GitHub`_.

    The ANI-2x model is an ensemble of 8 networks that was trained on the
    ANI-2x dataset. The target level of theory is wB97X/6-31G(d). It predicts
    energies on HCNOFSCl elements exclusively it shouldn't be used with other
    atom types.

    .. _ANI2x Results on GitHub:
        https://github.com/cdever01/ani-2x_results

    .. _ANI2x Paper:
        https://doi.org/10.26434/chemrxiv.11819268.v1
    """
    info_file = 'ani-2x_8x.info'
    if model_index is None:
        return BuiltinEnsemble._from_neurochem_resources(info_file, periodic_table_index)
    return BuiltinModel._from_neurochem_resources(info_file, periodic_table_index, model_index)

# Modification from TorchANI start here
# Create new model and ensemble to allow different neural network load
# See neurochem
# Author: Tu Nguyen Thien Phuc
# Last update: 2022-10-17

# General model class

class BuiltinModel2(BuiltinModel):
    '''
    Update from BuiltinModel to allow different model load
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _from_neurochem_resources(cls, info_file_path, periodic_table_index=False, model_index=0):
        # Rewrite load model to load from best.pt file
        from . import neurochem
        const_file, sae_file, ensemble_prefix, ensemble_size = neurochem.parse_neurochem_resources(info_file_path)
        if (model_index >= ensemble_size):
            raise ValueError("The ensemble size is only {}, model {} can't be loaded".format(ensemble_size, model_index))

        consts = neurochem.Constants(const_file)
        species_converter = SpeciesConverter(consts.species)
        aev_computer = AEVComputer(**consts)
        energy_shifter, sae_dict = neurochem.load_sae(sae_file, return_dict=True)
        species_to_tensor = consts.species_to_tensor

        aev_dim = aev_computer.aev_length
        network_dir = f'{ensemble_prefix}{model_index}'
        neural_networks = neurochem.load_model_2(network_dir, aev_dim)

        return cls(species_converter, aev_computer, neural_networks,
                   energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index)

class BuiltinEnsemble2(BuiltinEnsemble):
    '''
    Update from BuiltinEnsemble to allow different model load
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _from_neurochem_resources(cls, info_file_path, periodic_table_index=False):
        # Rewrite to load model from best.pt
        from . import neurochem
        const_file, sae_file, ensemble_prefix, ensemble_size = neurochem.parse_neurochem_resources(info_file_path)
        consts = neurochem.Constants(const_file)

        species_converter = SpeciesConverter(consts.species)
        aev_computer = AEVComputer(**consts)
        energy_shifter, sae_dict = neurochem.load_sae(sae_file, return_dict=True)
        species_to_tensor = consts.species_to_tensor

        aev_dim = aev_computer.aev_length
        neural_networks = neurochem.load_model_ensemble_2(ensemble_prefix, ensemble_size, aev_dim)

        return cls(species_converter, aev_computer, neural_networks,
                   energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index)

class BuiltinModel3(BuiltinModel):
    '''
    Update from BuiltinModel to allow different model load
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _from_neurochem_resources(cls, info_file_path, periodic_table_index=False, model_index=0):
        # Rewrite load model to load from best.pt file
        from . import neurochem
        const_file, sae_file, ensemble_prefix, ensemble_size = neurochem.parse_neurochem_resources(info_file_path)
        if (model_index >= ensemble_size):
            raise ValueError("The ensemble size is only {}, model {} can't be loaded".format(ensemble_size, model_index))

        consts = neurochem.Constants(const_file)
        species_converter = SpeciesConverter(consts.species)
        aev_computer = AEVComputer(**consts)
        energy_shifter, sae_dict = neurochem.load_sae(sae_file, return_dict=True)
        species_to_tensor = consts.species_to_tensor

        aev_dim = aev_computer.aev_length
        network_dir = f'{ensemble_prefix}{model_index}'
        neural_networks = neurochem.load_model_3(network_dir, aev_dim)

        return cls(species_converter, aev_computer, neural_networks,
                   energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index)

class BuiltinEnsemble3(BuiltinEnsemble):
    '''
    Update from BuiltinEnsemble to allow different model load
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _from_neurochem_resources(cls, info_file_path, periodic_table_index=False):
        # Rewrite to load model from best.pt
        from . import neurochem
        const_file, sae_file, ensemble_prefix, ensemble_size = neurochem.parse_neurochem_resources(info_file_path)
        consts = neurochem.Constants(const_file)

        species_converter = SpeciesConverter(consts.species)
        aev_computer = AEVComputer(**consts)
        energy_shifter, sae_dict = neurochem.load_sae(sae_file, return_dict=True)
        species_to_tensor = consts.species_to_tensor

        aev_dim = aev_computer.aev_length
        neural_networks = neurochem.load_model_ensemble_3(ensemble_prefix, ensemble_size, aev_dim)

        return cls(species_converter, aev_computer, neural_networks,
                   energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index)

# Elemental model

def ANIPBE0(periodic_table_index = None, model_index = None):
    '''
    Return the TorchANI model to predict PBE0 functional energy
    '''
    info_file = 'ani-pbe0_8x.info'
    if model_index is None:
        return BuiltinEnsemble2._from_neurochem_resources(info_file, periodic_table_index)
    return BuiltinModel2._from_neurochem_resources(info_file, periodic_table_index, model_index)

def ANIPBE0_2x(periodic_table_index = None, model_index = None):
    '''                                                                                                                                                                                                     
    Return the TorchANI model to predict PBE0 functional energy                                                                                                                                             
    '''
    info_file = 'ani-pbe0-2x_8x.info'
    if model_index is None:
        return BuiltinEnsemble3._from_neurochem_resources(info_file, periodic_table_index)
    return BuiltinModel3._from_neurochem_resources(info_file, periodic_table_index, model_index)

def XDM_CC(device = None):
    '''
    Return XDM model with constants parameters for M1, M2, M3 and V
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    path = os.path.join(torchani_dir, 'resources/dispersion/')
    ani_model = ANIPBE0()
    ani_model = ani_model.to(device)
    m1_net = CoefficientLayer._from_constants_2(f'{path}m1/', dtype=torch.float32, device=device)
    m2_net = CoefficientLayer._from_constants_2(f'{path}m2/', dtype=torch.float32, device=device)
    m3_net = CoefficientLayer._from_constants_2(f'{path}m3/', dtype=torch.float32, device=device)
    v_net = CoefficientLayer._from_constants_2(f'{path}v/', dtype=torch.float32, device=device)
    return DispersionLayer(ani_model.aev_computer, m1_net, m2_net, m3_net, v_net,
                            0.4186, 2.6791, [4.4997895, 11.87706886, 7.423168043, 5.412164335], 
                            [8.2794385587, 35.403450375, 26.774856263, 22.577665436], 
                            14.0, torch.float32, device, ['H', 'C', 'N', 'O'])

def XDM_2x_CC(device = None):
    '''                                                                                                                                                                                                     
    Return XDM model with constants parameters for M1, M2, M3 and V                                                                                                                                         
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    path = os.path.join(torchani_dir, 'resources/dispersion_2x/')
    ani_model = ANIPBE0_2x()
    ani_model = ani_model.to(device)
    m1_net = CoefficientLayer._from_constants_2(f'{path}m1/', dtype=torch.float32, device=device)
    m2_net = CoefficientLayer._from_constants_2(f'{path}m2/', dtype=torch.float32, device=device)
    m3_net = CoefficientLayer._from_constants_2(f'{path}m3/', dtype=torch.float32, device=device)
    v_net = CoefficientLayer._from_constants_2(f'{path}v/', dtype=torch.float32, device=device)
    return DispersionLayer(ani_model.aev_computer, m1_net, m2_net, m3_net, v_net,
                            0.4186, 2.6791, [4.4997895, 11.87706887, 7.42316804, 5.41216434, 19.57017029, 3.75882236, 14.71136939],
                            [8.2794385587230224, 35.403450375407488, 26.774856262986901, 22.577665436425793, 75.344227406670839, 18.604506038051770, 65.219744182377752],
                            14.0, torch.float32, device, ['H', 'C', 'N', 'O', 'S', 'F', 'Cl'])

def MLXDM(device=None):
    '''
    Main MLXDM model
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    path = os.path.join(torchani_dir, 'resources/dispersion/')
    ani_model = ANIPBE0()
    ani_model = ani_model.to(device)
    m1_net = CoefficientLayer._from_file_2(f'{path}m1/', device=device)
    m2_net = CoefficientLayer._from_file_2(f'{path}m2/', device=device)
    m3_net = CoefficientLayer._from_file_2(f'{path}m3/', device=device)
    v_net = CoefficientLayer._from_file_2(f'{path}v/', device=device)
    return DispersionLayer(ani_model.aev_computer, m1_net, m2_net, m3_net, v_net,
                            0.4186, 2.6791, [4.4997895, 11.87706886, 7.423168043, 5.412164335], 
                            [8.2794385587, 35.403450375, 26.774856263, 22.577665436], 
                            14.0, torch.float32, device, ['H', 'C', 'N', 'O'])

def MLXDM_2x(device=None):
    '''                                                                                                                                                                                                     
    Main MLXDM model                                                                                                                                                                                        
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    path = os.path.join(torchani_dir, 'resources/dispersion_2x/')
    ani_model = ANIPBE0_2x()
    ani_model = ani_model.to(device)
    m1_net = CoefficientLayer._from_file_3(f'{path}m1/', device=device)
    m2_net = CoefficientLayer._from_file_3(f'{path}m2/', device=device)
    m3_net = CoefficientLayer._from_file_3(f'{path}m3/', device=device)
    v_net = CoefficientLayer._from_file_3(f'{path}v/', device=device)
    return DispersionLayer(ani_model.aev_computer, m1_net, m2_net, m3_net, v_net,
                            0.4186, 2.6791, [4.4997895, 11.87706887, 7.42316804, 5.41216434, 19.57017029, 3.75882236, 14.71136939],
                            [8.2794385587230224, 35.403450375407488, 26.774856262986901, 22.577665436425793, 75.344227406670839, 18.604506038051770, 65.219744182377752],
                            14.0, torch.float32, device, ['H', 'C', 'N', 'O', 'S', 'F', 'Cl'])


# Combine

def ANIPBE0_MLXDM(device=None):
    '''
    The main model with ANIPBE0 and MLXDM
    '''
    ani_model = ANIPBE0()
    ani_model = ani_model.to(device)
    dispersion_model = MLXDM(device)
    return ANIDispersion(ani_model, dispersion_model)

def ANIPBE0_2x_MLXDM_2x(device=None):
    '''                                                                                                                                                                                                     
    The model with ANIPBE0_2x and MLXDM_2x                                                                                                                                                                   
    '''
    ani_model = ANIPBE0_2x()
    ani_model = ani_model.to(device)
    dispersion_model = MLXDM_2x(device)
    return ANIDispersion(ani_model, dispersion_model)

def ANI1x_MLXDM(device=None):
    '''
    MLXDM with the ANI1x
    '''
    ani_model = ANI1x()
    ani_model = ani_model.to(device)
    dispersion_model = MLXDM(device)
    return ANIDispersion(ani_model, dispersion_model)

def ANI1ccx_MLXDM(device=None):
    '''
    MLXDM with ANI1ccx
    '''
    ani_model = ANI1ccx()
    ani_model = ani_model.to(device)
    dispersion_model = MLXDM(device)
    return ANIDispersion(ani_model, dispersion_model)

def ANIPBE0_XDM_CC(device=None):
    '''
    ANIPBE0 with the constant M1, M2, M3 and V coefficients XDM
    '''
    ani_model = ANIPBE0()
    ani_model = ani_model.to(device)
    dispersion_model = XDM_CC(device)
    return ANIDispersion(ani_model, dispersion_model)

def ANIPBE0_2x_XDM_2x_CC(device=None):
    '''                                                                                                                                                                                                     
    ANIPBE02x with the constant M1, M2, M3 and V coefficients XDM                                                                                                                                             
    '''
    ani_model = ANIPBE0_2x()
    ani_model = ani_model.to(device)
    dispersion_model = XDM_2x_CC(device)
    return ANIDispersion(ani_model, dispersion_model)

def ANI1x_XDM_CC(device=None):
    '''
    ANI1x with the constant M1, M2, M3 and V coefficients XDM
    '''
    ani_model = ANI1x()
    ani_model = ani_model.to(device)
    dispersion_model = XDM_CC(device)
    return ANIDispersion(ani_model, dispersion_model)

def ANI1ccx_XDM_CC(device=None):
    '''
    ANI1ccx with the constant M1, M2, M3 and V coefficients XDM
    '''
    ani_model = ANI1ccx()
    ani_model = ani_model.to(device)
    dispersion_model = XDM_CC(device)
    return ANIDispersion(ani_model, dispersion_model)

# Dispersion coefficients and energy extration

def c6_energy(device = None):
    '''
    Return the dispersion energy prediction for 6th order components only
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    path = os.path.join(torchani_dir, 'resources/dispersion/')
    ani_model = ANIPBE0()
    ani_model = ani_model.to(device)
    m1_net = CoefficientLayer._from_file_2(f'{path}m1/', device=device)
    m2_net = CoefficientLayer._from_file_2(f'{path}m2/', device=device)
    m3_net = CoefficientLayer._from_file_2(f'{path}m3/', device=device)
    v_net = CoefficientLayer._from_file_2(f'{path}v/', device=device)
    return C6DispersionLayer(ani_model.aev_computer, m1_net, m2_net, m3_net, v_net,
                            0.4186, 2.6791, [4.4997895, 11.87706886, 7.423168043, 5.412164335], 
                            [8.2794385587, 35.403450375, 26.774856263, 22.577665436], 
                            14.0, torch.float32, device, ['H', 'C', 'N', 'O'])

def c8_energy(device = None):
    '''
    Return the dispersion energy prediction for 8th order components only
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    path = os.path.join(torchani_dir, 'resources/dispersion/')
    ani_model = ANIPBE0()
    ani_model = ani_model.to(device)
    m1_net = CoefficientLayer._from_file_2(f'{path}m1/', device=device)
    m2_net = CoefficientLayer._from_file_2(f'{path}m2/', device=device)
    m3_net = CoefficientLayer._from_file_2(f'{path}m3/', device=device)
    v_net = CoefficientLayer._from_file_2(f'{path}v/', device=device)
    return C8DispersionLayer(ani_model.aev_computer, m1_net, m2_net, m3_net, v_net,
                            0.4186, 2.6791, [4.4997895, 11.87706886, 7.423168043, 5.412164335], 
                            [8.2794385587, 35.403450375, 26.774856263, 22.577665436], 
                            14.0, torch.float32, device, ['H', 'C', 'N', 'O'])

def c10_energy(device = None):
    '''
    Return the dispersion energy prediction for 10th order components only
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    path = os.path.join(torchani_dir, 'resources/dispersion/')
    ani_model = ANIPBE0()
    ani_model = ani_model.to(device)
    m1_net = CoefficientLayer._from_file_2(f'{path}m1/', device=device)
    m2_net = CoefficientLayer._from_file_2(f'{path}m2/', device=device)
    m3_net = CoefficientLayer._from_file_2(f'{path}m3/', device=device)
    v_net = CoefficientLayer._from_file_2(f'{path}v/', device=device)
    return C10DispersionLayer(ani_model.aev_computer, m1_net, m2_net, m3_net, v_net,
                            0.4186, 2.6791, [4.4997895, 11.87706886, 7.423168043, 5.412164335], 
                            [8.2794385587, 35.403450375, 26.774856263, 22.577665436], 
                            14.0, torch.float32, device, ['H', 'C', 'N', 'O'])


def m1_coefficients(device=None):
    '''
    Return the coefficients M1
    Use compute_from_ase for this
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = ANIPBE0()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion/m1/')
    return CoefficientExtractor._from_file(path, ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)

def m2_coefficients(device=None):
    '''
    Return the coefficients M2
    Use compute_from_ase for this
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = ANIPBE0()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion/m2/')
    return CoefficientExtractor._from_file(path, ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)

def m3_coefficients(device=None):
    '''
    Return the coefficients M3
    Use compute_from_ase for this
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = ANIPBE0()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion/m3/')
    return CoefficientExtractor._from_file(path, ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)

def v_coefficients(device=None):
    '''
    Return the coefficients V
    Use compute_from_ase for this
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = ANIPBE0()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion/v/')
    return CoefficientExtractor._from_file(path, ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)


def m1_2x_coefficients(device=None): ##### HERE
    '''
    Return the coefficients M1
    Use compute_from_ase for this
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = ANIPBE0_2x()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion_2x/m1/')
    return CoefficientExtractor._from_file_3(path, ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)

def m2_2x_coefficients(device=None):
    '''
    Return the coefficients M2
    Use compute_from_ase for this
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = ANIPBE0_2x()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion_2x/m2/')
    return CoefficientExtractor._from_file_3(path, ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)

def m3_2x_coefficients(device=None):
    '''
    Return the coefficients M3
    Use compute_from_ase for this
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = ANIPBE0_2x()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion_2x/m3/')
    return CoefficientExtractor._from_file_3(path, ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)

def v_2x_coefficients(device=None):
    '''
    Return the coefficients V
    Use compute_from_ase for this
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = ANIPBE0_2x()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion_2x/v/')
    return CoefficientExtractor._from_file_3(path, ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)


def m1_coefficients_CC(device=None):
    '''
    Return the constant coefficients M1
    Use compute_from_ase for this
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = ANIPBE0()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion/m1/')
    return CoefficientExtractorCC._from_file(path, ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)

def m2_coefficients_CC(device=None):
    '''
    Return the constant coefficients M2
    Use compute_from_ase for this
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = ANIPBE0()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion/m2/')
    return CoefficientExtractorCC._from_file(path, ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)

def m3_coefficients_CC(device=None):
    '''
    Return the constant coefficients M3
    Use compute_from_ase for this
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = ANIPBE0()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion/m3/')
    return CoefficientExtractorCC._from_file(path, ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)

def v_coefficients_CC(device=None):
    '''
    Return the constant coefficients V
    Use compute_from_ase for this
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = ANIPBE0()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion/v/')
    return CoefficientExtractorCC._from_file(path, ani_model.aev_computer, ani_model.species_to_tensor, torch.float32, device)

def pairwise_energy_extractor(device=None):
    '''
    Return the extractor for pairwise interaction of systems
    The output is a set of: atom index, distance, c6 energy pair, c8 energy pair, c10 energy pair
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    ani_model = ANIPBE0()
    ani_model = ani_model.to(device)
    path = os.path.join(torchani_dir, 'resources/dispersion/')
    m1_net = CoefficientLayer._from_file_2(f'{path}m1/', device=device)
    m2_net = CoefficientLayer._from_file_2(f'{path}m2/', device=device)
    m3_net = CoefficientLayer._from_file_2(f'{path}m3/', device=device)
    v_net = CoefficientLayer._from_file_2(f'{path}v/', device=device)
    return EnergyExtractor(ani_model.aev_computer, m1_net, m2_net, m3_net, v_net,
                            0.4186, 2.6791, [4.4997895, 11.87706886, 7.423168043, 5.412164335], 
                            [8.2794385587, 35.403450375, 26.774856263, 22.577665436], 
                            14.0, ani_model.species_to_tensor, torch.float32, device)

def pairwise_energy_CC_extractor(device=None):
    '''
    Return the extractor for pairwise interaction of systems with constant coefficients M1, M2, M3, V
    The output is a set of: atom index, distance, c6 energy pair, c8 energy pair, c10 energy pair
    '''
    torchani_dir = Path(__file__).resolve().parent.as_posix()
    path = os.path.join(torchani_dir, 'resources/dispersion/')
    ani_model = ANIPBE0()
    ani_model = ani_model.to(device)
    m1_net = CoefficientLayer._from_constants_2(f'{path}m1/', dtype=torch.float32, device=device)
    m2_net = CoefficientLayer._from_constants_2(f'{path}m2/', dtype=torch.float32, device=device)
    m3_net = CoefficientLayer._from_constants_2(f'{path}m3/', dtype=torch.float32, device=device)
    v_net = CoefficientLayer._from_constants_2(f'{path}v/', dtype=torch.float32, device=device)
    return EnergyExtractor(ani_model.aev_computer, m1_net, m2_net, m3_net, v_net,
                            0.4186, 2.6791, [4.4997895, 11.87706886, 7.423168043, 5.412164335], 
                            [8.2794385587, 35.403450375, 26.774856263, 22.577665436], 
                            14.0, ani_model.species_to_tensor, torch.float32, device)
