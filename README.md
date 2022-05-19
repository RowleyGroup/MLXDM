# A Neural Network Potential with Rigorous Treatment of Long-Range Dispersion

## Contents
1. [Overview](#Overview)
2. [Repo Contents](#Repo-Contents)
3. [System Requirements](#System-Requirements)
4. [Installation Guide](#Installation-Guide)
5. [Demo](#Demo)
6. [Results](#Results)
7. [License](https://github.com/RowleyGroup/MLXDM/blob/main/LICENSE)
8. [Citation](#Citation)

## Overview
Neural Network Potentials (NNPs) like ANI are powerful tools to describe chemical systems with a high level of accuracy that is comparable to DFT but with a much lower computational cost. Because they use short-range cutoffs (e.g., 5 A), interactions outside this range, like London interactions, are not neglected. As a result, this limits the accuracy of these models for intermolecular interactions. In this project, we developed a new NNP that explicitly models London dispersion interactions. The goal was achieved by calculating atomic dispersion coefficients with 6th, 8th, and 10th order terms (i.e., C6, C8, and C10) through a second NN, which is trained to reproduce the coefficients from the quantum-mechanically derived exchange-hole dipole moment (XDM) model.
Aseries of benchmark simulations and examples are included.

## Repo Contents
[tests](https://github.com/RowleyGroup/MLXDM/tree/main/tests):

[torchanipbe0](https://github.com/RowleyGroup/MLXDM/tree/main/torchanipbe0): 
* There is a folder named ``` resources ``` including all the data, including the dispersion neural network in resources/dispersion folder.

## System Requirements
### Hardware Requirements

### Software Requirements
#### OS Requirements
This package requires a Python 3 environment and has been tested on  MacOS, Linux, and Windows distributions.
The ANIPBE0-MLXDM package is tested on Linux operating systems.

### Python Dependencies
* Python 3 version > 3.8.1
* PyTorch (torch) version > 1.10
* Atomic Simulation Environment (ase) package version > 3.22
* h5py package version > 3.6
* requests package version > 2.26

## Installation Guide
1. Download the package 
2. Installing the package locally
```bash
cd local-directory
pip install -e .
```
3. All the models are included in models.py unit, and can be called by
```python
import torchanipbe0
from torchanipbe0.models import ANIPBE0_MLXDM
model = ANIPBE0_MLXDM()
```
4. And the model can be attached to ase model by calling .ase()
```python
atoms.set_calculator(model.ase())
```

5. To compute dispersion part, one can use dispersion_only option
```python
model = ANIPBE0_MLXDM(dispersion_only = True)
```

<! ## Demo>

<! ## Results >

## Citation

This code is a modified version of the [TorchANI](https://github.com/aiqm/torchani) by Gao et al. that introduces a new dispersion term (MLXDM). All work using this code should cite:

TorchANI: A Free and Open Source PyTorch-Based Deep Learning Implementation of the ANI Neural Network Potentials
Xiang Gao, Farhad Ramezanghorbani, Olexandr Isayev, Justin S. Smith, and Adrian E. Roitberg
Journal of Chemical Information and Modeling 2020 60 (7), 3408-3415
DOI: 10.1021/acs.jcim.0c00451
<Modified version of TorchANI github repository, which includes PBE0 functional trained version and dispersion correction

<!--- Feature includes in this modification:
1. Dispersion module: includes the network to map from AEV to coefficient 
2. Interface with ase using existed torchani code to compute the energy and atomic forces
3. Coefficients module: extract the information about components of dispersion energy --->
