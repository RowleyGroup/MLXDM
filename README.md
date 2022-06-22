# A Neural Network Potential with Rigorous Treatment of Long-Range Dispersion

## Contents
1. [Overview](#Overview)
2. [Repo Contents](#Repo-Contents)
3. [System Requirements](#System-Requirements)
4. [Installation Guide](#Installation-Guide)
5. [Demos and Expected Results](#Demos-and-Expected-Results)
6. [License](https://github.com/RowleyGroup/MLXDM/blob/main/LICENSE)
7. [Citation](#Citation)

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
pip install .
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

## Demos and Expected Results
The program can be tested by running md.py file in tests folder:
```bash
python tests/md.py
```
The program should return the ase optimization convergence:
```bash
BFGS:  107 21:04:57    -4145.728931        0.4356
BFGS:  108 21:04:57    -4145.640510        3.8882
BFGS:  109 21:04:58    -4145.615297        4.5117
BFGS:  110 21:04:58    -4145.752791        1.5679
BFGS:  111 21:04:58    -4145.772124        0.4735
BFGS:  112 21:04:59    -4145.778665        0.2748
BFGS:  113 21:04:59    -4145.781417        0.3394
BFGS:  114 21:04:59    -4145.784879        0.2482
BFGS:  115 21:05:00    -4145.787949        0.1067
BFGS:  116 21:05:00    -4145.788536        0.0914
```

The package can use GPU acceleration with CUDA-support pytorch. runtime.py script can be used to test the implementation on your platform:
```bash
python tests/runtime.py
```

The output should return the time for each model in the platforms:
```bash
Model 0 :0.010524554252624514 +/- 0.0020818100745546514
Model 1 :0.015416702747344972 +/- 0.0009169145365179207
Model 2 :0.007183948040008545 +/- 0.0005930249659918058
Model 3 :0.01378281879425049 +/- 0.0011457861853631064
Model 4 :0.017068558216094973 +/- 0.00037427370213380894
Model 5 :0.028385756015777586 +/- 0.0004956346444015525
```

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
