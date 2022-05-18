# MLXDM
# TORCHANI PBE0 XDM DISPERSION

Modified version of TorchANI github repository, which includes PBE0 functional trained version and dispersion correction

Feature includes in this modification:
1. Dispersion module: includes the network to map from AEV to coefficient 
2. Interface with ase using existed torchani code to compute the energy and atomic forces
3. Coefficients module: extract the information about components of dispersion energy


All of the models are includes in models.py unit, and can be called by
```python
import torchanipbe0
from torchanipbe0.models import ANIPBE0_MLXDM
model = ANIPBE0_MLXDM()
```

And the model can be attached to ase model by .ase() call.
```python
atoms.set_calculator(model.ase())
```

To compute just dispersion part, one can use dispersion_only option
```python
model = ANIPBE0_MLXDM(dispersion_only = True)
```

The package can be installed locally in a few minutes from
```bash
cd local-directory
pip install -e .
```

System requirement:
* Python 3 version > 3.8.1
* PyTorch (torch) version > 1.10
* Atomic Simulation Environment (ase) package version > 3.22
* h5py package version > 3.6
* requests package version > 2.26

All of the data is stored in resources/ folder, including the dispersion neural network in resources/dispersion folder.

This code is a modified version of the [TorchANI](https://github.com/aiqm/torchani) by Gao et al. that introduces a new dispersion term (MLXDM). All work using this code should cite:

TorchANI: A Free and Open Source PyTorch-Based Deep Learning Implementation of the ANI Neural Network Potentials
Xiang Gao, Farhad Ramezanghorbani, Olexandr Isayev, Justin S. Smith, and Adrian E. Roitberg
Journal of Chemical Information and Modeling 2020 60 (7), 3408-3415
DOI: 10.1021/acs.jcim.0c00451
