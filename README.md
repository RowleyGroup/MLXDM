# MLXDM
# TORCHANI PBE0 XDM DISPERSION

Modified version of TorchANI github repository, which includes PBE0 functional trained version and dispersion correct

Feature includes in this modification:
1. Dispersion layer: includes the network to map from AEV to coefficient 

2. Interface with ase using existed torchani code to compute the energy and atomic forces

3. Use neighbor list to add the forces of intermolecular interaction arised from dispersion correction (currently test)

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

The package can be install locally from
```bash
cd local-directory
pip install -e .
```

All of the data is stored in resources/ folder, including the dispersion neural network in resources/dispersion folder.

This code is a modified version the [TorchANI](https://github.com/aiqm/torchani) by Gao et al. that introduces a new dispersion term (MLXDM). All work using this code should cite:

TorchANI: A Free and Open Source PyTorch-Based Deep Learning Implementation of the ANI Neural Network Potentials
Xiang Gao, Farhad Ramezanghorbani, Olexandr Isayev, Justin S. Smith, and Adrian E. Roitberg
Journal of Chemical Information and Modeling 2020 60 (7), 3408-3415
DOI: 10.1021/acs.jcim.0c00451
