# Kurucz1: Kurucz Stellar Atmosphere Emulator

KurucZone is a Python package that provides a fast neural network-based emulator for Kurucz stellar atmosphere models. It allows you to quickly generate atmospheric structures based on stellar parameters (effective temperature, surface gravity, metallicity, and alpha enhancement).

## Overview

This project implements a Physics-Informed Neural Network (PINN) to emulate Kurucz stellar atmospheric models, using both stellar parameters and optical depth (τ) as inputs to predict atmospheric structures.

## Importance

Kurucz stellar atmospheric models are crucial in astrophysics but computationally expensive. Our PINN approach:

- Accelerates predictions by 1000x over traditional methods
- Maintains physical consistency across optical depth
- Enables accurate interpolation between model grid points
- Supports large-scale stellar population studies

## Features

- Neural emulation with optical depth (τ) as a key input parameter
- Physics constraints enforcing fundamental relationships (hydrostatic equilibrium)
- Input parameters: Teff, log(g), [Fe/H], [α/Fe], and τ
- Prediction of atmospheric quantities:
  - Temperature (T)
  - Pressure (P)
  - Column mass (RHOX)
  - Electron number density (XNE)
  - Rosseland mean opacity (ABROSS)
  - Radiative acceleration (ACCRAD)

## Installation

You can install the package directly from the repository:

```bash
git clone https://github.com/jiadonglee/kurucz1.git
cd kurucz1
pip install -e .
```

## Usage

### Basic Usage

```python
import torch
from kuruczone import emulator

# Load pre-trained model
model = emulator.load_from_checkpoint("checkpoints_v0528/checkpoint_epoch_50.pt")

# Create stellar parameter inputs
stellar_params = torch.tensor([[5000.0, 4.5, -0.5, 0.0]])  # Teff, log(g), [Fe/H], [α/Fe]

# Create optical depth grid (optional)
tau_grid = torch.logspace(-6, 2, 100).unsqueeze(0)  # Shape: [1, 100]

# Predict atmospheric structure
atmosphere = model.predict(stellar_params, tau_grid)

# Access variables
temperature = atmosphere['T']  # Shape: [batch_size, n_depth_points]
pressure = atmosphere['P']
```

### Available Atmospheric Variables

The emulator provides the following atmospheric variables:

- `T`: Temperature (K)
- `P`: Pressure (dyn/cm²)
- `RHOX`: Mass column density (g/cm²)
- `XNE`: Electron number density (cm⁻³)
- `ABROSS`: Rosseland mean absorption coefficient (cm²/g)
- `ACCRAD`: Radiative acceleration (cm/s²)
- `TAU`: Optical depth

### Batch Processing

You can process multiple stellar models at once by providing a batch of stellar parameters:

```python
# Create multiple stellar parameter inputs
stellar_params = torch.tensor([
    [5000.0, 4.5, -0.5, 0.0],  # Model 1
    [6000.0, 4.0, 0.0, 0.0],   # Model 2
    [4500.0, 2.5, -1.0, 0.3]    # Model 3
])

# Predict atmospheric structures for all models
atmospheres = model.predict(stellar_params)

# Access variables for specific models
temperature_model1 = atmospheres['T'][0]  # First model
temperature_model2 = atmospheres['T'][1]  # Second model
```

## Training

```bash
python train_hydro.py --dataset data/kurucz_v5.pt --gpu --epochs 1000 --lr 1e-4 --batch_size 256 --physics_weight 1e-3 --scheduler plateau
```

## API Reference

### `emulator.load_from_checkpoint(checkpoint_path, norm_params_path=None, device='cpu')`

Loads a pre-trained model from a checkpoint file.

- **Parameters:**

  - `checkpoint_path` (str): Path to the checkpoint file
  - `norm_params_path` (str, optional): Path to the normalization parameters file. If None, tries to infer from common locations
  - `device` (str): Device to load the model on ('cpu' or 'cuda')
- **Returns:**

  - `AtmosphereEmulator`: Initialized emulator object

### `AtmosphereEmulator.predict(stellar_params, tau_grid=None)`

Predicts atmospheric structure for given stellar parameters and optical depth grid.

- **Parameters:**

  - `stellar_params` (torch.Tensor or array-like): Stellar parameters [Teff, log(g), [Fe/H], [α/Fe]]
    - Shape: [batch_size, 4] or [4]
  - `tau_grid` (torch.Tensor or array-like, optional): Optical depth grid
    - Shape: [batch_size, n_depth_points] or [n_depth_points]
    - If None, uses the default grid
- **Returns:**

  - `dict`: Dictionary containing atmospheric parameters for each depth point
    - Keys: 'T', 'P', 'RHOX', 'XNE', 'ABROSS', 'ACCRAD', 'TAU'

## Citation

```
@misc{kurucz1-pinn,
  author = {Jiadong Li},
  title = {Physics-Informed Neural Emulation of Kurucz Stellar Atmospheric Models with Optical Depth Integration},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/jiadonglee/kurucz1}
}
```