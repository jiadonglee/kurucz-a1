# Kurucz1: Stellar Atmospheric Model Neural Emulator

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

```bash
git clone https://github.com/jiadonglee/kurucz1.git
cd kurucz1
pip install -r requirements.txt
```

## Usage

```python
import torch
from model import AtmosphereNet

# Load pre-trained model
model = AtmosphereNet.load_from_checkpoint("checkpoints_v0327enc_hydro/best_model.pt")

# Create stellar parameter inputs
stellar_params = torch.tensor([[5000.0, 4.5, -0.5, 0.0]])  # Teff, log(g), [Fe/H], [α/Fe]

# Create optical depth grid
tau_grid = torch.logspace(-6, 2, 100).unsqueeze(0)  # Shape: [1, 100]

# Predict atmospheric structure
atmosphere = model.predict(stellar_params, tau_grid)

# Access variables
temperature = atmosphere['T']  # Shape: [batch_size, n_depth_points]
pressure = atmosphere['P']
```

## Training

```bash
python train.py --dataset /path/to/kurucz_dataset.pt --epochs 100 --batch_size 32 --gpu
```

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
