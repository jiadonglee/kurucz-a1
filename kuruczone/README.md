# Kurucz-a1: Kurucz Stellar Atmosphere Emulator

KurucZone is a Python package that provides a fast neural network-based emulator for Kurucz stellar atmosphere models. It allows you to quickly generate atmospheric structures based on stellar parameters (effective temperature, surface gravity, metallicity, and alpha enhancement).

## Installation

You can install the package directly from the repository:

```bash
pip install -e .
```

## Usage

### Basic Usage

```python
import torch
from kuruczone import emulator

# Load pre-trained model
model = emulator.load_from_checkpoint("model/a_one_weights.pt")

# Create stellar parameter inputs
stellar_params = torch.tensor([[5000.0, 4.5, -0.5, 0.0]])  # Teff, log(g), [Fe/H], [α/Fe]

# Create optical depth grid (optional)
tau_grid = torch.logspace(-7, 2, 100).unsqueeze(0)  # Shape: [1, 100]

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

## API Reference

### `emulator.load_from_checkpoint(checkpoint_path, dataset_path=None, device='cpu')`

Loads a pre-trained model from a checkpoint file.

- **Parameters:**

  - `checkpoint_path` (str): Path to the checkpoint file
  - `dataset_path` (str, optional): Path to the dataset file. If None, tries to infer from common locations
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
    - If None, uses the default grid from the dataset
- **Returns:**

  - `dict`: Dictionary containing atmospheric parameters for each depth point
    - Keys: 'T', 'P', 'RHOX', 'XNE', 'ABROSS', 'ACCRAD', 'TAU'
