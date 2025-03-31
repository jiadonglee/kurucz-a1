#!/usr/bin/env python
# examples/basic_usage.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add parent directory to path to import kuruczone
sys.path.append(str(Path(__file__).parent.parent))
from kuruczone import emulator
from kuruczone.visualization import plot_atmosphere, plot_hydrostatic_equilibrium

# Path to model checkpoint
checkpoint_path = "../checkpoints_v0327enc/best_model.pt"

# Load the emulator
print("Loading model...")
model = emulator.load_from_checkpoint(checkpoint_path)

# Create stellar parameter inputs
print("Creating stellar parameter inputs...")
stellar_params = torch.tensor([
    [5000.0, 4.5, 0.0, 0.0],   # Solar-like star
    [6500.0, 4.0, -0.5, 0.0],  # F-type star
    [4000.0, 2.0, -1.0, 0.3]    # K-giant star
])

# Create custom optical depth grid (optional)
tau_grid = torch.logspace(-6, 2, 80).unsqueeze(0)  # Shape: [1, 80]

# Predict atmospheric structure
print("Predicting atmospheric structure...")
atmosphere = model.predict(stellar_params, tau_grid)

# Print shapes of output arrays
print("\nOutput shapes:")
for key, value in atmosphere.items():
    print(f"{key}: {value.shape}")

# Access individual variables
temperature = atmosphere['T']  # Shape: [batch_size, n_depth_points]
pressure = atmosphere['P']

# Print temperature range for each model
print("\nTemperature ranges:")
for i in range(len(stellar_params)):
    teff = stellar_params[i, 0].item()
    t_min = temperature[i].min()
    t_max = temperature[i].max()
    print(f"Model {i+1} (Teff={teff:.0f}K): {t_min:.0f}K - {t_max:.0f}K")

# Plot atmospheric structure
print("\nPlotting atmospheric structure...")
fig1 = plot_atmosphere(atmosphere, variables=['T', 'P', 'RHOX'], figsize=(10, 12))
fig1.suptitle('Atmospheric Structure for Different Stellar Models', fontsize=16)
plt.savefig('atmosphere_structure.png', dpi=150, bbox_inches='tight')

# Plot hydrostatic equilibrium
print("Plotting hydrostatic equilibrium...")
# Convert log(g) to g in cm/s²
gravity = 10**stellar_params[:, 1].numpy() * 1e2  # Convert from log(g) in cgs
fig2 = plot_hydrostatic_equilibrium(atmosphere, gravity, figsize=(10, 6))
fig2.suptitle('Hydrostatic Equilibrium Check', fontsize=16)
plt.savefig('hydrostatic_equilibrium.png', dpi=150, bbox_inches='tight')

print("\nDone! Plots saved to atmosphere_structure.png and hydrostatic_equilibrium.png")

# Show plots if running interactively
plt.show()