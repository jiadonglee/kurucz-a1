#!/usr/bin/env python
# kuruczone/visualization.py

import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_atmosphere(atmosphere, variables=None, figsize=(12, 8), log_x=True, log_y=True):
    """
    Plot atmospheric structure variables against optical depth.
    
    Parameters:
        atmosphere (dict): Dictionary containing atmospheric variables
        variables (list, optional): List of variables to plot. If None, plots all available variables
        figsize (tuple): Figure size
        log_x (bool): Whether to use logarithmic scale for x-axis (tau)
        log_y (bool): Whether to use logarithmic scale for y-axis
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Default variables to plot
    all_vars = ['T', 'P', 'RHOX', 'XNE', 'ABROSS', 'ACCRAD']
    
    # Filter variables that exist in the atmosphere dictionary
    if variables is None:
        variables = [var for var in all_vars if var in atmosphere]
    else:
        variables = [var for var in variables if var in atmosphere]
    
    if not variables:
        raise ValueError("No valid variables to plot")
    
    # Get tau values
    if 'TAU' not in atmosphere:
        raise ValueError("Atmosphere dictionary must contain 'TAU' values")
    
    tau = atmosphere['TAU']
    
    # Convert to numpy if needed
    if isinstance(tau, torch.Tensor):
        tau = tau.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(len(variables), 1, figsize=figsize, sharex=True)
    if len(variables) == 1:
        axes = [axes]  # Make sure axes is always a list
    
    # Variable labels and units
    var_labels = {
        'T': 'Temperature (K)',
        'P': 'Pressure (dyn/cm²)',
        'RHOX': 'Mass Column Density (g/cm²)',
        'XNE': 'Electron Number Density (cm⁻³)',
        'ABROSS': 'Rosseland Mean Absorption (cm²/g)',
        'ACCRAD': 'Radiative Acceleration (cm/s²)'
    }
    
    # Plot each variable
    for i, var in enumerate(variables):
        var_data = atmosphere[var]
        
        # Convert to numpy if needed
        if isinstance(var_data, torch.Tensor):
            var_data = var_data.cpu().numpy()
        
        # Handle batch dimension
        if var_data.ndim > 1 and var_data.shape[0] > 1:
            for j in range(var_data.shape[0]):
                axes[i].plot(tau[j], var_data[j], label=f'Model {j+1}')
            axes[i].legend()
        else:
            # Squeeze to handle single batch dimension
            if var_data.ndim > 1:
                var_data = var_data.squeeze(0)
                tau_plot = tau.squeeze(0) if tau.ndim > 1 else tau
            else:
                tau_plot = tau
                
            axes[i].plot(tau_plot, var_data)
        
        # Set scales
        if log_x:
            axes[i].set_xscale('log')
        if log_y:
            axes[i].set_yscale('log')
        
        # Set labels
        axes[i].set_ylabel(var_labels.get(var, var))
        axes[i].grid(True, alpha=0.3)
    
    # Set common x label
    axes[-1].set_xlabel('Optical Depth (τ)')
    
    plt.tight_layout()
    return fig


def plot_hydrostatic_equilibrium(atmosphere, gravity, figsize=(10, 6)):
    """
    Plot the hydrostatic equilibrium condition: dP/dτ vs g/κ
    
    Parameters:
        atmosphere (dict): Dictionary containing atmospheric variables
        gravity (float or array-like): Surface gravity value(s) in cm/s²
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Check required variables
    required = ['P', 'TAU', 'ABROSS']
    for var in required:
        if var not in atmosphere:
            raise ValueError(f"Atmosphere dictionary must contain '{var}'")
    
    # Get values
    pressure = atmosphere['P']
    tau = atmosphere['TAU']
    kappa = atmosphere['ABROSS']  # Rosseland mean opacity
    
    # Convert to numpy if needed
    if isinstance(pressure, torch.Tensor):
        pressure = pressure.cpu().numpy()
    if isinstance(tau, torch.Tensor):
        tau = tau.cpu().numpy()
    if isinstance(kappa, torch.Tensor):
        kappa = kappa.cpu().numpy()
    
    # Convert gravity to numpy array if needed
    if not isinstance(gravity, np.ndarray):
        gravity = np.array(gravity)
    
    # Ensure gravity has the right shape
    if gravity.ndim == 0:
        gravity = np.array([gravity])
    
    # Calculate dP/dτ
    dp_dtau = np.zeros_like(pressure)
    
    # Handle batch dimension
    if pressure.ndim > 1:
        # For each model in the batch
        for i in range(pressure.shape[0]):
            # Calculate gradient for each model
            dp_dtau[i, 1:-1] = (pressure[i, 2:] - pressure[i, :-2]) / (tau[i, 2:] - tau[i, :-2])
            # Forward difference for first point
            dp_dtau[i, 0] = (pressure[i, 1] - pressure[i, 0]) / (tau[i, 1] - tau[i, 0])
            # Backward difference for last point
            dp_dtau[i, -1] = (pressure[i, -1] - pressure[i, -2]) / (tau[i, -1] - tau[i, -2])
    else:
        # Single model case
        dp_dtau[1:-1] = (pressure[2:] - pressure[:-2]) / (tau[2:] - tau[:-2])
        dp_dtau[0] = (pressure[1] - pressure[0]) / (tau[1] - tau[0])
        dp_dtau[-1] = (pressure[-1] - pressure[-2]) / (tau[-1] - tau[-2])
    
    # Calculate g/κ
    if pressure.ndim > 1:
        # Expand gravity to match the batch dimension
        if len(gravity) == 1 and pressure.shape[0] > 1:
            gravity = np.repeat(gravity, pressure.shape[0])
        
        # Ensure gravity has the right shape for broadcasting
        g_kappa = np.zeros_like(pressure)
        for i in range(pressure.shape[0]):
            g_kappa[i] = gravity[i] / kappa[i]
    else:
        g_kappa = gravity[0] / kappa
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot
    if pressure.ndim > 1 and pressure.shape[0] > 1:
        for i in range(pressure.shape[0]):
            ax.plot(tau[i], dp_dtau[i], label=f'dP/dτ (Model {i+1})')
            ax.plot(tau[i], g_kappa[i], '--', label=f'g/κ (Model {i+1})')
    else:
        # Squeeze to handle single batch dimension
        if pressure.ndim > 1:
            tau_plot = tau.squeeze(0)
            dp_dtau_plot = dp_dtau.squeeze(0)
            g_kappa_plot = g_kappa.squeeze(0)
        else:
            tau_plot = tau
            dp_dtau_plot = dp_dtau
            g_kappa_plot = g_kappa
            
        ax.plot(tau_plot, dp_dtau_plot, label='dP/dτ')
        ax.plot(tau_plot, g_kappa_plot, '--', label='g/κ')
    
    # Set scales and labels
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Optical Depth (τ)')
    ax.set_ylabel('dP/dτ and g/κ')
    ax.set_title('Hydrostatic Equilibrium Check')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_temperature_profile(atmospheres, stellar_params, figsize=(10, 6)):
    """
    Plot temperature profiles for multiple stellar models.
    
    Parameters:
        atmospheres (list): List of atmosphere dictionaries
        stellar_params (array-like): Array of stellar parameters [Teff, logg, [Fe/H], [α/Fe]]
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert stellar_params to numpy if needed
    if isinstance(stellar_params, torch.Tensor):
        stellar_params = stellar_params.cpu().numpy()
    
    # Ensure stellar_params is 2D
    if stellar_params.ndim == 1:
        stellar_params = stellar_params.reshape(1, -1)
    
    # Plot temperature profile for each model
    for i, atmosphere in enumerate(atmospheres):
        tau = atmosphere['TAU']
        temp = atmosphere['T']
        
        # Convert to numpy if needed
        if isinstance(tau, torch.Tensor):
            tau = tau.cpu().numpy()
        if isinstance(temp, torch.Tensor):
            temp = temp.cpu().numpy()
        
        # Squeeze to handle single batch dimension
        if tau.ndim > 1:
            tau = tau.squeeze(0)
        if temp.ndim > 1:
            temp = temp.squeeze(0)
        
        # Get stellar parameters for this model
        teff = stellar_params[i, 0]
        logg = stellar_params[i, 1]
        feh = stellar_params[i, 2]
        afe = stellar_params[i, 3] if stellar_params.shape[1] > 3 else 0.0
        
        # Plot
        ax.plot(tau, temp, label=f'Teff={teff:.0f}K, logg={logg:.1f}, [Fe/H]={feh:.1f}, [α/Fe]={afe:.1f}')
    
    # Set scales and labels
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Optical Depth (τ)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Temperature Profiles')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig