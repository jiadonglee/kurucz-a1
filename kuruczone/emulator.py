#!/usr/bin/env python
# kuruczone/emulator.py

import torch
import numpy as np
import os
import sys
from pathlib import Path

# Import from parent directory if needed
sys.path.append(str(Path(__file__).parent.parent))
from model import AtmosphereNetMLPtau
from dataset import load_dataset_file


class AtmosphereEmulator:
    """
    Emulator for Kurucz stellar atmosphere models.
    
    This class provides an interface to predict atmospheric structure
    based on stellar parameters and optical depth using a pre-trained
    neural network model.
    """
    
    def __init__(self, model, dataset, device='cpu'):
        """
        Initialize the emulator with a pre-trained model and dataset.
        
        Parameters:
            model (torch.nn.Module): Pre-trained neural network model
            dataset: Dataset object containing normalization parameters
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.to(device)
        self.model.eval()  # Set model to evaluation mode
        
    def predict(self, stellar_params, tau_grid=None):
        """
        Predict atmospheric structure for given stellar parameters and optical depth grid.
        
        Parameters:
            stellar_params (torch.Tensor or array-like): Stellar parameters [Teff, log(g), [Fe/H], [α/Fe]]
                                                        Shape: [batch_size, 4] or [4]
            tau_grid (torch.Tensor or array-like, optional): Optical depth grid
                                                           Shape: [batch_size, n_depth_points] or [n_depth_points]
                                                           If None, uses the default grid from the dataset
        
        Returns:
            dict: Dictionary containing atmospheric parameters for each depth point
                 Keys: 'T', 'P', 'RHOX', 'XNE', 'ABROSS', 'ACCRAD', 'TAU'
        """
        # Convert inputs to tensors if they aren't already
        if not isinstance(stellar_params, torch.Tensor):
            stellar_params = torch.tensor(stellar_params, dtype=torch.float32)
        
        # Add batch dimension if needed
        if stellar_params.dim() == 1:
            stellar_params = stellar_params.unsqueeze(0)
            
        # Move to the correct device
        stellar_params = stellar_params.to(self.device)
        
        # Extract individual parameters
        teff = stellar_params[:, 0]
        logg = stellar_params[:, 1]
        feh = stellar_params[:, 2]
        afe = stellar_params[:, 3]
        
        # Use default tau grid if none provided
        if tau_grid is None:
            tau_values = self.dataset.TAU[0].repeat(len(teff), 1)  # Use same TAU for all samples
        else:
            if not isinstance(tau_grid, torch.Tensor):
                tau_grid = torch.tensor(tau_grid, dtype=torch.float32)
            
            # Add batch dimension if needed
            if tau_grid.dim() == 1:
                tau_grid = tau_grid.unsqueeze(0)
                
            # Repeat if batch size doesn't match
            if tau_grid.size(0) == 1 and len(teff) > 1:
                tau_grid = tau_grid.repeat(len(teff), 1)
                
            # Move to the correct device
            tau_grid = tau_grid.to(self.device)
            
            # Normalize tau values
            tau_values = self.dataset.normalize('TAU', tau_grid)
        
        # Normalize stellar parameters
        teff_norm = self.dataset.normalize('teff', teff.unsqueeze(1))
        logg_norm = self.dataset.normalize('gravity', logg.unsqueeze(1))
        feh_norm = self.dataset.normalize('feh', feh.unsqueeze(1))
        afe_norm = self.dataset.normalize('afe', afe.unsqueeze(1))
        
        # Combine normalized parameters
        params_normalized = torch.cat([
            teff_norm,
            logg_norm,
            feh_norm,
            afe_norm,
            tau_values
        ], dim=1)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(params_normalized)
        
        # Reshape predictions (batch_size, depth_points, 6 features)
        num_depth_points = tau_values.size(1)
        predictions = predictions.view(-1, num_depth_points, 6)
        
        # Denormalize predictions
        output_features = {
            'RHOX': self.dataset.denormalize('RHOX', predictions[:, :, 0]),
            'T': self.dataset.denormalize('T', predictions[:, :, 1]),
            'P': self.dataset.denormalize('P', predictions[:, :, 2]),
            'XNE': self.dataset.denormalize('XNE', predictions[:, :, 3]),
            'ABROSS': self.dataset.denormalize('ABROSS', predictions[:, :, 4]),
            'ACCRAD': self.dataset.denormalize('ACCRAD', predictions[:, :, 5])
        }
        
        # Add tau values to output
        if tau_grid is None:
            output_features['TAU'] = self.dataset.denormalize('TAU', self.dataset.TAU[0:1]).repeat(len(teff), 1)
        else:
            output_features['TAU'] = self.dataset.denormalize('TAU', tau_values)
        
        # Convert to numpy arrays if on CPU
        if self.device == 'cpu':
            output_features = {k: v.cpu().numpy() for k, v in output_features.items()}
        
        return output_features


def load_from_checkpoint(checkpoint_path, dataset_path=None, device='cpu'):
    """
    Load a pre-trained model from a checkpoint file.
    
    Parameters:
        checkpoint_path (str): Path to the checkpoint file
        dataset_path (str, optional): Path to the dataset file
                                     If None, tries to infer from common locations
        device (str): Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        AtmosphereEmulator: Initialized emulator object
    """
    # Determine dataset path if not provided
    if dataset_path is None:
        # Try common locations
        possible_paths = [
            os.path.join(os.path.dirname(checkpoint_path), '../data/kurucz_vturb_0p5_tau_v3.pt'),
            os.path.join(os.path.dirname(checkpoint_path), 'data/kurucz_vturb_0p5_tau_v3.pt'),
            os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'data/kurucz_vturb_0p5_tau_v3.pt'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/kurucz_vturb_0p5_tau_v3.pt')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        if dataset_path is None:
            raise FileNotFoundError(
                "Dataset file not found. Please provide the dataset_path parameter."
            )
    
    # Load dataset
    try:
        dataset = load_dataset_file(dataset_path, device)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {dataset_path}: {e}")
    
    # Load model
    try:
        model = AtmosphereNetMLPtau(
            stellar_embed_dim=128, tau_embed_dim=64
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        
        print(f"Model loaded: trained for {checkpoint['epoch']} epochs, "
              f"final loss: {checkpoint['loss']:.6f}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {checkpoint_path}: {e}")
    
    # Create and return emulator
    return AtmosphereEmulator(model, dataset, device)