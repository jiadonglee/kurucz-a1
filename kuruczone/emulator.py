#!/usr/bin/env python
# kuruczone/emulator.py

import torch
import numpy as np
import os
import sys
from pathlib import Path

# Import the model class from the local module
from .model import AtmosphereNetMLPtau
from .normalization import NormalizationHelper, load_norm_params


class AtmosphereEmulator:
    """
    Emulator for Kurucz stellar atmosphere models.
    
    This class provides an interface to predict atmospheric structure
    based on stellar parameters and optical depth using a pre-trained
    neural network model.
    """
    
    def __init__(self, model, normalizer, default_tau_grid=None, device='cpu'):
        """
        Initialize the emulator with a pre-trained model and normalization helper.
        
        Parameters:
            model (torch.nn.Module): Pre-trained neural network model
            normalizer (NormalizationHelper): Normalization helper object
            default_tau_grid (torch.Tensor, optional): Default optical depth grid
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.model = model
        self.normalizer = normalizer
        self.default_tau_grid = default_tau_grid
        self.device = device
        self.model.to(device)
        self.model.eval()  # Set model to evaluation mode
        
        # Create default tau grid if none provided
        if self.default_tau_grid is None:
            self.default_tau_grid = torch.logspace(-6, 2, 80)  # 80 points from 1e-6 to 100
        
    def predict(self, stellar_params, tau_grid=None):
        """
        Predict atmospheric structure for given stellar parameters and optical depth grid.
        
        Parameters:
            stellar_params (torch.Tensor or array-like): Stellar parameters [Teff, log(g), [Fe/H], [α/Fe]]
                                                        Shape: [batch_size, 4] or [4]
            tau_grid (torch.Tensor or array-like, optional): Optical depth grid
                                                           Shape: [batch_size, n_depth_points] or [n_depth_points]
                                                           If None, uses the default grid
        
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
            tau_grid = self.default_tau_grid.unsqueeze(0).repeat(len(teff), 1)
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
        
        # Handle variable tau grid lengths by padding/truncating to model's expected size
        model_depth_points = 80  # This is what the model was trained with
        current_depth_points = tau_grid.size(1)
        
        # Store original tau grid for output before any modification
        original_tau_grid = tau_grid.clone()
        
        if current_depth_points != model_depth_points:
            if current_depth_points > model_depth_points:
                # Truncate to model size
                tau_grid = tau_grid[:, :model_depth_points]
            else:
                # Pad with the last tau value
                last_tau = tau_grid[:, -1:].expand(-1, model_depth_points - current_depth_points)
                tau_grid = torch.cat([tau_grid, last_tau], dim=1)
        
        # Normalize all parameters
        teff_norm = self.normalizer.normalize('teff', teff.unsqueeze(1))
        logg_norm = self.normalizer.normalize('gravity', logg.unsqueeze(1))
        feh_norm = self.normalizer.normalize('feh', feh.unsqueeze(1))
        afe_norm = self.normalizer.normalize('afe', afe.unsqueeze(1))
        tau_values = self.normalizer.normalize('TAU', tau_grid)
        
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
        
        # Truncate predictions to match original tau grid size if needed
        if current_depth_points != model_depth_points:
            predictions = predictions[:, :current_depth_points, :]
        
        # Denormalize predictions
        output_features = {
            'RHOX': self.normalizer.denormalize('RHOX', predictions[:, :, 0]),
            'T': self.normalizer.denormalize('T', predictions[:, :, 1]),
            'P': self.normalizer.denormalize('P', predictions[:, :, 2]),
            'XNE': self.normalizer.denormalize('XNE', predictions[:, :, 3]),
            'ABROSS': self.normalizer.denormalize('ABROSS', predictions[:, :, 4]),
        }
        
        # Add tau values to output (original, not normalized)
        output_features['TAU'] = original_tau_grid
        
        # Convert to numpy arrays if on CPU
        if self.device == 'cpu':
            output_features = {k: v.cpu().numpy()[0] for k, v in output_features.items()}
        
        output_features['teff'] = teff.item()
        output_features['logg'] = logg.item()  
        output_features['feh'] = feh.item()
        output_features['afe'] = afe.item()
        output_features['vturb'] = 2.0
        output_features['lonh'] = 1.25
        output_features['geom'] = 'PP'
        output_features['citation_info'] = r'''
            @ARTICLE{2025arXiv250706357L,
            author = {{Li}, Jiadong and {Jian}, Mingjie and {Ting}, Yuan-Sen and {Green}, Gregory M.},
            title = "{Differentiable Stellar Atmospheres with Physics-Informed Neural Networks}",
            journal = {arXiv e-prints},
            keywords = {Solar and Stellar Astrophysics, Earth and Planetary Astrophysics, Astrophysics of Galaxies, Instrumentation and Methods for Astrophysics},
            year = 2025,
            month = jul,
            eid = {arXiv:2507.06357},
            pages = {arXiv:2507.06357},
            doi = {10.48550/arXiv.2507.06357},
            archivePrefix = {arXiv},
            eprint = {2507.06357},
            primaryClass = {astro-ph.SR},
            adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250706357L},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}
            }
            '''

        return output_features


def load_from_checkpoint(checkpoint_path, hidden_size=512, norm_params_path=None, device='cpu'):
    """
    Load a pre-trained model from a checkpoint file.
    
    Parameters:
        checkpoint_path (str): Path to the checkpoint file
        norm_params_path (str, optional): Path to the normalization parameters file
                                         If None, tries to find it in the kuruczone directory
        device (str): Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        AtmosphereEmulator: Initialized emulator object
    """
    # Determine normalization parameters path if not provided
    if norm_params_path is None:
        # Try common locations
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'norm_params.pt'),
            os.path.join(os.path.dirname(checkpoint_path), 'norm_params.pt'),
            os.path.join(os.path.dirname(checkpoint_path), '../kuruczone/norm_params.pt')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                norm_params_path = path
                break
        
        if norm_params_path is None:
            raise FileNotFoundError(
                "Normalization parameters file not found. Please provide the norm_params_path parameter "
                "or ensure norm_params.pt exists in the kuruczone directory."
            )
    
    # Load normalization parameters
    try:
        norm_params = load_norm_params(norm_params_path)
        normalizer = NormalizationHelper(norm_params)
    except Exception as e:
        raise RuntimeError(f"Failed to load normalization parameters from {norm_params_path}: {e}")
    
    # Load model
    try:
        model = AtmosphereNetMLPtau(
            stellar_embed_dim=hidden_size, tau_embed_dim=hidden_size
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        
        print(f"Model loaded: trained for {checkpoint['epoch']} epochs, "
              f"final loss: {checkpoint['loss']:.6f}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {checkpoint_path}: {e}")
    
    # Create and return emulator
    return AtmosphereEmulator(model, normalizer, device=device)


def load_from_checkpoint_with_dataset(checkpoint_path, hidden_size=512, dataset_path=None, device='cpu'):
    """
    Load a pre-trained model from a checkpoint file using the full dataset (legacy method).
    
    This method is kept for backwards compatibility but requires the full dataset file.
    For production use, prefer load_from_checkpoint() which only needs normalization parameters.
    
    Parameters:
        checkpoint_path (str): Path to the checkpoint file
        dataset_path (str, optional): Path to the dataset file
                                     If None, tries to infer from common locations
        device (str): Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        AtmosphereEmulator: Initialized emulator object
    """
    from dataset import load_dataset_file
    
    # Determine dataset path if not provided
    if dataset_path is None:
        # Try common locations
        possible_paths = [
            os.path.join(os.path.dirname(checkpoint_path), '../data/kurucz_v5.pt'),
            os.path.join(os.path.dirname(checkpoint_path), 'data/kurucz_v5.pt'),
            os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'data/kurucz_v5.pt'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/kurucz_v5.pt')
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
        normalizer = NormalizationHelper(dataset.norm_params)
        # Extract default tau grid from dataset
        default_tau_grid = dataset.original['TAU'][0]  # Use first sample's tau grid
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {dataset_path}: {e}")
    
    # Load model
    try:
        model = AtmosphereNetMLPtau(
            stellar_embed_dim=hidden_size, tau_embed_dim=hidden_size
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        
        print(f"Model loaded: trained for {checkpoint['epoch']} epochs, "
              f"final loss: {checkpoint['loss']:.6f}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {checkpoint_path}: {e}")
    
    # Create and return emulator
    return AtmosphereEmulator(model, normalizer, default_tau_grid, device)