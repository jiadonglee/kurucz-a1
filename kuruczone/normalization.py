#!/usr/bin/env python
"""
Standalone normalization parameters for Kurucz stellar atmosphere models.

This module contains the normalization parameters extracted from the training dataset,
allowing the emulator to work independently without requiring the full dataset file.
"""

import torch


def extract_norm_params_from_dataset(dataset_path):
    """
    Extract normalization parameters from a saved dataset file.
    
    Parameters:
        dataset_path (str): Path to the dataset file
        
    Returns:
        dict: Normalization parameters
    """
    dataset = torch.load(dataset_path, map_location='cpu', weights_only=True)
    return dataset['norm_params']


def save_norm_params_to_file(norm_params, output_path):
    """
    Save normalization parameters to a standalone file.
    
    Parameters:
        norm_params (dict): Normalization parameters
        output_path (str): Path to save the parameters
    """
    torch.save(norm_params, output_path)
    print(f"Normalization parameters saved to {output_path}")


def load_norm_params(norm_params_path):
    """
    Load normalization parameters from a standalone file.
    
    Parameters:
        norm_params_path (str): Path to the normalization parameters file
        
    Returns:
        dict: Normalization parameters
    """
    return torch.load(norm_params_path, map_location='cpu', weights_only=True)


class NormalizationHelper:
    """
    Helper class for normalizing and denormalizing data using saved parameters.
    """
    
    def __init__(self, norm_params):
        """
        Initialize with normalization parameters.
        
        Parameters:
            norm_params (dict): Normalization parameters
        """
        self.norm_params = norm_params
        
    def normalize(self, param_name, data):
        """
        Normalize data to [-1, 1] range with optional log transform.

        Parameters:
            param_name (str): Name of the parameter to normalize
            data (torch.Tensor): Data to normalize

        Returns:
            torch.Tensor: Normalized data in [-1, 1] range
        """
        params = self.norm_params[param_name]
        
        # Apply log transform if needed
        if params['log_scale']:
            transformed_data = torch.log10(data + 1e-30)
        else:
            transformed_data = data
        
        # Apply min-max scaling to [-1, 1] range
        normalized = 2.0 * (transformed_data - params['min']) / (params['max'] - params['min']) - 1.0
        
        return normalized

    def denormalize(self, param_name, normalized_data):
        """
        Denormalize data from [-1, 1] range back to original scale.

        Parameters:
            param_name (str): Name of the parameter to denormalize
            normalized_data (torch.Tensor): Normalized data to convert back

        Returns:
            torch.Tensor: Denormalized data with gradients preserved
        """
        params = self.norm_params[param_name]
        
        # Reverse min-max scaling from [-1, 1] range
        transformed_data = (normalized_data + 1.0) / 2.0 * (params['max'] - params['min']) + params['min']
        
        # Reverse log transform if needed
        if params['log_scale']:
            denormalized_data = torch.pow(10.0, transformed_data) - 1e-30
        else:
            denormalized_data = transformed_data
        
        return denormalized_data


# Extract and save normalization parameters from the dataset
if __name__ == "__main__":
    import os
    
    # Path to the dataset file
    dataset_path = os.path.join(os.path.dirname(__file__), '../data/kurucz_v5.pt')
    
    # Extract normalization parameters
    norm_params = extract_norm_params_from_dataset(dataset_path)
    
    # Save to a standalone file
    output_path = os.path.join(os.path.dirname(__file__), 'norm_params.pt')
    save_norm_params_to_file(norm_params, output_path)
    
    print("Normalization parameters extracted and saved successfully!")