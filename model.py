#!/usr/bin/env python

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import glob
import numpy as np
# =============================================================================
# Model Architecture
# =============================================================================
class AtmosphereNet(torch.nn.Module):
    def __init__(self, input_size=5, hidden_size=256, output_size=6, depth_points=80):
        super(AtmosphereNet, self).__init__()
        self.input_size = input_size  # Now includes tau (teff, gravity, feh, afe, tau)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.depth_points = depth_points
        
        # Shared feature extractor for each depth point
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01)
        )
        
        # Final prediction layers
        self.output_layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            
            torch.nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, depth_points, input_size)
        batch_size, depth_points, _ = x.shape
        
        # Process each depth point independently
        # Reshape to (batch_size * depth_points, input_size)
        x_reshaped = x.reshape(-1, self.input_size)
        
        # Extract features for each depth point
        features = self.feature_extractor(x_reshaped)
        
        # Generate predictions for each depth point
        outputs = self.output_layers(features)
        
        # Reshape back to (batch_size, depth_points, output_size)
        outputs = outputs.reshape(batch_size, depth_points, self.output_size)
        
        return outputs

class AtmosphereNetMLP(torch.nn.Module):
    def __init__(self, input_size=5, hidden_size=256, output_size=6, depth_points=80):
        super(AtmosphereNetMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.depth_points = depth_points
        self.flattened_input_size = input_size * depth_points
        self.flattened_output_size = output_size * depth_points
        
        # MLP architecture
        self.layers = torch.nn.Sequential(
            # Input layer
            torch.nn.Linear(self.flattened_input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            
            # Hidden layers
            torch.nn.Linear(hidden_size, hidden_size*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            
            torch.nn.Linear(hidden_size*2, hidden_size*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            
            # Output layer
            torch.nn.Linear(hidden_size*2, self.flattened_output_size)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, depth_points, input_size)
        batch_size = x.size(0)
        
        # Flatten input
        x_flat = x.reshape(batch_size, -1)
        
        # Pass through MLP
        output_flat = self.layers(x_flat)
        
        # Reshape to (batch_size, depth_points, output_size)
        outputs = output_flat.reshape(batch_size, self.depth_points, self.output_size)
        
        return outputs

class StellarParamEncoder(nn.Module):
    """Encoder for global stellar parameters (Teff, logg, [M/H], [alpha/Fe])"""
    def __init__(self, input_dim=4, embed_dim=128):
        super(StellarParamEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        # x shape: (batch_size, 4)
        return self.encoder(x)  # output: (batch_size, embed_dim)

class TauPositionEncoder(nn.Module):
    """Position encoder for tau values at each depth point"""
    def __init__(self, embed_dim=64, depth_points=80):
        super(TauPositionEncoder, self).__init__()
        self.depth_points = depth_points
        self.encoder = nn.Sequential(
            nn.Linear(1, embed_dim//2),
            nn.GELU(),
            nn.Linear(embed_dim//2, embed_dim),
            nn.GELU()
        )
        
    def forward(self, tau):
        # tau shape: (batch_size, depth_points)
        batch_size = tau.size(0)
        # Reshape to process each tau value independently
        tau_reshaped = tau.reshape(batch_size * self.depth_points, 1)
        # Encode each tau value
        encoded = self.encoder(tau_reshaped)
        # Reshape back to separate batch and depth dimensions
        return encoded.reshape(batch_size, self.depth_points, -1)

class AtmosphereNetMLPtau(nn.Module):
    def __init__(self, output_size=6, depth_points=80, 
                 stellar_embed_dim=128, tau_embed_dim=64):
        super(AtmosphereNetMLPtau, self).__init__()
        self.depth_points = depth_points
        self.output_size = output_size
        
        # Encoders
        self.stellar_encoder = StellarParamEncoder(input_dim=4, embed_dim=stellar_embed_dim)
        self.tau_encoder = TauPositionEncoder(embed_dim=tau_embed_dim, depth_points=depth_points)
        
        # Combined embedding dimension
        combined_dim = stellar_embed_dim + tau_embed_dim
        
        # Atmospheric parameter predictor network
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, combined_dim*2),
            nn.GELU(),
            nn.Dropout(0.01),
            nn.Linear(combined_dim*2, combined_dim*2),
            nn.GELU(),
            nn.Dropout(0.01),
            nn.Linear(combined_dim*2, output_size)
        )
        
    def forward(self, x):
        # Extract stellar parameters and tau values
        # x shape: (batch_size, 84)
        batch_size = x.size(0)
        stellar_params = x[:, :4]  # (batch_size, 4)
        tau_values = x[:, 4:].reshape(batch_size, self.depth_points)  # (batch_size, depth_points)
        
        # Encode stellar parameters
        stellar_embedding = self.stellar_encoder(stellar_params)  # (batch_size, stellar_embed_dim)
        
        # Encode tau values
        tau_embedding = self.tau_encoder(tau_values)  # (batch_size, depth_points, tau_embed_dim)
        
        # Expand stellar embedding to match depth points dimension
        stellar_embedding = stellar_embedding.unsqueeze(1).expand(-1, self.depth_points, -1)
        
        # Combine embeddings
        combined = torch.cat([stellar_embedding, tau_embedding], dim=2)
        
        # Generate atmospheric parameters for each depth point
        outputs = self.predictor(combined)  # (batch_size, depth_points, output_size)
        
        return outputs