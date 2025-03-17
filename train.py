#!/usr/bin/env python
# kurucz_train.py - Training script for Kurucz atmospheric models

import torch
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import time
import logging
from datetime import datetime

# =============================================================================
# Model Architecture
# =============================================================================
class AtmosphereNet(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=256, output_size=6, depth_points=80):
        super(AtmosphereNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.depth_points = depth_points
        
        # MLP architecture
        self.layers = torch.nn.Sequential(
            # Input layer
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            
            # Hidden layers
            torch.nn.Linear(hidden_size, hidden_size*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            
            torch.nn.Linear(hidden_size*2, hidden_size*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            
            # Output layer - maps to all depth points and parameters
            torch.nn.Linear(hidden_size*2, depth_points * output_size)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, input_size)
        batch_size = x.size(0)
        
        # Pass through MLP
        x = self.layers(x)
        
        # Reshape to (batch_size, depth_points, output_size)
        x = x.view(batch_size, self.depth_points, self.output_size)
        
        return x


# =============================================================================
# Dataset Loader
# =============================================================================
class KuruczDataset(Dataset):
    def __init__(self, saved_data=None, device='cpu'):
        """Initialize dataset from saved data dictionary"""
        if saved_data is not None:
            self.device = device
            self.norm_params = saved_data['norm_params']
            self.max_depth_points = saved_data['max_depth_points']
            
            # Load all tensors to device
            for key, value in saved_data['data'].items():
                if key == 'original' and isinstance(value, dict):
                    original_dict = {}
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            original_dict[k] = v.to(device)
                        else:
                            original_dict[k] = v
                    setattr(self, key, original_dict)
                else:
                    setattr(self, key, value.to(device))
                    
            # Initialize empty models list
            self.models = []
    
    def __len__(self):
        return len(self.teff)
    
    def __getitem__(self, idx):
        """Return normalized input features and output features"""
        # Input features (already normalized)
        input_features = torch.cat([
            self.teff[idx],
            self.gravity[idx],
            self.feh[idx],
            self.afe[idx]
        ], dim=0)
        
        # Output features (already normalized)
        output_features = torch.stack([
            self.RHOX[idx],
            self.T[idx],
            self.P[idx],
            self.XNE[idx],
            self.ABROSS[idx],
            self.ACCRAD[idx]
        ], dim=1)
        
        return input_features, output_features
    
    def denormalize(self, param_name, normalized_data):
        """Denormalize data using stored parameters"""
        params = self.norm_params[param_name]
        
        if param_name == 'teff':
            # Reverse the min-max normalization on log scale
            log_data = normalized_data * (params['max'] - params['min']) + params['min']
            # Convert back from log scale
            denormalized_data = torch.pow(10.0, log_data) - 1e-30
        
        elif param_name == 'T':
            # Reverse linear scaling for temperature
            denormalized_data = normalized_data * params['scale']
        
        elif params.get('scale_type') == 'min_max':
            # Reverse Min-Max scaling
            denormalized_data = normalized_data * (params['max'] - params['min']) + params['min']
        
        elif params.get('log_scale', False):
            # Reverse min-max scaling of log values
            log_data = normalized_data * (params['max'] - params['min']) + params['min']
            # Convert back from log scale
            denormalized_data = torch.pow(10.0, log_data) - 1e-30
        
        else:
            raise ValueError(f"Unknown denormalization approach for parameter: {param_name}")
        
        return denormalized_data
    
    def inverse_transform_outputs(self, outputs):
        """Transform normalized outputs back to physical units"""
        batch_size, depth_points, num_features = outputs.shape
        
        # Initialize containers for denormalized data
        result = {
            'RHOX': None,
            'T': None,
            'P': None,
            'XNE': None,
            'ABROSS': None,
            'ACCRAD': None
        }
        
        # Get column names
        columns = list(result.keys())
        
        # Denormalize each feature
        for i, param_name in enumerate(columns):
            # Extract the i-th feature for all samples and depths
            feature_data = outputs[:, :, i]
            # Denormalize
            result[param_name] = self.denormalize(param_name, feature_data)
        
        return result

# =============================================================================
# Training Functions
# =============================================================================
def custom_loss(pred, target, weights=None):
    """Calculate weighted loss for atmospheric parameters"""
    if weights is None:
        # Default weights prioritize temperature and pressure
        weights = {
            'RHOX': 1.0,
            'T': 2.0,
            'P': 1.5,
            'XNE': 1.0,
            'ABROSS': 1.0,
            'ACCRAD': 1.0
        }
    
    # Compute MSE loss for each output feature
    mse_loss = torch.nn.MSELoss(reduction='none')
    
    param_map = {0: 'RHOX', 1: 'T', 2: 'P', 3: 'XNE', 4: 'ABROSS', 5: 'ACCRAD'}
    
    total_loss = 0
    param_losses = {}
    
    for i, param in param_map.items():
        # Extract the i-th feature
        pred_feature = pred[:, :, i]
        target_feature = target[:, :, i]
        
        # Compute MSE
        feature_loss = mse_loss(pred_feature, target_feature)
        
        # Average over depth points
        feature_loss = feature_loss.mean(dim=1).mean()
        
        # Apply weight
        weighted_loss = feature_loss * weights[param]
        param_losses[param] = weighted_loss
        total_loss += weighted_loss
    
    return total_loss, param_losses

def save_checkpoint(model, optimizer, epoch, loss, filepath, logger):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    logger.info(f"Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, checkpoint_path, device, logger):
    """Load model checkpoint"""
    if not os.path.isfile(checkpoint_path):
        logger.warning(f"Checkpoint {checkpoint_path} not found, starting from scratch")
        return 0, float('inf')
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint.get('loss', float('inf'))
    logger.info(f"Resuming from epoch {start_epoch}, best loss: {best_loss}")
    
    return start_epoch, best_loss

# =============================================================================
# Logging
# =============================================================================
def setup_logger(log_dir):
    """Set up logger to write to file and console"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logger
    logger = logging.getLogger("AtmosphereModel")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to {log_file}")
    return logger

# =============================================================================
# Main Training Script
# =============================================================================
def train(model, dataloader, optimizer, device, start_epoch, args, logger):
    """Main training loop"""
    best_loss = float('inf')
    
    logger.info(f"Starting training for {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        
        # Initialize per-parameter loss tracking
        param_running_losses = {'RHOX': 0.0, 'T': 0.0, 'P': 0.0, 
                               'XNE': 0.0, 'ABROSS': 0.0, 'ACCRAD': 0.0}
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Forward pass and loss calculation
            optimizer.zero_grad()
            outputs = model(inputs)
            loss, param_losses = custom_loss(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            for param, param_loss in param_losses.items():
                param_running_losses[param] += param_loss.item()
            
            # Log batch progress periodically
            if (batch_idx + 1) % args.log_freq == 0:
                logger.info(f'Epoch: {epoch+1}/{args.epochs}, Batch: {batch_idx+1}/{len(dataloader)}, '
                           f'Loss: {loss.item():.6f}')
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(dataloader)
        epoch_time = time.time() - epoch_start_time
        
        # Calculate average parameter losses
        avg_param_losses = {k: v / len(dataloader) for k, v in param_running_losses.items()}
        param_loss_str = ', '.join([f"{k}: {v:.6f}" for k, v in avg_param_losses.items()])
        
        logger.info(f'Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {epoch_loss:.6f}')
        logger.info(f'Parameter losses: {param_loss_str}')
        
        # Save checkpoint if improvement or checkpoint frequency reached
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            logger.info(f'New best loss: {best_loss:.6f}')
            save_checkpoint(model, optimizer, epoch+1, epoch_loss, 
                           os.path.join(args.output_dir, 'best_model.pt'), logger)
        
        if (epoch + 1) % args.checkpoint_freq == 0:
            save_checkpoint(model, optimizer, epoch+1, epoch_loss,
                           os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'), logger)
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs, epoch_loss,
                   os.path.join(args.output_dir, 'final_model.pt'), logger)
    
    logger.info("Training completed!")
    return best_loss

def load_dataset_file(filepath, device):
    """
    Load a saved dataset.
    
    Parameters:
        filepath (str): Path to the saved dataset
        device (str): Device to load the data to ('cpu' or 'cuda')
        
    Returns:
        KuruczDataset: Loaded dataset
    """
    # Create an empty dataset
    dataset = KuruczDataset.__new__(KuruczDataset)
    
    # Load the saved data
    save_dict = torch.load(filepath, map_location=device)
    
    # Restore attributes
    dataset.norm_params = save_dict['norm_params']
    dataset.max_depth_points = save_dict['max_depth_points']
    dataset.device = device
    
    # Move data to the specified device
    for key, value in save_dict['data'].items():
        if key == 'original':
            # Handle the 'original' dictionary specially
            original_dict = {}
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    original_dict[k] = v.to(device)
                else:
                    original_dict[k] = v
            setattr(dataset, key, original_dict)
        else:
            # Normal tensor values
            setattr(dataset, key, value.to(device))
    
    # Initialize empty models list (not needed after loading)
    dataset.models = []
    
    return dataset

# =============================================================================
# Main Function
# =============================================================================
def create_dataloader_from_saved(filepath, batch_size=32, num_workers=4, device='cpu'):
    """
    Create a DataLoader from a saved dataset.
    
    Parameters:
        filepath (str): Path to the saved dataset
        batch_size (int): Batch size for the DataLoader
        num_workers (int): Number of worker processes
        device (str): Device to load the data to ('cpu' or 'cuda')
        
    Returns:
        tuple: (DataLoader, Dataset)
    """
    # Load the dataset
    dataset = load_dataset_file(filepath, device)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device=='cuda' else False,
    )
    
    return dataloader, dataset

def create_dataloader_from_saved(filepath, batch_size=32, num_workers=4, device='cpu'):
    """
    Create a DataLoader from a saved dataset.
    
    Parameters:
        filepath (str): Path to the saved dataset
        batch_size (int): Batch size for the DataLoader
        num_workers (int): Number of worker processes
        device (str): Device to load the data to ('cpu' or 'cuda')
        
    Returns:
        tuple: (DataLoader, Dataset)
    """
    # Load the dataset
    dataset = load_dataset_file(filepath, device)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device=='cuda' else False,
    )
    
    return dataloader, dataset

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Kurucz Atmospheric Model')
    parser.add_argument('--dataset', type=str, default="/Users/jdli/Desktop/AI4astro/kurucz1/data/kurucz_dataset.pt", 
                        required=True, help='load dataset file')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the model')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Checkpoint frequency (epochs)')
    parser.add_argument('--log_freq', type=int, default=10, help='Logging frequency (batches)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(args.log_dir)
    
    # Set device
    device = torch.device('mps' if torch.cuda.is_available() and args.gpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Log training configuration
    logger.info("Training configuration:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Load dataset and create dataloader
    logger.info(f"Loading dataset from {args.dataset}")
    dataloader, dataset = create_dataloader_from_saved(
        filepath=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers if device.type == 'cuda:1' else 0,
        device=device
    )
    logger.info(f"Dataset loaded, {len(dataset)} samples")
    
    # Create model
    model = AtmosphereNet(
        input_size=4,
        hidden_size=args.hidden_size,
        output_size=6,
        depth_points=dataset.max_depth_points
    ).to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device, logger)
    
    # Train the model
    train(model, dataloader, optimizer, device, start_epoch, args, logger)

if __name__ == "__main__":
    main()