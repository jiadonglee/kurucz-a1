#!/usr/bin/env python
# train_hydro.py - Training script for atmospheric models with physics-informed loss

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import time
import logging
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from model import AtmosphereNet, KuruczDataset

# =============================================================================
# Physics-Informed Loss Functions
# =============================================================================
def hydrostatic_equilibrium_loss(outputs, inputs, param_indices, dataset, model):
    """
    Log-space hydrostatic equilibrium loss without sign penalty.
    Compares log(dP/dτ) with log(g/κ) for better numerical stability.
    """
    batch_size, depth_points, _ = outputs.shape
    device = outputs.device
    
    # Get parameter indices
    P_idx = param_indices.get('P', 2)
    ABROSS_idx = param_indices.get('ABROSS', 4)
    
    # Prepare optical depth for gradient calculation
    tau_norm = inputs[:, :, 4].detach().clone().requires_grad_(True)
    logg_norm = inputs[:, 0, 1].detach().clone()
    
    # Forward pass with gradient tracking
    grad_inputs = inputs.detach().clone()
    grad_inputs[:, :, 4] = tau_norm
    
    with torch.enable_grad():
        outputs_with_grad = model(grad_inputs)
        P_norm = outputs_with_grad[:, :, P_idx].clamp(-10.0, 10.0)
        
        # Calculate pressure gradient
        try:
            dP_dtau_norm = torch.autograd.grad(
                outputs=P_norm,
                inputs=tau_norm,
                grad_outputs=torch.ones_like(P_norm),
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Handle any non-finite values
            dP_dtau_norm = torch.where(
                torch.isfinite(dP_dtau_norm),
                dP_dtau_norm.clamp(-1e4, 1e4),
                torch.zeros_like(dP_dtau_norm)
            )
        except Exception as e:
            print(f"Gradient calculation error: {e}")
            dP_dtau_norm = torch.zeros_like(tau_norm)
    
    # Convert to physical units
    logg = dataset.denormalize('gravity', logg_norm)
    g = 10**logg.view(-1, 1)
    
    # Handle pressure and gradient calculation
    if dataset.norm_params['P'].get('log_scale', False):
        log_P = dataset.denormalize('P', P_norm)
        P = 10**log_P.clamp(-30.0, 30.0)
        p_scale = dataset.norm_params['P'].get('scale', 1.0)
        dP_dtau = dP_dtau_norm * P * np.log(10) * p_scale
    else:
        P = dataset.denormalize('P', P_norm).clamp(1e-10, 1e10)
        p_scale = dataset.norm_params['P'].get('scale', 
                dataset.norm_params['P'].get('max', 1.0) - dataset.norm_params['P'].get('min', 0.0))
        tau_scale = dataset.norm_params['TAU'].get('scale',
                  dataset.norm_params['TAU'].get('max', 1.0) - dataset.norm_params['TAU'].get('min', 0.0))
        dP_dtau = dP_dtau_norm * (p_scale / tau_scale)
    
    # Get opacity (kappa)
    ABROSS_norm = outputs[:, :, ABROSS_idx].clamp(-5.0, 5.0)
    kappa = dataset.denormalize('ABROSS', ABROSS_norm)
    kappa_safe = kappa.clamp(1e-30, 1e30)
    
    # Calculate g/kappa
    g_kappa = g / kappa_safe
    
    # Convert to log space with clamping for stability
    log_dP_dtau = torch.log10(dP_dtau.abs().clamp(min=1e-10))
    log_g_kappa = torch.log10(g_kappa.clamp(min=1e-10))
    
    # Calculate MSE in log space
    log_mse = F.mse_loss(log_dP_dtau, log_g_kappa)
    
    return log_mse

def simplified_hydrostatic_loss(outputs, inputs, param_indices, dataset):
    """
    Simplified hydrostatic equilibrium loss using finite differences.
    Computes: dP/dtau ≈ g/kappa in log space for better numerical stability.
    
    Args:
        outputs: Model predictions
        inputs: Model inputs containing stellar parameters
        param_indices: Dictionary mapping parameter names to indices
        dataset: Dataset containing denormalization methods
        
    Returns:
        torch.Tensor: Log-space MSE loss between dP/dtau and g/kappa
    """
    # Extract parameter indices
    P_idx = param_indices['P']
    ABROSS_idx = param_indices['ABROSS']
    
    # Get normalized values
    P_norm = outputs[:, :, P_idx]
    kappa_norm = outputs[:, :, ABROSS_idx]
    tau_norm = inputs[:, :, 4]  # Optical depth
    logg_norm = inputs[:, 0, 1]  # Log surface gravity
    
    # Denormalize to physical units
    P = dataset.denormalize('P', P_norm)
    tau = dataset.denormalize('TAU', tau_norm)
    kappa = dataset.denormalize('ABROSS', kappa_norm)
    logg = dataset.denormalize('gravity', logg_norm)
    
    # Convert log(g) to linear scale (cm/s²)
    g = 10.0 ** logg
    
    # Calculate dP/dtau using finite differences
    dP = P[:, 1:] - P[:, :-1]
    dtau = tau[:, 1:] - tau[:, :-1]
    dP_dtau = dP / torch.clamp(dtau, min=1e-10)
    
    # Calculate opacity at grid midpoints
    kappa_avg = 0.5 * (kappa[:, 1:] + kappa[:, :-1])
    
    # Compute g/kappa term
    g_expanded = g.view(-1, 1).expand(-1, dP_dtau.size(1))
    g_kappa = g_expanded / torch.clamp(kappa_avg, min=1e-10)
    
    # Use log-space loss for improved numerical stability
    dP_dtau_safe = torch.clamp(torch.abs(dP_dtau), min=1e-10, max=1e10)
    g_kappa_safe = torch.clamp(g_kappa, min=1e-10, max=1e10)
    
    log_residual = torch.log10(dP_dtau_safe) - torch.log10(g_kappa_safe)
    
    # Use Huber loss to reduce outlier impact
    delta = 0.5
    abs_residual = torch.abs(log_residual)
    loss = torch.where(
        abs_residual < delta,
        0.5 * abs_residual**2,
        delta * (abs_residual - 0.5 * delta)
    )
    
    return torch.mean(loss)

# =============================================================================
# Training Functions
# =============================================================================
def custom_loss(pred, target, inputs, dataset, model, use_physical_loss=True, 
                use_data_loss=True, physical_loss_type='auto_diff'):
    """
    Flexible loss function that can use only physical loss if specified
    """
    # Initialize loss components
    total_loss = 0.0
    param_losses = {'RHOX': 0.0, 'T': 0.0, 'P': 0.0, 
                   'XNE': 0.0, 'ABROSS': 0.0, 'ACCRAD': 0.0, 'HYDRO': 0.0}
    
    # Setup parameter indices
    param_map = {0: 'RHOX', 1: 'T', 2: 'P', 3: 'XNE', 4: 'ABROSS', 5: 'ACCRAD'}
    param_indices = {v: k for k, v in param_map.items()}
    
    # Add data-driven loss if enabled
    if use_data_loss:
        mse_loss = torch.nn.MSELoss(reduction='none')
        for i, param in param_map.items():
            pred_feature = pred[:, :, i]
            target_feature = target[:, :, i]
            feature_loss = mse_loss(pred_feature, target_feature).mean()
            param_losses[param] = feature_loss
            total_loss += feature_loss
    
    # Add physical loss if enabled
    if use_physical_loss:
        if physical_loss_type == 'auto_diff':
            hydro_loss = hydrostatic_equilibrium_loss(pred, inputs, param_indices, dataset, model)
        else:
            hydro_loss = simplified_hydrostatic_loss(pred, inputs, param_indices, dataset)
        
        param_losses['HYDRO'] = hydro_loss
        total_loss += hydro_loss
    
    # Ensure at least one loss component is used
    if not use_data_loss and not use_physical_loss:
        use_data_loss = True  # Default to data loss if neither is enabled
        total_loss = F.mse_loss(pred, target)
    
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

def validate(model, dataloader, dataset, device, use_physical_loss=False, physical_weight=0.1, 
             physical_loss_type='simplified'):
    """Validation loop"""
    model.eval()
    total_loss = 0
    param_losses = {'RHOX': 0.0, 'T': 0.0, 'P': 0.0, 
                   'XNE': 0.0, 'ABROSS': 0.0, 'ACCRAD': 0.0, 'HYDRO': 0.0}
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # 始终使用simplified方法进行验证，避免auto_diff的不稳定性
            loss, batch_param_losses = custom_loss(
                outputs, targets, inputs, dataset, model,
                use_physical_loss=use_physical_loss,
                physical_weight=physical_weight,
                physical_loss_type='simplified'  # 强制使用simplified方法
            )
            
            total_loss += loss.item()
            for param, param_loss in batch_param_losses.items():
                # 修复：检查param_loss是否为tensor，如果是则调用item()，否则直接使用
                if isinstance(param_loss, torch.Tensor):
                    param_losses[param] += param_loss.item()
                else:
                    param_losses[param] += param_loss
    
    avg_loss = total_loss / len(dataloader)
    avg_param_losses = {k: v / len(dataloader) for k, v in param_losses.items()}
    
    return avg_loss, avg_param_losses

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
def train(model, train_loader, val_loader, dataset, optimizer, device, start_epoch, args, logger):
    """Main training loop with physics-informed loss"""
    best_loss = float('inf')
    patience_counter = 0
    
    # 初始化上一轮的参数损失
    prev_param_losses = {'RHOX': 0.0, 'T': 0.0, 'P': 0.0, 
                         'XNE': 0.0, 'ABROSS': 0.0, 'ACCRAD': 0.0, 'HYDRO': 0.0}
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(os.path.join(args.log_dir, 'tensorboard'))
    
    # Set up gradient clipping
    if args.grad_clip:
        logger.info(f"Gradient clipping enabled with max norm: {args.grad_clip_max_norm}")
    
    logger.info(f"Starting training for {args.epochs} epochs")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        epoch_param_losses = {'RHOX': 0.0, 'T': 0.0, 'P': 0.0, 
                             'XNE': 0.0, 'ABROSS': 0.0, 'ACCRAD': 0.0, 'HYDRO': 0.0}
        
        # Determine if we should use data-driven loss in this epoch
        use_physical_loss = args.physical_loss
        use_data_loss = not args.physical_loss_only
        
        # Calculate physical loss weight with ramp-up
        physical_weight = args.physical_weight
        if use_physical_loss and epoch < args.physical_warmup_epochs + args.physical_rampup_epochs:
            # Gradually increase from 0 to args.physical_weight
            ramp_progress = (epoch - args.physical_warmup_epochs) / args.physical_rampup_epochs
            physical_weight = args.physical_weight * ramp_progress
        
        # 动态调整物理损失权重 (从第二个epoch开始)
        if epoch > start_epoch and use_physical_loss:
            data_loss = sum([v for k, v in prev_param_losses.items() if k != 'HYDRO'])
            hydro_ratio = prev_param_losses['HYDRO'] / (data_loss + 1e-8)
            # 当物理损失过高时降低权重
            adaptive_physical_weight = physical_weight / (1.0 + max(0, hydro_ratio - 1.0) * 2)
            physical_weight = min(physical_weight, adaptive_physical_weight)
            logger.info(f"Adaptive physical weight: {physical_weight:.6f} (ratio: {hydro_ratio:.6f})")
        
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Use the custom_loss function with the model parameter
            loss, param_losses = custom_loss(
                outputs, targets, inputs, dataset, model,
                use_physical_loss=use_physical_loss,
                use_data_loss=use_data_loss,
                physical_loss_type=args.physical_loss_type
            )
            
            # Remove these duplicate calls that are causing the error
            # loss, batch_param_losses = custom_loss(...)
            # loss, batch_param_losses = custom_loss(...)
            
            loss.backward()
            
            # Apply gradient clipping if enabled
            # 在train函数中
            start_time = time.time()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Use the custom_loss function with the model parameter
                loss, param_losses = custom_loss(
                    outputs, targets, inputs, dataset, model,
                    use_physical_loss=use_physical_loss,
                    use_data_loss=use_data_loss,
                    physical_loss_type=args.physical_loss_type
                )
                
                loss.backward()
                
                # Apply gradient clipping if enabled
                if args.grad_clip:
                    clip_norm = max(0.5, 1.0 - (epoch / args.epochs) * 0.5)  # 从1.0逐渐减小到0.5
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                
                optimizer.step()
                
                # Update running losses
                epoch_loss += loss.item()
                for param, param_loss in param_losses.items():
                    # 修复：检查param_loss是否为tensor，如果是则调用item()，否则直接使用
                    if isinstance(param_loss, torch.Tensor):
                        epoch_param_losses[param] += param_loss.item()
                    else:
                        epoch_param_losses[param] += param_loss
                
                # Log batch progress
                if (batch_idx + 1) % args.log_freq == 0:
                    logger.info(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                                f"Loss: {loss.item():.6f}")
        
            # Log batch progress
            if (batch_idx + 1) % args.log_freq == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                           f"Loss: {loss.item():.6f}")
        
        # Calculate average losses
        avg_loss = epoch_loss / len(train_loader)
        avg_param_losses = {k: v / len(train_loader) for k, v in epoch_param_losses.items()}
        prev_param_losses = avg_param_losses.copy()  # 保存当前损失用于下一轮
        
        # Validate
        val_loss, val_param_losses = validate(
            model, val_loader, dataset, device, 
            use_physical_loss=use_physical_loss,
            physical_weight=physical_weight,
            physical_loss_type=args.physical_loss_type
        )
        
        # Log epoch results
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s | "
                   f"Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Log detailed losses
        logger.info(f"Train Losses: " + " | ".join([f"{k}: {v:.6f}" for k, v in avg_param_losses.items()]))
        logger.info(f"Val Losses: " + " | ".join([f"{k}: {v:.6f}" for k, v in val_param_losses.items()]))
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        for param in epoch_param_losses.keys():
            writer.add_scalar(f'Loss/{param}/train', avg_param_losses[param], epoch)
            writer.add_scalar(f'Loss/{param}/val', val_param_losses[param], epoch)
        
        # Save checkpoint if improved
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.output_dir, 'best_model.pt'),
                logger
            )
            logger.info(f"New best model saved with validation loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs (best: {best_loss:.6f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'),
                logger
            )
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    writer.close()
    logger.info("Training completed")
    return model

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
def create_dataloader_from_saved(filepath, batch_size=32, num_workers=4, device='cpu', validation_split=0.1):
    """Create train and validation DataLoaders from a saved dataset."""
    dataset = load_dataset_file(filepath, device)
    
    # Calculate split sizes
    dataset_size = len(dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    # Split dataset
    from torch.utils.data import random_split
    generator = torch.Generator().manual_seed(42)  # For reproducibility
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    return train_loader, val_loader, dataset

def main():
    parser = argparse.ArgumentParser(description='Train Atmospheric Model with Physics-Informed Loss')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Path to the saved dataset file')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', 
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', 
                        help='Directory to save logs')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=500, 
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, 
                        help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=64, 
                        help='Hidden size of the model')
    parser.add_argument('--checkpoint_freq', type=int, default=50, 
                        help='Checkpoint frequency (epochs)')
    parser.add_argument('--log_freq', type=int, default=10, 
                        help='Logging frequency (batches)')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', action='store_true', 
                        help='Use GPU if available')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of worker processes for data loading')
    parser.add_argument('--validation_split', type=float, default=0.1, 
                        help='Validation set split ratio')
    parser.add_argument('--patience', type=int, default=30, 
                        help='Early stopping patience')
    
    # Physical loss parameters
    parser.add_argument('--physical_loss', action='store_true', default=True,
                        help='Enable physical constraint loss')
    parser.add_argument('--physical_loss_type', type=str, default='simplified',
                        choices=['simplified', 'auto_diff'],
                        help='Type of physical loss calculation method')
    parser.add_argument('--physical_weight', type=float, default=100,
                        help='Weight for the physical constraint loss term')
    parser.add_argument('--physical_warmup_epochs', type=int, default=1,
                        help='Number of epochs to train without physical loss')
    parser.add_argument('--physical_rampup_epochs', type=int, default=10,
                        help='Number of epochs to gradually increase physical loss weight')
    parser.add_argument('--physical_loss_only', action='store_true', help='Use only physical loss (no data loss)')
    
    # Gradient clipping parameters
    parser.add_argument('--grad_clip', action='store_true',
                        help='Enable gradient clipping')
    parser.add_argument('--grad_clip_max_norm', type=float, default=1.0,
                        help='Maximum norm for gradient clipping')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(args.log_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Log training configuration
    logger.info("Training configuration:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Load dataset and create dataloaders
    logger.info(f"Loading dataset from {args.dataset}")
    train_loader, val_loader, dataset = create_dataloader_from_saved(
        filepath=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers if device.type == 'cuda' else 0,
        device=device,
        validation_split=args.validation_split
    )
    logger.info(f"Dataset loaded, {len(dataset)} total samples, "
               f"{len(train_loader.dataset)} training, {len(val_loader.dataset)} validation")
    
    # Create model
    model = AtmosphereNet(
        input_size=5,  # Stellar parameters (Teff, logg, [Fe/H], [alpha/Fe], tau)
        hidden_size=args.hidden_size,
        output_size=6,  # RHOX, T, P, XNE, ABROSS, ACCRAD
        depth_points=dataset.max_depth_points
    ).to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # logger.info(f"Using Adam optimizer with learning rate {args.lr}")
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        try:
            _, best_loss = load_checkpoint(model, optimizer, args.resume, device, logger)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}. Starting from scratch.")
           
    
    # Train the model
    train(model, train_loader, val_loader, dataset, optimizer, device, start_epoch, args, logger)

if __name__ == "__main__":
    main()