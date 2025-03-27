#!/usr/bin/env python
# train.py - Training script for Kurucz atmospheric models

import torch
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import time
import logging
import math
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from model import AtmosphereNet, AtmosphereNetMLP, AtmosphereNetMLPtau
from dataset import KuruczDataset, load_dataset_file, create_dataloader_from_saved

# =============================================================================
# Training Functions
# =============================================================================
def custom_loss(pred, target, weights=None):
    """Calculate weighted loss for atmospheric parameters with improved stability"""
    if weights is None:
        # Default weights prioritize temperature and pressure
        weights = {
            'RHOX': 5.0,
            'T': 10.0,
            'P': 5.0,
            'XNE': 1.0,
            'ABROSS': 10.0,
            'ACCRAD': 1.0
        }
    
    # Compute MSE loss for each output feature
    mse_loss = torch.nn.MSELoss(reduction='none')
    
    # Updated param_map to match the new output structure (without TAU)
    param_map = {0: 'RHOX', 1: 'T', 2: 'P', 3: 'XNE', 4: 'ABROSS', 5: 'ACCRAD'}
    
    total_loss = 0
    param_losses = {}
    
    # First check for any NaNs in inputs
    if torch.isnan(pred).any() or torch.isnan(target).any():
        # Log warning
        print("Warning: NaN detected in model predictions or targets")
        # Replace NaN values with zeros
        pred = torch.nan_to_num(pred, nan=0.0)
        target = torch.nan_to_num(target, nan=0.0)
    
    for i, param in param_map.items():
        # Extract the i-th feature
        pred_feature = pred[:, :, i]
        target_feature = target[:, :, i]
        
        # Additional NaN checks per feature
        if torch.isnan(pred_feature).any() or torch.isnan(target_feature).any():
            pred_feature = torch.nan_to_num(pred_feature, nan=0.0)
            target_feature = torch.nan_to_num(target_feature, nan=0.0)
        
        # Add small epsilon for numerical stability
        epsilon = 1e-8
        pred_feature = pred_feature + epsilon
        
        # Compute MSE
        feature_loss = mse_loss(pred_feature, target_feature)
        
        # Clip extremely large loss values
        feature_loss = torch.clamp(feature_loss, max=1e6)
        
        # Average over depth points
        feature_loss = feature_loss.mean(dim=1).mean()
        
        # Apply weight
        weighted_loss = feature_loss * weights[param]
        param_losses[param] = weighted_loss
        total_loss += weighted_loss
    
    # Final safety check
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("Warning: NaN/Inf detected in final loss computation")
        # Return a stable value instead
        return torch.tensor(1.0, device=pred.device, requires_grad=True), param_losses
    
    return total_loss, param_losses

def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath, logger):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }, filepath)
    logger.info(f"Checkpoint saved to {filepath}")
    
def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device, logger, strict=True):
    """Load model checkpoint with validation and error handling"""
    if not os.path.isfile(checkpoint_path):
        logger.warning(f"Checkpoint {checkpoint_path} not found, starting from scratch")
        return 0, float('inf')
    
    try:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Validate checkpoint structure
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
        if not all(key in checkpoint for key in required_keys):
            missing_keys = [key for key in required_keys if key not in checkpoint]
            raise KeyError(f"Checkpoint missing required keys: {missing_keys}")
        
        # Load model weights with validation
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        except RuntimeError as e:
            if 'size mismatch' in str(e) or 'Missing key(s)' in str(e):
                logger.error(f"Model architecture mismatch: {str(e)}")
                raise
            raise
        
        # Load optimizer state
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            logger.error(f"Failed to load optimizer state: {str(e)}")
            raise
        
        # Load scheduler state if it exists in the checkpoint and scheduler is provided
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Scheduler state loaded successfully")
            except ValueError as e:
                logger.error(f"Failed to load scheduler state: {str(e)}")
                raise
        elif scheduler:
            logger.warning("No scheduler state found in checkpoint")
        
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('loss', float('inf'))
        logger.info(f"Successfully loaded checkpoint from epoch {start_epoch}, best loss: {best_loss}")
        
        return start_epoch, best_loss
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        raise

def validate(model, dataloader, device, permute_inputs=False):
    """Validation loop for model evaluation with input permutation support"""
    model.eval()
    total_loss = 0
    param_losses = {'RHOX': 0.0, 'T': 0.0, 'P': 0.0, 
                   'XNE': 0.0, 'ABROSS': 0.0, 'ACCRAD': 0.0}
    
    valid_batches = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply permutation if needed
            if permute_inputs:
                inputs = inputs.permute(0, 2, 1)
            
            # Ensure tensors are contiguous before passing to the model
            inputs = inputs.reshape(inputs.shape)
            targets = targets.reshape(targets.shape)
            
            try:
                outputs = model(inputs)
                
                # Skip batches with NaN outputs
                if torch.isnan(outputs).any():
                    continue
                
                loss, batch_param_losses = custom_loss(outputs, targets)
                
                # Skip batches with NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                total_loss += loss.item()
                for param, param_loss in batch_param_losses.items():
                    param_losses[param] += param_loss.item()
                
                valid_batches += 1
                
            except RuntimeError:
                # Skip any batches that cause runtime errors
                continue
    
    # Avoid division by zero if all batches were skipped
    if valid_batches == 0:
        return float('inf'), {k: float('inf') for k in param_losses}
    
    avg_loss = total_loss / valid_batches
    avg_param_losses = {k: v / valid_batches for k, v in param_losses.items()}
    
    return avg_loss, avg_param_losses

# =============================================================================
# Learning Rate Schedulers
# =============================================================================
def get_scheduler(scheduler_type, optimizer, args, logger):
    """Create a learning rate scheduler based on specified type"""
    if scheduler_type == 'step':
        # Step decay: lr = lr * gamma^(epoch // step_size)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
        logger.info(f"Created StepLR scheduler with step_size={args.lr_step_size}, gamma={args.lr_gamma}")
    elif scheduler_type == 'multistep':
        # Multi-step decay: lr decays by gamma at specified milestones
        milestones = [int(m) for m in args.lr_milestones.split(',')]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=args.lr_gamma
        )
        logger.info(f"Created MultiStepLR scheduler with milestones={milestones}, gamma={args.lr_gamma}")
    elif scheduler_type == 'exponential':
        # Exponential decay: lr = lr * gamma^epoch
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=args.lr_gamma
        )
        logger.info(f"Created ExponentialLR scheduler with gamma={args.lr_gamma}")
    elif scheduler_type == 'cosine':
        # Cosine annealing: cosine function from initial lr to eta_min
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr_min
        )
        logger.info(f"Created CosineAnnealingLR scheduler with T_max={args.epochs}, eta_min={args.lr_min}")
    elif scheduler_type == 'plateau':
        # Reduce on plateau: reduce lr when metric plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_gamma,
            patience=args.lr_patience,
            verbose=True,
            min_lr=args.lr_min
        )
        logger.info(f"Created ReduceLROnPlateau scheduler with mode='min', factor={args.lr_gamma}, patience={args.lr_patience}, min_lr={args.lr_min}")
    elif scheduler_type == 'cyclic':
        # Cyclic LR: cycles between base_lr and max_lr
        step_size_up = int(args.epochs * len(args.train_loader) / args.lr_cycles / 2)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.lr_min,
            max_lr=args.lr,
            step_size_up=step_size_up,
            mode=args.lr_cycle_mode,
            cycle_momentum=False
        )
        logger.info(f"Created CyclicLR scheduler with base_lr={args.lr_min}, max_lr={args.lr}, step_size_up={step_size_up}, mode={args.lr_cycle_mode}")
    elif scheduler_type == 'onecycle':
        # One Cycle LR: one cycle with cosine annealing
        steps_per_epoch = len(args.train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=args.lr_warmup_pct,
            anneal_strategy='cos',
            div_factor=args.lr_div_factor,
            final_div_factor=args.lr_final_div_factor
        )
        logger.info(f"Created OneCycleLR scheduler with max_lr={args.lr}, epochs={args.epochs}, steps_per_epoch={steps_per_epoch}, pct_start={args.lr_warmup_pct}, div_factor={args.lr_div_factor}, final_div_factor={args.lr_final_div_factor}")
    else:
        # No scheduler
        scheduler = None
        
    # Add debug info for each scheduler type
    if scheduler is None:
        logger.info("No scheduler selected, using constant learning rate")
    
    return scheduler

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
def train(model, train_loader, val_loader, optimizer, scheduler, device, start_epoch, args, logger):
    """Main training loop with validation and learning rate scheduling"""
    best_loss = float('inf')
    patience_counter = 0
    
    # Set train_loader attribute for scheduler creation
    args.train_loader = train_loader
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(os.path.join(args.log_dir, 'tensorboard'))
    
    # Log model graph - ensure correct tensor shape
    example_batch = next(iter(train_loader))
    example_input = example_batch[0][:1].to(device)
    
    # Check if the model expects a permuted tensor and apply if needed
    try:
        # Test the shape that's used in the training loop
        _ = model(example_input)
        permute_inputs = False
        logger.info("Model accepts unpermuted inputs")
    except Exception:
        try:
            # Try with permuted input
            _ = model(example_input.permute(0, 2, 1))
            permute_inputs = True
            logger.info("Model requires permuted inputs (0, 2, 1)")
        except Exception as e:
            logger.error(f"Error determining input shape: {str(e)}")
            raise
    
    logger.info(f"Starting training for {args.epochs} epochs")
    
    # Keep track of NaN occurrences
    nan_epochs = 0
    
    for epoch in range(start_epoch, args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        nan_batches = 0
        
        # Initialize per-parameter loss tracking
        param_running_losses = {'RHOX': 0.0, 'T': 0.0, 'P': 0.0, 
                               'XNE': 0.0, 'ABROSS': 0.0, 'ACCRAD': 0.0}
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply permutation if needed
            if permute_inputs:
                inputs = inputs.permute(0, 2, 1)
            
            # Ensure tensors are contiguous before passing to the model
            inputs = inputs.reshape(inputs.shape)
            targets = targets.reshape(targets.shape)
            
            # Forward pass and loss calculation
            optimizer.zero_grad()
            
            # Catch any runtime errors in the forward pass
            try:
                outputs = model(inputs)
                
                # Check for NaN values before loss calculation
                if torch.isnan(outputs).any():
                    logger.warning(f"NaN detected in outputs at epoch {epoch+1}, batch {batch_idx}")
                    nan_batches += 1
                    # Skip backprop for this batch
                    continue
                
                loss, param_losses = custom_loss(outputs, targets)
                
                # Check if loss is NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss at epoch {epoch+1}, batch {batch_idx}")
                    nan_batches += 1
                    continue
                
                # Backward pass and optimize
                loss.backward()
                
                # Add gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Step for batch-based schedulers
                if args.scheduler in ['cyclic', 'onecycle']:
                    scheduler.step()
                
                # Update statistics
                running_loss += loss.item()
                for param, param_loss in param_losses.items():
                    param_running_losses[param] += param_loss.item()
                
            except RuntimeError as e:
                logger.error(f"Runtime error at epoch {epoch+1}, batch {batch_idx}: {str(e)}")
                nan_batches += 1
                continue
        
        # Check if we had too many NaN batches
        if nan_batches > len(train_loader) * 0.5:  # If more than 50% batches had NaNs
            logger.warning(f"Too many NaN batches ({nan_batches}/{len(train_loader)}) in epoch {epoch+1}")
            nan_epochs += 1
            
            # If we have consecutive NaN epochs, reduce learning rate
            if nan_epochs >= 2:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
                logger.warning(f"Reduced learning rate to {optimizer.param_groups[0]['lr']} due to NaNs")
                nan_epochs = 0
                
                # If learning rate is too small, stop training
                if optimizer.param_groups[0]['lr'] < 1e-8:
                    logger.error("Learning rate too small, stopping training")
                    break
            
            continue  # Skip to next epoch
        else:
            nan_epochs = 0  # Reset counter if we had a good epoch
        
        # Calculate training epoch statistics
        processed_batches = len(train_loader) - nan_batches
        if processed_batches > 0:
            train_epoch_loss = running_loss / processed_batches
            train_avg_param_losses = {k: v / processed_batches for k, v in param_running_losses.items()}
        else:
            train_epoch_loss = float('nan')
            train_avg_param_losses = {k: float('nan') for k in param_running_losses}
        
        epoch_time = time.time() - epoch_start_time
        train_param_loss_str = ', '.join([f"{k}: {v:.6f}" for k, v in train_avg_param_losses.items()])
        
        # Validation phase
        val_loss, val_param_losses = validate(model, val_loader, device, permute_inputs)
        val_param_loss_str = ', '.join([f"{k}: {v:.6f}" for k, v in val_param_losses.items()])
        
        # Step for epoch-based schedulers
        current_lr = optimizer.param_groups[0]['lr']
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        elif args.scheduler not in ['none', 'cyclic', 'onecycle']:
            scheduler.step()
        
        # Get updated learning rate
        new_lr = optimizer.param_groups[0]['lr']
        if current_lr != new_lr:
            logger.info(f'Learning rate changed from {current_lr:.6e} to {new_lr:.6e}')
        
        # Logging
        logger.info(f'Epoch {epoch+1}, Train Loss: {train_epoch_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Log epoch metrics to TensorBoard
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Time/epoch', epoch_time, epoch)
        writer.add_scalar('LearningRate/epoch', optimizer.param_groups[0]['lr'], epoch)
        for param, loss_val in train_avg_param_losses.items():
            writer.add_scalar(f'Loss_train/{param}', loss_val, epoch)
        for param, loss_val in val_param_losses.items():
            writer.add_scalar(f'Loss_val/{param}', loss_val, epoch)
        
        # Save checkpoint if validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            logger.info(f'New best validation loss: {best_loss:.6f}')
            save_checkpoint(model, optimizer, scheduler, epoch+1, val_loss, 
                           os.path.join(args.output_dir, 'best_model.pt'), logger)
        else:
            patience_counter += 1
            
        # Early stopping check
        if args.patience > 0 and patience_counter >= args.patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        if (epoch + 1) % args.checkpoint_freq == 0:
            save_checkpoint(model, optimizer, scheduler, epoch+1, val_loss,
                           os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'), logger)
    
    # Save final model
    save_checkpoint(model, optimizer, scheduler, args.epochs, train_epoch_loss,
                   os.path.join(args.output_dir, 'final_model.pt'), logger)
    
    logger.info("Training completed!")
    writer.close()
    return best_loss


# =============================================================================
# Main Function
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train Kurucz Atmospheric Model')
    # Dataset and model parameters
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Path to the saved dataset file')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the model')
    parser.add_argument('--model', type=str, default='tau', choices=['mlp', 'atm', 'conv', 'tau'], help='Model architecture to use')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate') # Reduced from 1e-4
    parser.add_argument('--checkpoint_freq', type=int, default=100, help='Checkpoint frequency (epochs)')
    parser.add_argument('--log_freq', type=int, default=10, help='Logging frequency (batches)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--validation_split', type=float, default=0.1, help='Validation set split ratio')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    
    # Hardware parameters
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker processes for data loading')
    
    # Learning rate scheduler parameters
    parser.add_argument('--scheduler', type=str, default='plateau', 
                        choices=['none', 'step', 'multistep', 'exponential', 'cosine', 'plateau', 'cyclic', 'onecycle'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_gamma', type=float, default=0.1, 
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--lr_step_size', type=int, default=100, 
                        help='Period of learning rate decay (epochs)')
    parser.add_argument('--lr_milestones', type=str, default='200,400,600', 
                        help='Epochs at which to decay learning rate (comma-separated)')
    parser.add_argument('--lr_min', type=float, default=1e-6, 
                        help='Minimum learning rate')
    parser.add_argument('--lr_patience', type=int, default=5, 
                        help='Epochs with no improvement after which learning rate will be reduced')
    parser.add_argument('--lr_cycles', type=int, default=4, 
                        help='Number of cycles for cyclic LR scheduler')
    parser.add_argument('--lr_cycle_mode', type=str, default='triangular2', 
                        choices=['triangular', 'triangular2', 'exp_range'],
                        help='Mode for cyclic LR scheduler')
    parser.add_argument('--lr_warmup_pct', type=float, default=0.3, 
                        help='Percentage of total iterations for warmup in OneCycleLR')
    parser.add_argument('--lr_div_factor', type=float, default=25.0, 
                        help='Initial learning rate division factor for OneCycleLR')
    parser.add_argument('--lr_final_div_factor', type=float, default=1e4, 
                        help='Final learning rate division factor for OneCycleLR')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                       help='Optimizer to use (adam or sgd)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(args.log_dir)
    
    # Set device
    if args.gpu:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 
                             ('mps' if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and 
                              torch.backends.mps.is_available() else 'cpu'))
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Log training configuration
    logger.info("Training configuration:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Load dataset and create dataloaders
    logger.info(f"Loading dataset from {args.dataset}")
    try:
        train_loader, val_loader, dataset = create_dataloader_from_saved(
            filepath=args.dataset,
            batch_size=args.batch_size,
            num_workers=0,
            device=device,
            validation_split=args.validation_split
        )
        logger.info(f"Dataset loaded, {len(dataset)} total samples, "
                  f"{len(train_loader.dataset)} training, {len(val_loader.dataset)} validation")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise
    
    # Get the input shape from the first batch
    sample_batch = next(iter(train_loader))
    input_shape = sample_batch[0].shape
    logger.info(f"Input shape: {input_shape}")
    
    # The input now includes tau at each depth point (5 features per depth point)
    # The last dimension of the input tensor is the number of features per depth point
    input_features_per_point = input_shape[-1]
    logger.info(f"Input features per point: {input_features_per_point}")

    # Create model based on selected architecture
    if args.model == 'atm':
        model = AtmosphereNet(
            input_size=input_features_per_point,
            hidden_size=args.hidden_size,
            output_size=6,
            depth_points=dataset.max_depth_points
        ).to(device)
        logger.info("Created AtmosphereNet model")

    elif args.model == 'mlp':
        model = AtmosphereNetMLP(
            input_size=input_features_per_point,
            hidden_size=args.hidden_size,
            output_size=6,
            depth_points=dataset.max_depth_points,
        ).to(device)
        logger.info("Created AtmosphereNetMLP model")
        
    else:
        model = AtmosphereNetMLPtau(
            stellar_embed_dim=128, tau_embed_dim=64
            ).to(device)
        logger.info("Created AtmosphereNetMLPtau model")
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        logger.info(f"Using Adam optimizer with lr={args.lr}, weight_decay={args.weight_decay}")
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
        logger.info(f"Using SGD optimizer with lr={args.lr}, momentum=0.9, weight_decay={args.weight_decay}")
    
    # Create scheduler
    scheduler = get_scheduler(args.scheduler, optimizer, args, logger)
    if scheduler:
        logger.info(f"Using {args.scheduler} learning rate scheduler")
    else:
        logger.info("No learning rate scheduler selected")
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        try:
            _, best_loss = load_checkpoint(model, optimizer, scheduler, args.resume, device, logger)
            logger.info(f"Resuming training from epoch {start_epoch} with best loss {best_loss}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}. Starting from scratch.")
            start_epoch = 0
    
    # Train the model
    try:
        train(model, train_loader, val_loader, optimizer, scheduler, device, start_epoch, args, logger)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save current model state
        save_checkpoint(model, optimizer, scheduler, start_epoch, float('inf'),
                      os.path.join(args.output_dir, 'interrupted.pt'), logger)
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        # Save current model state
        save_checkpoint(model, optimizer, scheduler, start_epoch, float('inf'),
                      os.path.join(args.output_dir, 'error.pt'), logger)
        raise

if __name__ == "__main__":
    main()
    # Example usage:
    # python train.py --dataset data/kurucz_dataset.pt --gpu --scheduler cosine --epochs 500 --lr 0.001 --batch_size 128