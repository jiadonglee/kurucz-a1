#!/usr/bin/env python
# train_physics_bo.py - Training script for Kurucz atmospheric models with physics loss optimization

import torch
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import time
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from model import AtmosphereNet, AtmosphereNetMLP, AtmosphereNetMLPtau
from physics import hydro_equilibrium_loss
from utils import get_scheduler, setup_logger
from dataset import KuruczDataset, load_dataset_file, create_dataloader_from_saved

# --- Bayesian Optimization Imports ---
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import numpy as np

# =============================================================================
# Loss Function
# =============================================================================
def custom_loss(pred, target, dataset, model, inputs, use_physics=True, physics_weight=1e-2, weights=None, is_training=True):
    """Calculate weighted loss including physics-informed components"""
    # Calculate data-driven loss component
    if weights is None:
        # Default weights prioritize temperature and pressure
        weights = {
            'RHOX': 5.0,
            'T': 5.0,
            'P': 5.0,
            'XNE': 1.0,
            'ABROSS': 5.0,
            'ACCRAD': 1.0
        }

    # Compute MSE loss for each output feature
    mse_loss = torch.nn.MSELoss(reduction='none')

    param_map = {0: 'RHOX', 1: 'T', 2: 'P', 3: 'XNE', 4: 'ABROSS', 5: 'ACCRAD'}

    total_data_loss = 0
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
        total_data_loss += weighted_loss

    # Calculate physics-informed loss component
    if use_physics:
        physics_loss_val = hydro_equilibrium_loss(pred, inputs, dataset, model)
        param_losses['physics'] = physics_loss_val
        
        # Calculate total loss with physics component
        total_loss = (1.0 - physics_weight) * total_data_loss + physics_weight * physics_loss_val
    else:
        total_loss = total_data_loss
        param_losses['physics'] = torch.tensor(0.0, device=pred.device)

    return total_loss, param_losses

# =============================================================================
# Checkpoint/Validation Functions
# =============================================================================
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

def validate(model, dataloader, device, dataset, current_physics_weight, logger=None, physics_only=False):
    """Validation loop for model evaluation - can return only physics loss if requested"""
    model.eval()
    total_loss = 0
    param_losses = {'RHOX': 0.0, 'T': 0.0, 'P': 0.0,
                   'XNE': 0.0, 'ABROSS': 0.0, 'ACCRAD': 0.0, 'physics': 0.0}
    physics_losses = []
    batch_sizes = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Calculate loss (full or physics component)
            loss, batch_param_losses = custom_loss(outputs, targets, dataset, model, inputs, True, current_physics_weight, None, False)

            # Track physics loss specifically
            if 'physics' in batch_param_losses:
                if isinstance(batch_param_losses['physics'], torch.Tensor):
                    physics_losses.append(batch_param_losses['physics'].item())
                else:
                    physics_losses.append(batch_param_losses['physics'])
            
            batch_sizes.append(inputs.size(0))

            # Track full loss components
            total_loss += loss.item()
            for param, param_loss in batch_param_losses.items():
                if isinstance(param_loss, torch.Tensor):
                    param_losses[param] += param_loss.item()
                else:
                    param_losses[param] += param_loss

    # Calculate average losses
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_param_losses = {k: v / len(dataloader) if len(dataloader) > 0 else 0 for k, v in param_losses.items()}
    avg_physics_loss = sum(physics_losses) / len(physics_losses) if physics_losses else 0

    # Log validation statistics if needed
    if len(physics_losses) > 0 and logger is not None and logger.level <= logging.DEBUG:
        min_physics = min(physics_losses)
        max_physics = max(physics_losses)
        mean_physics = sum(physics_losses) / len(physics_losses)
        logger.debug(f"Validation physics loss stats - Min: {min_physics:.6f}, Max: {max_physics:.6f}, Mean: {mean_physics:.6f}")
        logger.debug(f"Validation batch sizes - Min: {min(batch_sizes)}, Max: {max(batch_sizes)}, Mean: {sum(batch_sizes)/len(batch_sizes):.2f}")

    if physics_only:
        return avg_physics_loss
    else:
        return avg_loss, avg_param_losses

# =============================================================================
# Modified Training Function for Physics Loss Optimization
# =============================================================================
def train_for_bo_physics_only(model, train_loader, val_loader, optimizer, scheduler, device,
                 start_epoch, num_epochs, current_physics_weight, patience,
                 output_dir, checkpoint_freq, logger, dataset):
    """Training loop that optimizes and returns the physics loss component."""
    best_physics_loss = float('inf')
    patience_counter = 0
    
    logger.info(f"Starting BO training trial for {num_epochs} epochs with physics_weight={current_physics_weight:.6e}")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_start_time = time.time()
        
        # Tracking for training losses
        physics_losses_train = []
        data_losses_train = []
        
        # Training loop
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Get full loss and component losses
            loss, param_losses = custom_loss(outputs, targets, dataset, model, inputs, 
                                           True, current_physics_weight, None, True)
            
            # Track physics loss specifically
            if 'physics' in param_losses:
                if isinstance(param_losses['physics'], torch.Tensor):
                    physics_losses_train.append(param_losses['physics'].item())
                else:
                    physics_losses_train.append(param_losses['physics'])
            
            # Calculate data loss (total minus physics)
            data_loss = sum(v for k, v in param_losses.items() if k != 'physics')
            if isinstance(data_loss, torch.Tensor):
                data_losses_train.append(data_loss.item())
            else:
                data_losses_train.append(data_loss)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Step scheduler if needed
            if scheduler is not None and isinstance(scheduler, (torch.optim.lr_scheduler.CyclicLR, torch.optim.lr_scheduler.OneCycleLR)):
                scheduler.step()
        
        # Calculate average training losses
        train_physics_loss = sum(physics_losses_train) / len(physics_losses_train) if physics_losses_train else 0
        train_data_loss = sum(data_losses_train) / len(data_losses_train) if data_losses_train else 0
        epoch_time = time.time() - epoch_start_time
        
        # Validation phase - get physics loss only
        val_physics_loss = validate(model, val_loader, device, dataset, current_physics_weight, logger, physics_only=True)
        
        # Adjust learning rate with scheduler if using validation loss
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Use physics loss for scheduling
                scheduler.step(val_physics_loss)
            elif not isinstance(scheduler, (torch.optim.lr_scheduler.CyclicLR, torch.optim.lr_scheduler.OneCycleLR)):
                scheduler.step()
        
        # Log progress
        logger.debug(f'Epoch {epoch+1}, Physics Loss: Train={train_physics_loss:.6f}, Val={val_physics_loss:.6f}')
        logger.debug(f'Data Loss: Train={train_data_loss:.6f}, Time: {epoch_time:.2f}s')
        
        # Track best validation physics loss
        if val_physics_loss < best_physics_loss:
            best_physics_loss = val_physics_loss
            patience_counter = 0
            logger.debug(f'New best physics loss: {best_physics_loss:.6f}')
            
            # Optionally save best model for this trial
            # save_checkpoint(model, optimizer, scheduler, epoch+1, val_physics_loss,
            #               os.path.join(output_dir, f'best_physics_model_w{current_physics_weight:.6e}.pt'), logger)
        else:
            patience_counter += 1
            
        # Early stopping based on physics loss improvement
        if patience_counter >= patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs for physics weight {current_physics_weight:.6e}')
            break
            
    logger.info(f"BO trial finished for weight {current_physics_weight:.6e}. Best validation physics loss: {best_physics_loss:.6f}")
    
    # Ensure a valid number is returned
    if best_physics_loss == float('inf') or np.isnan(best_physics_loss):
        return 1e10  # Return a large but finite number if optimization failed
    return best_physics_loss

# =============================================================================
# Bayesian Optimization Setup
# =============================================================================
# Define search space
search_space = [
    Real(1e-4, 1e1, prior='log-uniform', name='physics_weight')
    # Add other hyperparameters if optimizing more than physics weight
]

# Global variables for BO objective function
GLOBAL_ARGS = None
GLOBAL_LOGGER = None
GLOBAL_DEVICE = None
GLOBAL_TRAIN_LOADER = None
GLOBAL_VAL_LOADER = None
GLOBAL_DATASET = None

@use_named_args(search_space)
def objective(**params):
    """Objective function for Bayesian Optimization - focused on physics loss."""
    current_physics_weight = params['physics_weight']
    
    # Access globals
    args = GLOBAL_ARGS
    logger = GLOBAL_LOGGER
    device = GLOBAL_DEVICE
    train_loader = GLOBAL_TRAIN_LOADER
    val_loader = GLOBAL_VAL_LOADER
    dataset = GLOBAL_DATASET

    logger.info(f"\n--- Starting BO Trial with physics_weight={current_physics_weight:.6e} ---")
    
    # Create new model instance for each trial
    model = AtmosphereNetMLPtau(
        stellar_embed_dim=args.hidden_size, 
        tau_embed_dim=64
    ).to(device)
    logger.debug(f"BO Trial: Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Setup scheduler (simpler for BO trials)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 
        patience=args.lr_patience // 2,
        factor=args.lr_gamma
    )
    
    # Train model focusing on physics loss
    physics_loss = train_for_bo_physics_only(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        start_epoch=0,
        num_epochs=args.bo_epochs,
        current_physics_weight=current_physics_weight,
        patience=args.bo_patience,
        output_dir=args.output_dir,
        checkpoint_freq=args.checkpoint_freq,
        logger=logger,
        dataset=dataset
    )
    
    # Clean up to avoid memory leaks
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"--- BO Trial Finished --- physics_weight={current_physics_weight:.6e}, Physics Loss: {physics_loss:.6f}\n")
    
    return physics_loss  # BO will minimize this value

# =============================================================================
# Main Function 
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train Kurucz Atmospheric Model with Physics Loss Optimization')
    
    # Dataset and model parameters
    parser.add_argument('--dataset', type=str, required=True, help='Path to the saved dataset file')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_physics', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs_physics', help='Directory to save logs')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for final training run')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--checkpoint_freq', type=int, default=100, help='Checkpoint frequency (epochs)')
    parser.add_argument('--log_freq', type=int, default=10, help='Logging frequency (batches)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume final training from')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience for final training')
    parser.add_argument('--validation_split', type=float, default=0.1, help='Validation set split ratio')
    
    # Hardware parameters
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker processes for data loading')
    
    # Bayesian Optimization parameters
    parser.add_argument('--run_bo', action='store_true', help='Run Bayesian Optimization to find optimal physics_weight')
    parser.add_argument('--bo_calls', type=int, default=20, help='Number of Bayesian Optimization trials')
    parser.add_argument('--bo_epochs', type=int, default=50, help='Number of epochs per BO trial')
    parser.add_argument('--bo_patience', type=int, default=10, help='Early stopping patience for BO trials')
    parser.add_argument('--final_physics_weight', type=float, default=1e-3, help='Physics weight to use if not running BO')
    
    # Learning rate scheduler parameters
    parser.add_argument('--scheduler', type=str, default='plateau', 
                      choices=['none', 'step', 'multistep', 'exponential', 'cosine', 'plateau', 'cyclic', 'onecycle'],
                      help='Learning rate scheduler type')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--lr_step_size', type=int, default=100, help='Period of learning rate decay (epochs)')
    parser.add_argument('--lr_milestones', type=str, default='200,400,600', help='Epochs at which to decay learning rate (comma-separated)')
    parser.add_argument('--lr_min', type=float, default=1e-7, help='Minimum learning rate')
    parser.add_argument('--lr_patience', type=int, default=5, help='Epochs with no improvement after which learning rate will be reduced')
    parser.add_argument('--lr_cycles', type=int, default=4, help='Number of cycles for cyclic LR scheduler')
    parser.add_argument('--lr_cycle_mode', type=str, default='triangular2', 
                      choices=['triangular', 'triangular2', 'exp_range'],
                      help='Mode for cyclic LR scheduler')
    parser.add_argument('--lr_warmup_pct', type=float, default=0.3, help='Percentage of total iterations for warmup in OneCycleLR')
    parser.add_argument('--lr_div_factor', type=float, default=25.0, help='Initial learning rate division factor for OneCycleLR')
    parser.add_argument('--lr_final_div_factor', type=float, default=1e4, help='Final learning rate division factor for OneCycleLR')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(args.log_dir)
    
    # Set device
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
            logger.warning("GPU requested but CUDA/MPS not available, using CPU.")
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Log configuration
    logger.info("Configuration:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    train_loader, val_loader, dataset = create_dataloader_from_saved(
        filepath=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        validation_split=args.validation_split
    )
    logger.info(f"Dataset loaded, {len(dataset)} total samples, "
               f"{len(train_loader.dataset)} training, {len(val_loader.dataset)} validation")
    
    # Set up globals for BO
    global GLOBAL_ARGS, GLOBAL_LOGGER, GLOBAL_DEVICE
    global GLOBAL_TRAIN_LOADER, GLOBAL_VAL_LOADER, GLOBAL_DATASET
    GLOBAL_ARGS = args
    GLOBAL_LOGGER = logger
    GLOBAL_DEVICE = device
    GLOBAL_TRAIN_LOADER = train_loader
    GLOBAL_VAL_LOADER = val_loader
    GLOBAL_DATASET = dataset
    
    best_physics_weight = args.final_physics_weight  # Default value
    
    # Run Bayesian Optimization if requested
    if args.run_bo:
        logger.info(f"Starting Bayesian Optimization for physics_weight ({args.bo_calls} calls)")
        
        # Set random seed for reproducibility
        random_seed = 42
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Run BO
        bo_result = gp_minimize(
            func=objective,           # Function to minimize
            dimensions=search_space,  # Parameters to search
            acq_func="EI",            # Acquisition function (Expected Improvement)
            n_calls=args.bo_calls,    # Number of evaluations
            random_state=random_seed,
            verbose=True,
            n_initial_points=5        # Number of initial random points
        )
        
        best_physics_weight = bo_result.x[0]  # Extract optimal physics_weight
        best_physics_loss = bo_result.fun     # Best physics loss value found
        
        logger.info("\n--- Bayesian Optimization Results ---")
        logger.info(f"Best physics_weight found: {best_physics_weight:.6e}")
        logger.info(f"Best physics loss achieved: {best_physics_loss:.6f}")
        
        # Save BO results
        try:
            from skopt.plots import plot_convergence, plot_objective
            import matplotlib.pyplot as plt
            
            # Create plots directory
            plots_dir = os.path.join(args.log_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Save convergence plot
            plt.figure(figsize=(10, 6))
            plot_convergence(bo_result)
            plt.savefig(os.path.join(plots_dir, "bo_convergence.png"))
            plt.close()
            
            # Save objective plot
            plt.figure(figsize=(10, 6))
            plot_objective(bo_result)
            plt.savefig(os.path.join(plots_dir, "bo_objective.png"))
            plt.close()
            
            logger.info(f"BO plots saved to {plots_dir}")
        except Exception as e:
            logger.warning(f"Failed to save BO plots: {e}")
    
    # Train final model with best physics weight
    logger.info(f"\n--- Starting Final Training Run ---")
    logger.info(f"Using physics_weight: {best_physics_weight:.6e}")
    
    # Create final model
    final_model = AtmosphereNetMLPtau(
        stellar_embed_dim=args.hidden_size,
        tau_embed_dim=64
    ).to(device)
    logger.info(f"Final model created with {sum(p.numel() for p in final_model.parameters())} parameters")
    
    # Create optimizer and scheduler for final training
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=args.lr)
    final_scheduler = get_scheduler(args.scheduler, final_optimizer, args, logger)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        try:
            start_epoch, _ = load_checkpoint(final_model, final_optimizer, final_scheduler, args.resume, device, logger)
            logger.info(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            start_epoch = 0
    
    # Set up TensorBoard writer
    writer = SummaryWriter(os.path.join(args.log_dir, 'tensorboard_final'))
    
    # Final training loop
    best_loss = float('inf')
    best_physics_loss_final = float('inf')
    patience_counter = 0
    
    logger.info(f"Starting final training for {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        final_model.train()
        running_loss = 0.0
        running_physics_loss = 0.0
        epoch_start_time = time.time()
        
        # Training loop
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            final_optimizer.zero_grad()
            outputs = final_model(inputs)
            
            # Full loss with best physics weight
            loss, param_losses = custom_loss(outputs, targets, dataset, final_model, inputs, 
                                           True, best_physics_weight, None, True)
            
            # Track physics loss component
            if 'physics' in param_losses:
                if isinstance(param_losses['physics'], torch.Tensor):
                    running_physics_loss += param_losses['physics'].item()
                else:
                    running_physics_loss += param_losses['physics']
            
            loss.backward()
            final_optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Step batch-based schedulers
            if args.scheduler in ['cyclic', 'onecycle'] and final_scheduler:
                final_scheduler.step()
        
        # Calculate epoch losses
        train_loss = running_loss / len(train_loader)
        train_physics_loss = running_physics_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        # Validation
        val_loss, val_param_losses = validate(final_model, val_loader, device, dataset, best_physics_weight, logger)
        val_physics_loss = val_param_losses['physics']
        
        # Step epoch-based schedulers
        if final_scheduler:
            if args.scheduler == 'plateau':
                # Use full loss or physics loss for scheduler - your choice
                final_scheduler.step(val_loss)  # Using full loss
                # OR: final_scheduler.step(val_physics_loss)  # Using physics loss
            elif args.scheduler not in ['none', 'cyclic', 'onecycle']:
                final_scheduler.step()
        
        # Log progress
        logger.info(f'Epoch {epoch+1}, Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        logger.info(f'Physics Loss: Train={train_physics_loss:.6f}, Val={val_physics_loss:.6f}')
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('PhysicsLoss/train', train_physics_loss, epoch)
        writer.add_scalar('PhysicsLoss/val', val_physics_loss, epoch)
        writer.add_scalar('Time/epoch', epoch_time, epoch)
        writer.add_scalar('LearningRate', final_optimizer.param_groups[0]['lr'], epoch)
        
        # Save if best model (by validation loss)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            logger.info(f'New best validation loss: {best_loss:.6f}')
            save_checkpoint(final_model, final_optimizer, final_scheduler, epoch+1, val_loss,
                           os.path.join(args.output_dir, 'best_model.pt'), logger)
        else:
            patience_counter += 1
        
        # Also track best physics loss
        if val_physics_loss < best_physics_loss_final:
            best_physics_loss_final = val_physics_loss
            logger.info(f'New best physics loss: {best_physics_loss_final:.6f}')
            save_checkpoint(final_model, final_optimizer, final_scheduler, epoch+1, val_physics_loss,
                           os.path.join(args.output_dir, 'best_physics_model.pt'), logger)
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        # Regular checkpoints
        if (epoch + 1) % args.checkpoint_freq == 0:
            save_checkpoint(final_model, final_optimizer, final_scheduler, epoch+1, val_loss,
                           os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'), logger)
    
    # Save final model
    save_checkpoint(final_model, final_optimizer, final_scheduler, epoch+1, val_loss,
                   os.path.join(args.output_dir, 'final_model.pt'), logger)
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_loss:.6f}")
    logger.info(f"Best physics loss: {best_physics_loss_final:.6f}")
    writer.close()
    
    return best_loss, best_physics_loss_final

if __name__ == "__main__":
    main()
    
    # Example command:
    # python train_bo.py --dataset data/kurucz_v4.pt --run_bo --bo_calls 25 --bo_epochs 50 --gpu --output_dir ./physics_checkpoints --log_dir ./physics_logs --batch_size 256 --lr 5e-4 --epochs 500 --patience 30