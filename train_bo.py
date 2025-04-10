#!/usr/bin/env python
# train_bo.py - Training script for Kurucz atmospheric models

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
import numpy as np # Often needed with skopt

# =============================================================================
# Loss Function (Unchanged from your original)
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

    # Add physics-informed loss component if requested
    if use_physics:
            param_indices = {param: i for i, param in param_map.items()}
            physics_loss_val = hydro_equilibrium_loss(pred, inputs, dataset, model) # Renamed to avoid conflict
            total_loss = (1.0 - physics_weight) * total_data_loss + physics_weight * physics_loss_val
            param_losses['physics'] = physics_loss_val
    else:
        total_loss = total_data_loss
        param_losses['physics'] = torch.tensor(0.0) # Ensure physics key exists even if not used

    return total_loss, param_losses

# =============================================================================
# Checkpoint/Validation Functions (Mostly Unchanged)
# =============================================================================
def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath, logger):
    """Save model checkpoint"""
    # During BO, maybe skip saving intermediate checkpoints unless needed
    # For simplicity here, we keep it, but you might disable it during optimization runs
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
    # ... (Keep your original load_checkpoint function) ...
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

def validate(model, dataloader, device, dataset, current_physics_weight, logger=None): # Pass physics_weight
    """Validation loop for model evaluation"""
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

            # Use the current physics weight being tested
            loss, batch_param_losses = custom_loss(outputs, targets, dataset, model, inputs, True, current_physics_weight, None, False)

            physics_losses.append(batch_param_losses['physics'].item())
            batch_sizes.append(inputs.size(0))

            total_loss += loss.item()
            for param, param_loss in batch_param_losses.items():
                 # Handle potential tensor vs float issues if physics loss is 0
                if isinstance(param_loss, torch.Tensor):
                    param_losses[param] += param_loss.item()
                else:
                    param_losses[param] += param_loss # Assume it's already a float


    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_param_losses = {k: v / len(dataloader) if len(dataloader) > 0 else 0 for k, v in param_losses.items()}

    if len(physics_losses) > 0 and logger is not None:
        min_physics = min(physics_losses)
        max_physics = max(physics_losses)
        mean_physics = sum(physics_losses) / len(physics_losses)
        logger.debug(f"Validation physics loss stats - Min: {min_physics:.6f}, Max: {max_physics:.6f}, Mean: {mean_physics:.6f}")
        logger.debug(f"Validation batch sizes - Min: {min(batch_sizes)}, Max: {max(batch_sizes)}, Mean: {sum(batch_sizes)/len(batch_sizes):.2f}")

    return avg_loss, avg_param_losses

# =============================================================================
# Modified Training Function
# =============================================================================
# =============================================================================
# Modified Training Function (Returns last train physics loss)
# =============================================================================
def train_for_bo(model, train_loader, val_loader, optimizer, scheduler, device,
                 start_epoch, num_epochs, current_physics_weight, patience,
                 output_dir, checkpoint_freq, logger, dataset):
    """Modified training loop for BO. Returns best_val_loss and last_train_physics_loss."""
    best_val_loss = float('inf')
    last_train_physics_loss = float('inf') # Initialize high
    patience_counter = 0

    logger.info(f"Starting BO training trial for {num_epochs} epochs with physics_weight={current_physics_weight:.6e}")
    final_epoch = start_epoch # Keep track of the last epoch run
    for epoch in range(start_epoch, num_epochs):
        final_epoch = epoch # Update last epoch successfully started
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        param_running_losses = {'RHOX': 0.0, 'T': 0.0, 'P': 0.0,
                               'XNE': 0.0, 'ABROSS': 0.0, 'ACCRAD': 0.0, 'physics': 0.0}
        # ... (rest of the batch loop inside the epoch is the same) ...
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss, param_losses = custom_loss(outputs, targets, dataset, model, inputs, True, current_physics_weight, None, True)

            # Accumulate losses
            if 'physics' in param_losses and isinstance(param_losses['physics'], torch.Tensor):
                 param_running_losses['physics'] += param_losses['physics'].item()
            elif 'physics' in param_losses: # Handle float case (e.g., if physics loss is 0.0)
                 param_running_losses['physics'] += param_losses['physics']


            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Accumulate other param losses (excluding physics already handled)
            for param, param_loss in param_losses.items():
                if param != 'physics':
                    if isinstance(param_loss, torch.Tensor):
                        param_running_losses[param] += param_loss.item()
                    else:
                         param_running_losses[param] += param_loss

        # Calculate epoch statistics
        train_epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_time = time.time() - epoch_start_time
        train_avg_param_losses = {k: v / len(train_loader) if len(train_loader) > 0 else 0 for k, v in param_running_losses.items()}

        # --- Store the physics loss from this epoch ---
        last_train_physics_loss = train_avg_param_losses.get('physics', float('inf')) # Update with the latest value, default to inf if missing

        data_loss = sum(v for k, v in train_avg_param_losses.items() if k != 'physics')
        # physics_loss = last_train_physics_loss # Redundant now stored above

        # Validation phase
        val_loss, val_param_losses = validate(model, val_loader, device, dataset, current_physics_weight, logger)
        val_data_loss = sum(v for k, v in val_param_losses.items() if k != 'physics')
        val_physics_loss = val_param_losses.get('physics', 0.0)

        # Scheduler Step (simplified)
        # ... (scheduler logic remains the same) ...
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
             if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                 scheduler.step(val_loss)
             new_lr = optimizer.param_groups[0]['lr']
             if current_lr != new_lr:
                 logger.debug(f'BO Trial LR changed from {current_lr:.6e} to {new_lr:.6e}')

        logger.debug(f'BO Trial Epoch {epoch+1}, Train Loss: {train_epoch_loss:.6f}, Val Loss: {val_loss:.6f}')
        logger.debug(f'  Data Loss: Train={data_loss:.6f}, Val={val_data_loss:.6f}')
        logger.debug(f'  Physics Loss: Train={last_train_physics_loss:.6f}, Val={val_physics_loss:.6f} (weight: {current_physics_weight:.6e})')


        # Check validation loss for saving best model *within the trial* and for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            logger.debug(f'BO Trial New best validation loss: {best_val_loss:.6f}')
            # Optional saving...
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f'BO Trial Early stopping triggered after {epoch+1} epochs for weight {current_physics_weight:.6e}')
            break # Exit the epoch loop

    logger.info(f"BO training trial finished after epoch {final_epoch+1} for weight {current_physics_weight:.6e}.")
    logger.info(f"  Returning: Best Validation Loss={best_val_loss:.6f}, Last Training Physics Loss={last_train_physics_loss:.6f}")

    # --- Return both values ---
    return best_val_loss, last_train_physics_loss


# =============================================================================
# Bayesian Optimization Objective Function
# =============================================================================
# =============================================================================
# Bayesian Optimization Objective Function (Minimizes Trained Physics Loss)
# =============================================================================

# Search space definition remains the same
search_space = [
    Real(1e-6, 1e+1, prior='log-uniform', name='physics_weight')
]

# Global variables remain the same
GLOBAL_ARGS = None
GLOBAL_LOGGER = None
GLOBAL_DEVICE = None
GLOBAL_TRAIN_LOADER = None
GLOBAL_VAL_LOADER = None
GLOBAL_DATASET = None

@use_named_args(search_space)
def objective(**params):
    """Objective function for BO. Minimizes the final trained physics loss."""
    current_physics_weight = params['physics_weight']

    args = GLOBAL_ARGS
    logger = GLOBAL_LOGGER
    device = GLOBAL_DEVICE
    train_loader = GLOBAL_TRAIN_LOADER
    val_loader = GLOBAL_VAL_LOADER
    dataset = GLOBAL_DATASET

    logger.info(f"\n--- Starting BO Trial ---")
    logger.info(f"Testing parameters: {params}")
    logger.info(f"Objective: Minimize LAST TRAINING epoch's average PHYSICS loss.") # Clarify objective

    # Setup Model, Optimizer for this trial (remains the same)
    model = AtmosphereNetMLPtau(
        stellar_embed_dim=args.hidden_size, # Use hidden_size from args
        tau_embed_dim=64
    ).to(device)
    logger.debug(f"BO Trial: Model created with {sum(p.numel() for p in model.parameters())} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.debug(f"BO Trial: Using Adam optimizer with initial learning rate {args.lr}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=max(1, args.bo_patience // 2), factor=args.lr_gamma) # Simplified scheduler


    bo_epochs = args.bo_epochs
    bo_patience = args.bo_patience

    # --- Call train_for_bo and capture both return values ---
    best_val_loss, final_train_physics_loss = train_for_bo(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        start_epoch=0,
        num_epochs=bo_epochs,
        current_physics_weight=current_physics_weight,
        patience=bo_patience,
        output_dir=args.output_dir,
        checkpoint_freq=args.checkpoint_freq,
        logger=logger,
        dataset=dataset
    )

    # Clean up GPU memory
    del model
    del optimizer
    del scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Return the value to be minimized by BO ---
    objective_value = final_train_physics_loss
    # Handle cases where training might fail or physics loss is inf
    if objective_value == float('inf') or np.isnan(objective_value):
         logger.warning(f"BO Trial resulted in invalid physics loss ({objective_value}). Returning a large penalty value.")
         objective_value = 1e10 # Return a large number as penalty

    logger.info(f"--- BO Trial Finished --- Parameters: {params}")
    logger.info(f"  Best Validation Loss during trial: {best_val_loss:.6f}")
    logger.info(f"  Final Training Physics Loss (Objective Value): {objective_value:.6f}\n")


    return objective_value # gp_minimize will try to minimize THIS value

# =============================================================================
# Main Function (Modified for BO)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Kurucz Atmospheric Model with Bayesian Optimization')
    # ... (keep all your original arguments) ...
    parser.add_argument('--dataset', type=str, required=True, help='Path to the saved dataset file')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for FINAL training run') # Final run epochs
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--checkpoint_freq', type=int, default=100, help='Checkpoint frequency (epochs)')
    parser.add_argument('--log_freq', type=int, default=10, help='Logging frequency (batches)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume FINAL training from')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience for FINAL training')
    parser.add_argument('--validation_split', type=float, default=0.1, help='Validation set split ratio')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker processes for data loading')
    # parser.add_argument('--physics_weight', type=float, default=1e-3, help='Weight for physics-informed loss component (will be optimized)') # BO will set this
    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'step', 'multistep', 'exponential', 'cosine', 'plateau', 'cyclic', 'onecycle'], help='Learning rate scheduler type')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--lr_step_size', type=int, default=100, help='Period of learning rate decay (epochs)')
    parser.add_argument('--lr_milestones', type=str, default='200,400,600', help='Epochs at which to decay learning rate (comma-separated)')
    parser.add_argument('--lr_min', type=float, default=1e-7, help='Minimum learning rate')
    parser.add_argument('--lr_patience', type=int, default=5, help='Epochs with no improvement after which learning rate will be reduced')
    # ... (keep other LR scheduler args) ...
    parser.add_argument('--lr_cycles', type=int, default=4, help='Number of cycles for cyclic LR scheduler')
    parser.add_argument('--lr_cycle_mode', type=str, default='triangular2', choices=['triangular', 'triangular2', 'exp_range'], help='Mode for cyclic LR scheduler')
    parser.add_argument('--lr_warmup_pct', type=float, default=0.3, help='Percentage of total iterations for warmup in OneCycleLR')
    parser.add_argument('--lr_div_factor', type=float, default=25.0, help='Initial learning rate division factor for OneCycleLR')
    parser.add_argument('--lr_final_div_factor', type=float, default=1e4, help='Final learning rate division factor for OneCycleLR')


    # --- BO Specific Arguments ---
    parser.add_argument('--run_bo', action='store_true', help='Run Bayesian Optimization to find physics_weight')
    parser.add_argument('--bo_calls', type=int, default=20, help='Number of Bayesian Optimization trials')
    parser.add_argument('--bo_epochs', type=int, default=50, help='Number of epochs per BO trial')
    parser.add_argument('--bo_patience', type=int, default=10, help='Early stopping patience for BO trials')
    parser.add_argument('--final_physics_weight', type=float, default=1e-3, help='Physics weight to use if not running BO or after BO')


    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True) # Ensure log dir exists

    # Setup logger
    logger = setup_logger(args.log_dir)

    # Set device
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda:1') # Or 'cuda:0' or dynamic selection
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
            logger.warning("GPU requested but CUDA/MPS not available, using CPU.")
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # Log training configuration
    logger.info("Full configuration:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")

    # Load dataset and create dataloaders
    logger.info(f"Loading dataset from {args.dataset}")
    train_loader, val_loader, dataset = create_dataloader_from_saved(
        filepath=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers, # Use the arg
        device=device, # Pass device here if your function supports it
        validation_split=args.validation_split
    )
    logger.info(f"Dataset loaded, {len(dataset)} total samples, "
               f"{len(train_loader.dataset)} training, {len(val_loader.dataset)} validation")

    # --- Assign Globals for BO Objective Function ---
    # This is a simple way; passing explicitly is often cleaner
    global GLOBAL_ARGS, GLOBAL_LOGGER, GLOBAL_DEVICE
    global GLOBAL_TRAIN_LOADER, GLOBAL_VAL_LOADER, GLOBAL_DATASET
    GLOBAL_ARGS = args
    GLOBAL_LOGGER = logger
    GLOBAL_DEVICE = device
    GLOBAL_TRAIN_LOADER = train_loader
    GLOBAL_VAL_LOADER = val_loader
    GLOBAL_DATASET = dataset

    best_physics_weight = args.final_physics_weight # Default value

    # --- Run Bayesian Optimization ---
    if args.run_bo:
        logger.info(f"Starting Bayesian Optimization for 'physics_weight' ({args.bo_calls} calls, {args.bo_epochs} epochs/call)")

        # Make sure random state is fixed for reproducibility if desired
        random_seed = 123
        np.random.seed(random_seed)

        bo_result = gp_minimize(
            func=objective,          # Function to minimize
            dimensions=search_space, # Search space boundaries/prior
            acq_func="EI",           # Acquisition function (Expected Improvement)
            n_calls=args.bo_calls,   # Number of evaluations of func
            random_state=random_seed,
            verbose=True             # Print progress
        )

        best_physics_weight = bo_result.x[0]
        best_objective_score = bo_result.fun

        logger.info("\n--- Bayesian Optimization Finished ---")
        logger.info(f"Best physics_weight found: {best_physics_weight:.6e}")
        logger.info(f"Best validation score achieved during BO: {best_objective_score:.6f}")

        # Save BO results (optional)
        try:
            from skopt.plots import plot_convergence, plot_objective
            import matplotlib.pyplot as plt

            plot_convergence(bo_result)
            plt.savefig(os.path.join(args.log_dir, "bo_convergence.png"))
            plt.close()

            plot_objective(bo_result)
            plt.savefig(os.path.join(args.log_dir, "bo_objective.png"))
            plt.close()
            logger.info("Saved BO convergence and objective plots.")
        except ImportError:
            logger.warning("Install matplotlib to save BO plots (`pip install matplotlib`)")
        except Exception as e:
             logger.error(f"Could not save BO plots: {e}")


    # --- Final Training Run ---
    logger.info(f"\n--- Starting Final Training Run ---")
    logger.info(f"Using physics_weight: {best_physics_weight:.6e} for {args.epochs} epochs")

    # Set the physics weight for the final run
    args.physics_weight = best_physics_weight # Add this attribute dynamically for the final run

    # Create final model instance
    final_model = AtmosphereNetMLPtau(
        stellar_embed_dim=args.hidden_size, tau_embed_dim=64 # Use final hidden size
    ).to(device)
    logger.info(f"Final Model created with {sum(p.numel() for p in final_model.parameters())} parameters")

    # Create final optimizer and scheduler
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=args.lr)
    final_scheduler = get_scheduler(args.scheduler, final_optimizer, args, logger) # Use full scheduler config

    # Load checkpoint if resuming final run
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        try:
            # Pass final_scheduler to load_checkpoint
            start_epoch, best_loss = load_checkpoint(final_model, final_optimizer, final_scheduler, args.resume, device, logger)
            logger.info(f"Resuming final training from epoch {start_epoch} with best loss {best_loss}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint for final run: {str(e)}. Starting final run from scratch.")
            start_epoch = 0

    # Setup TensorBoard for the final run
    final_writer = SummaryWriter(os.path.join(args.log_dir, 'tensorboard_final'))
    try:
        example_input = next(iter(train_loader))[0][:1].to(device)
        final_writer.add_graph(final_model, example_input)
    except Exception as e:
        logger.warning(f"Could not add model graph to TensorBoard: {e}")


    # Call the *original* train function (or a version adapted for final logging/checkpointing)
    # We need to modify the original `train` to accept `physics_weight` and `writer`
    # Or create a wrapper function `final_train`
    # For simplicity, let's adapt `train_for_bo` slightly for the final run

    logger.info(f"Starting final training for {args.epochs} epochs")
    final_best_loss = float('inf')
    final_patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        final_model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        param_running_losses = {'RHOX': 0.0, 'T': 0.0, 'P': 0.0,
                               'XNE': 0.0, 'ABROSS': 0.0, 'ACCRAD': 0.0, 'physics': 0.0}

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            final_optimizer.zero_grad()
            outputs = final_model(inputs)
            loss, param_losses = custom_loss(outputs, targets, dataset, final_model, inputs, True, args.physics_weight, None, True)

            loss.backward()
            final_optimizer.step()

            if args.scheduler in ['cyclic', 'onecycle'] and final_scheduler:
                final_scheduler.step()

            running_loss += loss.item()
            for param, param_loss in param_losses.items():
                 if isinstance(param_loss, torch.Tensor):
                    param_running_losses[param] += param_loss.item()
                 else:
                    param_running_losses[param] += param_loss


        train_epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_time = time.time() - epoch_start_time
        train_avg_param_losses = {k: v / len(train_loader) if len(train_loader) > 0 else 0 for k, v in param_running_losses.items()}
        data_loss = sum(v for k, v in train_avg_param_losses.items() if k != 'physics')
        physics_loss = train_avg_param_losses.get('physics', 0.0)

        # Validation phase
        val_loss, val_param_losses = validate(final_model, val_loader, device, dataset, args.physics_weight, logger)
        val_data_loss = sum(v for k, v in val_param_losses.items() if k != 'physics')
        val_physics_loss = val_param_losses.get('physics', 0.0)

        # Step for epoch-based schedulers
        current_lr = final_optimizer.param_groups[0]['lr']
        if final_scheduler:
            if args.scheduler == 'plateau':
                final_scheduler.step(val_loss)
            elif args.scheduler not in ['none', 'cyclic', 'onecycle']:
                final_scheduler.step()
            new_lr = final_optimizer.param_groups[0]['lr']
            if current_lr != new_lr:
                logger.info(f'Final Run: Learning rate changed from {current_lr:.6e} to {new_lr:.6e}')

        logger.info(f'Final Run Epoch {epoch+1}, Train Loss: {train_epoch_loss:.6f}, Val Loss: {val_loss:.6f}')
        logger.info(f'  Data Loss: Train={data_loss:.6f}, Val={val_data_loss:.6f}')
        logger.info(f'  Physics Loss: Train={physics_loss:.6f}, Val={val_physics_loss:.6f} (weight: {args.physics_weight:.6e})')

        # Log epoch metrics to TensorBoard for final run
        final_writer.add_scalar('Loss/train', train_epoch_loss, epoch)
        final_writer.add_scalar('Loss/val', val_loss, epoch)
        final_writer.add_scalar('Time/epoch', epoch_time, epoch)
        final_writer.add_scalar('LearningRate/epoch', final_optimizer.param_groups[0]['lr'], epoch)
        for param, loss_val in train_avg_param_losses.items():
            final_writer.add_scalar(f'Loss_train/{param}', loss_val, epoch)
        for param, loss_val in val_param_losses.items():
            final_writer.add_scalar(f'Loss_val/{param}', loss_val, epoch)

        # Save checkpoint if validation loss improves
        if val_loss < final_best_loss:
            final_best_loss = val_loss
            final_patience_counter = 0
            logger.info(f'Final Run: New best validation loss: {final_best_loss:.6f}')
            save_checkpoint(final_model, final_optimizer, final_scheduler, epoch+1, val_loss,
                           os.path.join(args.output_dir, 'best_model_final.pt'), logger)
        else:
            final_patience_counter += 1

        if final_patience_counter >= args.patience:
            logger.info(f'Final Run: Early stopping triggered after {epoch+1} epochs')
            break

        if (epoch + 1) % args.checkpoint_freq == 0:
            save_checkpoint(final_model, final_optimizer, final_scheduler, epoch+1, val_loss,
                           os.path.join(args.output_dir, f'checkpoint_final_epoch_{epoch+1}.pt'), logger)

    save_checkpoint(final_model, final_optimizer, final_scheduler, args.epochs, train_epoch_loss,
                   os.path.join(args.output_dir, 'final_model_final.pt'), logger)

    logger.info("Final training completed!")
    final_writer.close()


if __name__ == '__main__':
    main()