
#!/usr/bin/env python
# utils.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import os
import logging
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots


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


def calculate_dP_dtau_ground_truth(dataset, sample_indices=None, device='cpu'):
    """
    Calculate the ground truth pressure gradient with respect to optical depth (dP/dtau)
    directly from the dataset.
    Parameters:
        dataset (KuruczDataset): Dataset containing the atmospheric models
        sample_indices (list, optional): Indices of samples to process. If None, use all samples.
        device (str): Device to run calculations on ('cpu' or 'cuda')
        
    Returns:
        dict: Dictionary containing tau and dP/dtau values for each sample
    """
    if sample_indices is None:
        sample_indices = range(len(dataset))
    
    results = {
        'tau': [],
        'dP_dtau': [],
        'g_kappa': [],
        'teff': [],
        'gravity': [],
        'feh': [],
        'afe': []
    }
    
    for idx in sample_indices:
        # Get data from dataset
        # p_normalized = dataset.P[idx]
        abross_normalized = dataset.ABROSS[idx]
        
        # Use original tau values directly (recommended approach based on code comments)
        tau = dataset.original['TAU'][idx].cpu().numpy()
        
        # Denormalize other parameters
        # pressure = dataset.denormalize('P', p_normalized).cpu().numpy()
        pressure = dataset.original['P'][idx].cpu().numpy()
        kappa = dataset.denormalize('ABROSS', abross_normalized).cpu().numpy()
        
        # Calculate dP/dtau
        dP_dtau = calculate_gradient_torch(
            torch.tensor(tau).unsqueeze(0), 
            torch.tensor(pressure).unsqueeze(0)
        )[0].numpy()
        
        # Get stellar parameters
        teff = dataset.denormalize('teff', dataset.teff[idx]).item()
        gravity = 10**dataset.denormalize('gravity', dataset.gravity[idx]).item()
        feh = dataset.denormalize('feh', dataset.feh[idx]).item()
        afe = dataset.denormalize('afe', dataset.afe[idx]).item()
        
        # Calculate g/kappa
        g_kappa = gravity / kappa
        
        # Store results
        results['tau'].append(tau)
        results['dP_dtau'].append(dP_dtau)
        results['g_kappa'].append(g_kappa)
        results['teff'].append(teff)
        results['gravity'].append(gravity)
        results['feh'].append(feh)
        results['afe'].append(afe)
    
    return results

def calculate_dP_dtau_predicted(model, dataset, sample_indices=None, device='cpu'):
    """
    Calculate the predicted pressure gradient with respect to optical depth (dP/dtau)
    and g/kappa using the model predictions.
    
    Parameters:
        model (AtmosphereNet): Trained neural network model
        dataset (KuruczDataset): Dataset containing the atmospheric models
        sample_indices (list, optional): Indices of samples to process. If None, use all samples.
        device (str): Device to run calculations on ('cpu' or 'cuda')
        
    Returns:
        dict: Dictionary containing tau, predicted dP/dtau, and g/kappa values for each sample
    """
    if sample_indices is None:
        sample_indices = range(len(dataset))
    
    results = {
        'tau': [],
        'dP_dtau_pred': [],
        'g_kappa_pred': [],
        'teff': [],
        'gravity': [],
        'feh': [],
        'afe': []
    }
    
    model.eval()
    with torch.no_grad():
        for idx in sample_indices:
            # Get model prediction
            outputs = model(dataset[idx][0].unsqueeze(0))
            
            # Extract predicted values
            p_pred_norm = outputs[0, :, 2].cpu()
            abross_pred_norm = outputs[0, :, 4].cpu()
            
            # Use original tau values directly
            tau = dataset.original['TAU'][idx].cpu().numpy()
            
            # Denormalize predicted values
            pressure_pred = dataset.denormalize('P', p_pred_norm).cpu().numpy()
            abross_pred = dataset.denormalize('ABROSS', abross_pred_norm).cpu().numpy()
            
            # Calculate predicted dP/dtau
            dP_dtau_pred = calculate_gradient_torch(
                torch.tensor(tau).unsqueeze(0), 
                torch.tensor(pressure_pred).unsqueeze(0)
            )[0].numpy()
            
            # Get stellar parameters
            teff = dataset.denormalize('teff', dataset.teff[idx]).item()
            gravity = dataset.denormalize('gravity', dataset.gravity[idx]).item()
            feh = dataset.denormalize('feh', dataset.feh[idx]).item()
            afe = dataset.denormalize('afe', dataset.afe[idx]).item()
            
            # Calculate predicted g/kappa
            g_kappa_pred = 10**gravity / abross_pred
            
            # Store results
            results['tau'].append(tau)
            results['dP_dtau_pred'].append(dP_dtau_pred)
            results['g_kappa_pred'].append(g_kappa_pred)
            results['teff'].append(teff)
            results['gravity'].append(gravity)
            results['feh'].append(feh)
            results['afe'].append(afe)
    
    return results


# --- Generate plots ---
# Plot residuals
def plot_residuals(predictions, targets):
    param_names = ['RHOX', 'T', 'P', 'XNE', 'ABROSS', 'ACCRAD']
    batch_size = predictions.shape[0]
    indices = torch.randperm(batch_size)[:min(5000, batch_size)]
    
    plt.figure(figsize=(15, 10))
    
    for i, param in enumerate(param_names):
        plt.subplot(2, 3, i + 1)
        
        residuals = predictions[indices, :, i] - targets[indices, :, i]
        residuals = residuals.flatten().numpy()
        
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.title(f'{param} Residuals')
        plt.axvline(np.mean(residuals), color='r', linestyle='dashed')
    
    plt.tight_layout()
    return plt


# Plot sample depth profiles
def plot_depth_profile(idx, predictions, targets):
    param_names = ['RHOX', 'T', 'P', 'XNE', 'ABROSS', 'ACCRAD']
    
    plt.figure(figsize=(15, 10))
    for i, param in enumerate(param_names):
        plt.subplot(2, 3, i + 1)
        
        pred_values = predictions[idx, :, i].numpy()
        true_values = targets[idx, :, i].numpy()
        depth_points = np.arange(len(pred_values))
        
        plt.plot(depth_points, pred_values, 'b-', label='Predicted')
        plt.plot(depth_points, true_values, 'r--', label='True')
        
        plt.title(f'{param}')
        plt.xlabel('Depth Point')
        plt.ylabel('Value')
        plt.legend()
    
    plt.tight_layout()
    return plt

def plot_comprehensive_comparison(dP_dtau_predicted, dP_dtau_ground_truth, sample_idx=0):
    """
    Create a 2x2 grid of plots for a single sample showing:
    1. g/kappa truth vs g/kappa predicted
    2. dP/dtau truth vs dP/dtau predicted
    3. g/kappa truth vs dP/dtau truth
    4. g/kappa predicted vs dP/dtau predicted
    
    Parameters:
        results_diff_grad (dict): Results from differential gradient calculation
        dP_dtau_ground_truth (dict): Ground truth dP/dtau values
        sample_idx (int): Index of the sample to plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create a 2x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # Get data for the specified sample
    
    # Get data for the specified sample
    tau = dP_dtau_predicted['tau'][sample_idx]
    dP_dtau_pred = dP_dtau_predicted['dP_dtau_pred'][sample_idx]
    g_kappa_pred = dP_dtau_predicted['g_kappa_pred'][sample_idx]
    g_kappa_true = dP_dtau_ground_truth['g_kappa'][sample_idx]
    
    # Get ground truth dP/dtau
    tau_gt = dP_dtau_ground_truth['tau'][sample_idx]
    dP_dtau_true = dP_dtau_ground_truth['dP_dtau'][sample_idx]
    
    # 过滤NaN和Inf值
    valid_tau = ~np.isnan(tau) & ~np.isinf(tau)
    valid_gk_true = ~np.isnan(g_kappa_true) & ~np.isinf(g_kappa_true)
    valid_gk_pred = ~np.isnan(g_kappa_pred) & ~np.isinf(g_kappa_pred)
    valid_dp_pred = ~np.isnan(dP_dtau_pred) & ~np.isinf(dP_dtau_pred)
    valid_tau_gt = ~np.isnan(tau_gt) & ~np.isinf(tau_gt)
    valid_dp_true = ~np.isnan(dP_dtau_true) & ~np.isinf(dP_dtau_true)
    
    # 组合掩码
    valid_plot1_true = valid_tau & valid_gk_true
    valid_plot1_pred = valid_tau & valid_gk_pred
    valid_plot2_true = valid_tau_gt & valid_dp_true
    valid_plot2_pred = valid_tau & valid_dp_pred

    axes[0, 0].scatter(tau[valid_plot1_true], g_kappa_true[valid_plot1_true], s=30, label='g/κ (True)', color='red', alpha=0.7)
    axes[0, 0].scatter(tau[valid_plot1_pred], g_kappa_pred[valid_plot1_pred], s=30, label='g/κ (Predicted)', color='blue', alpha=0.7)
    
    # Plot 2: dP/dtau truth vs dP/dtau predicted
    valid_plot2_true = valid_tau_gt & valid_dp_true
    valid_plot2_pred = valid_tau & valid_dp_pred
    axes[0, 1].scatter(tau_gt[valid_plot2_true], dP_dtau_true[valid_plot2_true], s=30, label='dP/dτ (True)', color='red', alpha=0.7)
    axes[0, 1].scatter(tau[valid_plot2_pred], dP_dtau_pred[valid_plot2_pred], s=30, label='dP/dτ (Predicted)', color='blue', alpha=0.7)
    
    # Plot 3: g/kappa truth vs dP/dtau truth
    # We need to interpolate to compare at the same tau points
    from scipy.interpolate import interp1d
    
    # Find common tau range for valid points only
    if np.any(valid_tau & valid_gk_true) and np.any(valid_tau_gt & valid_dp_true):
        min_tau = max(np.min(tau[valid_tau & valid_gk_true]), np.min(tau_gt[valid_tau_gt & valid_dp_true]))
        max_tau = min(np.max(tau[valid_tau & valid_gk_true]), np.max(tau_gt[valid_tau_gt & valid_dp_true]))
        
        # Filter points within common range
        g_kappa_mask = (tau >= min_tau) & (tau <= max_tau) & valid_tau & valid_gk_true
        dP_dtau_mask = (tau_gt >= min_tau) & (tau_gt <= max_tau) & valid_tau_gt & valid_dp_true
        
        # Interpolate dP/dtau truth to g/kappa tau points
        if sum(dP_dtau_mask) > 1 and sum(g_kappa_mask) > 1:
            dP_dtau_interp = interp1d(tau_gt[dP_dtau_mask], dP_dtau_true[dP_dtau_mask], 
                                     bounds_error=False, fill_value="extrapolate")
            dP_dtau_true_at_g_kappa_points = dP_dtau_interp(tau[g_kappa_mask])
            
            # Filter out any NaN or Inf values that might have been introduced by interpolation
            valid_interp = ~np.isnan(dP_dtau_true_at_g_kappa_points) & ~np.isinf(dP_dtau_true_at_g_kappa_points)
            
            if np.any(valid_interp):
                # Plot g/kappa truth vs interpolated dP/dtau truth
                axes[1, 0].scatter(dP_dtau_true_at_g_kappa_points[valid_interp], 
                                  g_kappa_true[g_kappa_mask][valid_interp], 
                                  s=30, color='purple', alpha=0.7)
                
                # Add identity line (y=x)
                min_val = min(np.min(dP_dtau_true_at_g_kappa_points[valid_interp]), 
                              np.min(g_kappa_true[g_kappa_mask][valid_interp]))
                max_val = max(np.max(dP_dtau_true_at_g_kappa_points[valid_interp]), 
                              np.max(g_kappa_true[g_kappa_mask][valid_interp]))
                axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    
    # Plot 4: g/kappa predicted vs dP/dtau predicted
    valid_plot4 = valid_gk_pred & valid_dp_pred
    if np.any(valid_plot4):
        axes[1, 1].scatter(dP_dtau_pred[valid_plot4], g_kappa_pred[valid_plot4], s=30, color='green', alpha=0.7)
        # print(dP_dtau_pred[valid_plot4], g_kappa_pred[valid_plot4])
        # Add identity line (y=x)
        min_val = min(np.min(dP_dtau_pred[valid_plot4]), np.min(g_kappa_pred[valid_plot4]))
        max_val = max(np.max(dP_dtau_pred[valid_plot4]), np.max(g_kappa_pred[valid_plot4]))
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    
    # Calculate metrics for each comparison, ignoring NaN and Inf values
    # 1. g/kappa truth vs predicted
    valid_metrics1 = valid_gk_true & valid_gk_pred
    if np.any(valid_metrics1):
        g_kappa_mse = np.mean((g_kappa_true[valid_metrics1] - g_kappa_pred[valid_metrics1])**2)
        g_kappa_rel_error = np.mean(np.abs((g_kappa_true[valid_metrics1] - g_kappa_pred[valid_metrics1]) / 
                                          (np.abs(g_kappa_true[valid_metrics1]) + 1e-10))) * 100
    else:
        g_kappa_mse = np.nan
        g_kappa_rel_error = np.nan
    
    # 2. dP/dtau truth vs predicted
    # Use the interpolation results from above if available
    if 'dP_dtau_true_at_g_kappa_points' in locals() and 'valid_interp' in locals() and np.any(valid_interp):
        dP_dtau_pred_at_common_points = dP_dtau_pred[g_kappa_mask][valid_interp]
        dP_dtau_mse = np.mean((dP_dtau_true_at_g_kappa_points[valid_interp] - dP_dtau_pred_at_common_points)**2)
        dP_dtau_rel_error = np.mean(np.abs((dP_dtau_true_at_g_kappa_points[valid_interp] - dP_dtau_pred_at_common_points) / 
                                          (np.abs(dP_dtau_true_at_g_kappa_points[valid_interp]) + 1e-10))) * 100
    else:
        dP_dtau_mse = np.nan
        dP_dtau_rel_error = np.nan
    
    # 3. g/kappa truth vs dP/dtau truth
    if 'dP_dtau_true_at_g_kappa_points' in locals() and 'valid_interp' in locals() and np.any(valid_interp):
        gk_vs_dp_true_mse = np.mean((g_kappa_true[g_kappa_mask][valid_interp] - dP_dtau_true_at_g_kappa_points[valid_interp])**2)
        gk_vs_dp_true_rel_error = np.mean(np.abs((g_kappa_true[g_kappa_mask][valid_interp] - dP_dtau_true_at_g_kappa_points[valid_interp]) / 
                                               (np.abs(g_kappa_true[g_kappa_mask][valid_interp]) + 1e-10))) * 100
    else:
        gk_vs_dp_true_mse = np.nan
        gk_vs_dp_true_rel_error = np.nan
    
    # 4. g/kappa predicted vs dP/dtau predicted
    if np.any(valid_plot4):
        gk_vs_dp_pred_mse = np.mean((g_kappa_pred[valid_plot4] - dP_dtau_pred[valid_plot4])**2)
        gk_vs_dp_pred_rel_error = np.mean(np.abs((g_kappa_pred[valid_plot4] - dP_dtau_pred[valid_plot4]) / 
                                               (np.abs(g_kappa_pred[valid_plot4]) + 1e-10))) * 100
    else:
        gk_vs_dp_pred_mse = np.nan
        gk_vs_dp_pred_rel_error = np.nan
    

        # 计算并绘制相对误差
    # Plot 5: Relative error between g/kappa truth and dP/dtau truth
    if 'dP_dtau_true_at_g_kappa_points' in locals() and 'valid_interp' in locals() and np.any(valid_interp):
        # Calculate relative difference between g/kappa truth and dP/dtau truth
        rel_diff_truth = np.abs((g_kappa_true[g_kappa_mask][valid_interp] - dP_dtau_true_at_g_kappa_points[valid_interp]) / 
                               (np.abs(g_kappa_true[g_kappa_mask][valid_interp]) + 1e-10)) * 100
        
        # Plot relative difference vs optical depth
        axes[2, 0].scatter(tau[g_kappa_mask][valid_interp], rel_diff_truth, 
                          s=30, color='purple', alpha=0.7)
        
        axes[2, 0].set_xlabel('Optical Depth (τ)')
        axes[2, 0].set_ylabel('Relative Difference (%)')
        axes[2, 0].set_title('Truth: |g/κ - dP/dτ|/g/κ')
        axes[2, 0].set_xscale('log')
        
        axes[2, 0].grid(True, which='both', linestyle='--', alpha=0.6)
        
        # Add mean relative difference
        mean_rel_diff_truth = np.mean(rel_diff_truth)
        axes[2, 0].axhline(y=mean_rel_diff_truth, color='r', linestyle='--')
        axes[2, 0].text(0.05, 0.95, f'Mean: {mean_rel_diff_truth:.2f}%', 
                       transform=axes[2, 0].transAxes, 
                       verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Plot 6: Relative error between g/kappa predicted and dP/dtau predicted
    valid_plot6 = valid_gk_pred & valid_dp_pred
    if np.any(valid_plot6):
        # Calculate relative difference between g/kappa pred and dP/dtau pred
        rel_diff_pred = np.abs((g_kappa_pred[valid_plot6] - dP_dtau_pred[valid_plot6]) / 
                              (np.abs(g_kappa_pred[valid_plot6]) + 1e-10)) * 100
        
        # Plot relative difference vs optical depth
        axes[2, 1].scatter(tau[valid_plot6], rel_diff_pred, 
                          s=30, color='green', alpha=0.7)
        
        axes[2, 1].set_xlabel('Optical Depth (τ)')
        axes[2, 1].set_ylabel('Relative Difference (%)')
        axes[2, 1].set_title('Predicted: |g/κ - dP/dτ|/g/κ')
        axes[2, 1].set_xscale('log')
        
        
        axes[2, 1].grid(True, which='both', linestyle='--', alpha=0.6)
        
        # Add mean relative difference
        mean_rel_diff_pred = np.mean(rel_diff_pred)
        axes[2, 1].axhline(y=mean_rel_diff_pred, color='r', linestyle='--')
        axes[2, 1].text(0.05, 0.95, f'Mean: {mean_rel_diff_pred:.2f}%', 
                       transform=axes[2, 1].transAxes, 
                       verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # axes[2,2]
        # Plot 7: Relative error between g/kappa truth and dP/dtau truth


    # Set plot properties
    axes[0, 0].set_xlabel('Optical Depth (τ)')
    axes[0, 0].set_ylabel('g/κ')
    axes[0, 0].set_title(f'g/κ Truth vs Predicted\nMSE={g_kappa_mse:.2e}, Rel Error={g_kappa_rel_error:.2f}%')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, which='both', linestyle='--', alpha=0.6)
    axes[0, 0].legend()
    
    axes[0, 1].set_xlabel('Optical Depth (τ)')
    axes[0, 1].set_ylabel('dP/dτ')
    axes[0, 1].set_title(f'dP/dτ Truth vs Predicted\nMSE={dP_dtau_mse:.2e}, Rel Error={dP_dtau_rel_error:.2f}%')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, which='both', linestyle='--', alpha=0.6)
    axes[0, 1].legend()
    
    axes[1, 0].set_xlabel('dP/dτ (Truth)')
    axes[1, 0].set_ylabel('g/κ (Truth)')
    axes[1, 0].set_title(f'g/κ Truth vs dP/dτ Truth\nMSE={gk_vs_dp_true_mse:.2e}, Rel Error={gk_vs_dp_true_rel_error:.2f}%')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    
    axes[1, 1].set_xlabel('dP/dτ (Predicted)')
    axes[1, 1].set_ylabel('g/κ (Predicted)')
    axes[1, 1].set_title(f'g/κ Pred vs dP/dτ Pred\nMSE={gk_vs_dp_pred_mse:.2e}, Rel Error={gk_vs_dp_pred_rel_error:.2f}%')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    
    # Add overall title with stellar parameters
    plt.suptitle(f'Teff={dP_dtau_ground_truth["teff"][sample_idx]:.0f}, '
                f'log g={np.log10(dP_dtau_ground_truth["gravity"][sample_idx]):.2f}, '
                f'[Fe/H]={dP_dtau_ground_truth["feh"][sample_idx]:.2f}, '
                f'[α/Fe]={dP_dtau_ground_truth["afe"][sample_idx]:.2f}', 
                fontsize=16, y=1.05)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Adjust for suptitle
    
    return fig

def hydro_equilibrium_loss(outputs, inputs, dataset, model, is_training=True):
    """
    Calculate hydrostatic equilibrium physics-informed loss
    
    Args:
        outputs (torch.Tensor): Model outputs (normalized)
        inputs (torch.Tensor): Model inputs
        dataset (KuruczDataset): Dataset object containing normalization parameters
        model (nn.Module): The neural network model
        is_training (bool): Whether we're in training mode
    
    Returns:
        torch.Tensor: Computed physics loss
    """
    # Constants for numerical stability and gradient clipping
    GRAD_CLIP = 1e3
    SAFE_LOG = 1e-30
    
    # ===== Parameter indices =====
    P_idx = 2        # Pressure index in output (3rd parameter)
    ABROSS_idx = 4    # ABROSS index in output (5th parameter)
    TAU_START = 4     # Starting index of tau parameters in input

    # ===== 1. Data Preparation =====
    p_params = dataset.norm_params['P']
    tau_params = dataset.norm_params['TAU']
    
    # Create inputs that require gradients
    grad_inputs = inputs.detach().clone()
    grad_inputs.requires_grad_(True)

    # ===== 2. Gradient Calculation =====
    # Temporarily set model to eval/train mode as needed
    original_training = model.training
    if is_training:
        model.train()
    else:
        model.eval()
        
    # Forward pass through the model
    model_outputs = model(grad_inputs)  # Output shape: (n_batch, 6, 80)
    P_norm = model_outputs[:, :, P_idx]  # [batch, depth]

    # Compute gradients
    grad_outputs = torch.ones_like(P_norm)
    try:
        gradients = torch.autograd.grad(
            outputs=P_norm,
            inputs=grad_inputs,
            grad_outputs=grad_outputs,
            create_graph=is_training,
            retain_graph=True,
            allow_unused=False
        )[0]
    
        # Extract tau gradients and clamp
        dlogP_norm_dlogtau_norm = gradients[:, TAU_START:TAU_START+80]
        dlogP_norm_dlogtau_norm = torch.clamp(dlogP_norm_dlogtau_norm, -GRAD_CLIP, GRAD_CLIP)

        # ===== 3. Physical Conversion =====
        logtau = (grad_inputs[:, TAU_START:TAU_START+80] * tau_params['scale'] 
                + (tau_params['min'] + tau_params['max'])/2)
        logP = (P_norm * p_params['scale'] 
            + (p_params['min'] + p_params['max'])/2)
        
        scale_ratio = p_params['scale'] / tau_params['scale']
        dlogP_dlogtau = dlogP_norm_dlogtau_norm * scale_ratio
        
        # Improved numerical stability in logarithm
        eps_mask = dlogP_dlogtau.abs() < SAFE_LOG
        safe_dlogP = dlogP_dlogtau.clone()
        safe_dlogP[eps_mask] = torch.sign(safe_dlogP[eps_mask]) * SAFE_LOG
        log_dP_dtau = logP - logtau + torch.log10(safe_dlogP.abs())

    except RuntimeError as e:
        if is_training:
            # During training, re-raise the error for debugging
            raise RuntimeError(f"Gradient calculation failed during training: {e}")
        else:
            # During validation/inference, log warning and return zero loss
            print(f"[WARNING] d(logP)/d(tau) gradient calculation failed: {e}")
            # Restore model state and return zero loss
            model.train(original_training)
            return torch.tensor(0.0, device=outputs.device)

    # ===== 4. Equilibrium Condition =====
    ABROSS_norm = torch.clamp(outputs[:, :, ABROSS_idx], -5.0, 5.0)
    kappa = dataset.denormalize('ABROSS', ABROSS_norm)
    log_kappa = torch.log10(torch.clamp(kappa, min=SAFE_LOG))
    
    logg = dataset.denormalize('gravity', inputs[:, 1])
    logg = torch.clamp(logg, -1, 6).unsqueeze(-1)

    term_left = log_dP_dtau     # (n_batch, 80)
    term_right = logg - log_kappa  # (n_batch, 80)

    # ===== 5. Modified Loss Calculation =====
    valid_mask = torch.isfinite(term_left) & torch.isfinite(term_right)
    
    # Create depth exclusion mask
    depth_mask = torch.ones_like(term_left, dtype=torch.bool)
    depth_mask[:, :1] = False    # Exclude first 1 depth points
    depth_mask[:, -21] = False   # Exclude last  1 depth points
    final_mask = valid_mask & depth_mask

    # Handle empty masks properly
    if not final_mask.any():
        # Restore model state
        model.train(original_training)
        # Return a small non-zero loss to maintain gradient flow
        return torch.tensor(1e-6, device=outputs.device, requires_grad=True)
    
    # Main MSE loss between terms
    mse_loss = F.mse_loss(term_left[final_mask], term_right[final_mask])
    
    # Reinclude physics sign constraint (gradient should have correct sign)
    # Pressure should increase with optical depth
    sign_loss = F.relu(-dlogP_dlogtau * torch.sign(term_right.detach())).mean()
    
    # Combine losses
    total_loss = mse_loss + 0.1 * sign_loss
    
    # Restore original model state
    model.train(original_training)
    
    return total_loss, term_left.detach(), term_right.detach()



def calculate_gradient_torch(x, y):
    """    
    参数:
    - x: (batch_size, n_points)
    - y:  batch_size, n_points)
    
    返回:
    - gradient: dy/dx (batch_size, n_points)
    """
    batch_size, n_points = x.shape
    
    # 初始化梯度张量
    gradient = torch.zeros_like(y)
    
    if n_points < 2:
        return gradient
    
    # 前向差分（第一个点）
    gradient[:, 0] = (y[:, 1] - y[:, 0]) / (x[:, 1] - x[:, 0] + 1e-15)
    
    # 中心差分（中间点）
    if n_points > 2:
        dx = x[:, 2:] - x[:, :-2]
        dy = y[:, 2:] - y[:, :-2]
        gradient[:, 1:-1] = dy / (dx + 1e-15)
    
    # 后向差分（最后一个点）
    gradient[:, -1] = (y[:, -1] - y[:, -2]) / (x[:, -1] - x[:, -2] + 1e-15)
    
    return gradient