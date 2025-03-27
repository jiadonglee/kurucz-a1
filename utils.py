
#!/usr/bin/env python
# utils.py
import torch
import numpy as np
from physics import calculate_gradient_torch
from matplotlib import pyplot as plt

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