#!/usr/bin/env python
# physics.py
import torch
import torch.nn.functional as F
import numpy as np
# =============================================================================
# Physics-Informed Loss Functions
# =============================================================================
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
    TAU_START = 4    # Starting index of tau parameters in input (after teff, gravity, feh, afe)

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
        # Calculate gradients for both training and validation
        # Note: changed to always use create_graph=True to ensure gradients
        # are properly computed in both training and validation
        gradients = torch.autograd.grad(
            outputs=P_norm,
            inputs=grad_inputs,
            grad_outputs=grad_outputs,
            create_graph=True,  # Always create graph for both training and validation
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
        # IMPORTANT: Instead of returning 0, calculate a fallback physics loss
        # Log the error with more detail including batch size
        print(f"[WARNING] d(logP)/d(tau) gradient calculation failed: {e}")
        print(f"Input shape: {inputs.shape}, Output shape: {outputs.shape}")
        
        # Restore model state
        model.train(original_training)
        
        # Compute a simplified fallback physics loss instead of returning zero
        # This ensures we still have meaningful physics loss during validation
        ABROSS_norm = torch.clamp(outputs[:, :, ABROSS_idx], -5.0, 5.0)
        kappa = dataset.denormalize('ABROSS', ABROSS_norm)
        
        # Simple pressure gradient estimation (approximate, without exact gradients)
        P_values = dataset.denormalize('P', outputs[:, :, P_idx])
        tau_values = dataset.denormalize('TAU', inputs[:, TAU_START:TAU_START+80])
        
        # Calculate pressure differences (approximate gradient)
        # Shift and compute differences across depth points
        P_next = P_values[:, 1:]
        P_prev = P_values[:, :-1]
        tau_next = tau_values[:, 1:]
        tau_prev = tau_values[:, :-1]
        
        # Compute approximate derivatives (skip first and last 5 points for stability)
        dP = P_next[:, 5:-5] - P_prev[:, 5:-5]
        dtau = tau_next[:, 5:-5] - tau_prev[:, 5:-5]
        
        # Ensure non-zero denominator
        dtau = torch.clamp(dtau, min=1e-10)
        
        # Compute dP/dtau
        dP_dtau = dP / dtau
        
        # Simple physics loss based on pressure gradient
        fallback_loss = torch.mean(torch.abs(dP_dtau))
        
        return fallback_loss

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
    depth_mask[:, :2] = False    # Exclude first 2 depth points
    depth_mask[:, -20:] = False   # Exclude last 25 depth points
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
    
    # Add some debug information when in validation mode
    if not is_training:
        print(f"Physics validation - MSE: {mse_loss.item():.6f}, Sign: {sign_loss.item():.6f}, Total: {total_loss.item():.6f}")
    
    # Restore original model state
    model.train(original_training)
    
    return total_loss