#!/usr/bin/env python
# physics.py
import torch
import torch.nn.functional as F
import numpy as np
# =============================================================================
# Physics-Informed Loss Functions
# =============================================================================
import torch
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F

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
    depth_mask[:, :1] = False    # Exclude first 2 depth points
    depth_mask[:, -25:] = False   # Exclude last 25 depth points
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
    
    return total_loss

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