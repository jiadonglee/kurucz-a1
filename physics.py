#!/usr/bin/env python
# physics.py
import torch
import torch.nn.functional as F
import numpy as np
from model import AtmosphereNet, KuruczDataset
# =============================================================================
# Physics-Informed Loss Functions
# =============================================================================
import torch
import torch.nn.functional as F
import numpy as np

def hydrostatic_equilibrium_loss(
    outputs, inputs, dataset, model, is_training=True
):
    """
    改进版的流体静力学平衡损失函数，包含多重数值保护机制
    特征：
    1. 对数空间梯度直接计算
    2. 动态梯度截断
    3. 物理先验约束
    4. 自适应无效数据处理
    
    参数:
    - outputs: 模型的前向输出 (batch_size, depth_points, num_outputs)
    - inputs:  模型的输入 (batch_size, depth_points, num_inputs)
    - dataset: 数据集对象，包含归一化/反归一化函数
    - model:   用于梯度计算的可微分模型
    - is_training: 是否处于训练模式，影响损失的加权和异常处理
    """
    device = outputs.device
    # 一些可调参数
    MIN_LOG_VALUE = -20.0    # 10^-20
    MAX_LOG_VALUE = 20.0     # 10^20
    GRAD_CLIP_INIT = 1e5
    GRAD_CLIP_DECAY = 0.95
    EPS = 1e-30              # 防止 log(0)

    # --------------------
    # 参数/索引定义
    # --------------------
    # 例如: outputs[..., 2] = P, outputs[..., 4] = ABROSS
    P_idx = 2
    ABROSS_idx = 4
    # 假设 inputs[..., 4] = optical depth (tau)
    TAU_idx = 4

    # 动态梯度截断器
    class GradientClipper:
        def __init__(self):
            self.current_max = GRAD_CLIP_INIT
        def clip(self, tensor):
            self.current_max *= GRAD_CLIP_DECAY
            return torch.clamp(tensor, -self.current_max, self.current_max)

    grad_clipper = GradientClipper()

    # ========== (1) 数据准备阶段 ==========

    # (a) 光学深度 (tau) 约束：避免极端值
    #    注意此处只对 tau 做 clamp，不要对 inputs 直接 detach()，
    #    保持梯度可回传到 model 里，若你需要固定 tau 则可以保留 detach。
    tau_norm = inputs[..., TAU_idx]
    tau_norm = torch.clamp(tau_norm, -3.0, 3.0)  # 3σ原则约束

    # (b) 重力加速度 g 处理
    #    假设 inputs[..., 1] = logg_norm
    logg_norm = inputs[:, 0, 1]  # 取batch里第一个 depth_point 的 logg
    logg = dataset.denormalize('gravity', logg_norm)
    # 限制 g 的对数范围 [1,5] => g ∈ [10^1, 10^5]
    logg_clamped = torch.clamp(logg, -1, 5.5)
    g = 10**logg_clamped  # shape: (batch_size,)

    # ========== (2) 梯度计算阶段 ==========

    # 为了计算 dP/dtau，必须让 tau_norm 成为 requires_grad=True
    # 这里需要构造一个专门的输入副本，以便保留梯度图
    grad_inputs = inputs.clone()
    grad_inputs[..., TAU_idx] = tau_norm
    grad_inputs.requires_grad = True

    # 前向传播，得到模型输出(含P)
    outputs_with_grad = model(grad_inputs)
    P_norm = torch.clamp(outputs_with_grad[..., P_idx], -5.0, 5.0)

    # 如果数据集中 P 就是对数归一化(log_scale=True)，则反归一化后本身就是 log10(P)
    if dataset.norm_params['P']['log_scale']:
        # 直接取得物理量的 logP
        log_P = dataset.denormalize('P', P_norm)  
    else:
        # 否则，需要先反归一化成 P，再取 log10
        P_physical = dataset.denormalize('P', P_norm) + EPS
        log_P = torch.log10(P_physical)

    # 计算对数压力对 tau 的梯度
    try:
        dlogP_dtau = torch.autograd.grad(
            outputs=log_P,
            inputs=grad_inputs,
            grad_outputs=torch.ones_like(log_P),
            create_graph=True,
            retain_graph=True
        )[0][..., TAU_idx]  # 只取对 tau_idx 的梯度
        dlogP_dtau = grad_clipper.clip(dlogP_dtau)
    except RuntimeError as e:
        print(f"[警告] d(logP)/d(tau) 梯度计算失败: {e}")
        dlogP_dtau = torch.zeros_like(tau_norm)

    # ========== (3) 物理量转换阶段 ==========

    # (a) 对数梯度 log(dP/dtau) 的修正:
    #     如果 dataset.norm_params['P'] 是对数标度，那么 log_P = P_phys 的对数，
    #     d(logP)/d(tau) 本质上就是 (1/P) * dP/dtau，故需要把“真实的” dP/dtau 关系考虑进来。
    #
    #     如果要严格使用 dP/dtau = g/kappa，就需要：
    #        d(logP)/d(tau) = 1/P * dP/dtau
    #     在 log10 下，d(logP)/d(tau) = (1 / (P ln(10))) * dP/dtau
    #
    #     这里可以根据你的 scale 做额外的校正。如果你只想在对数域直接对比，
    #     可以用 log(dP/dtau) = log_g - log_kappa 的形式做 MSE，避免再指数化。
    #
    #     简化做法：因为 d(logP)/d(tau) ~ dP/dtau / (P * ln(10))，要对比 g/kappa，
    #     所以可以写成:
    #        log10(dP/dtau) = log10(g) - log10(kappa)
    #     而 dP/dtau = P * ln(10) * d(logP)/d(tau)。
    #
    #     综合起来:
    #       log10( dP/dtau ) = log10( P * ln(10) ) + log10( d(logP)/d(tau) )
    #                        = log10(P) + log10( ln(10) ) + log10( d(logP)/d(tau) )
    #
    #     注意：需要保证 d(logP)/d(tau) > 0 才能取 log10。
    #
    #     下方示例：log_dP_dtau = log_P + log10(ln(10)) + log10( dlogP_dtau )。

    ln10_tensor = torch.tensor(np.log(10.0), dtype=torch.float32, device=device)
    # clamp 避免 log(负数或0)
    safe_dlogP_dtau = torch.clamp(dlogP_dtau, min=1e-12)
    log_dP_dtau = log_P + torch.log10(ln10_tensor) + torch.log10(safe_dlogP_dtau)

    # (b) 计算 kappa
    ABROSS_norm = torch.clamp(outputs[..., ABROSS_idx], -5.0, 5.0)
    log_kappa = dataset.denormalize('ABROSS', ABROSS_norm)
    log_kappa_safe = torch.clamp(log_kappa, MIN_LOG_VALUE, MAX_LOG_VALUE)

    # (c) 计算 log(g/kappa)
    #     g shape: (batch_size, 1 or []) => 需要和 depth_points 对齐
    g_expanded = g.unsqueeze(1).expand_as(outputs[..., P_idx])
    log_g_kappa = torch.log10(g_expanded) - log_kappa_safe
    log_g_kappa = torch.clamp(log_g_kappa, MIN_LOG_VALUE, MAX_LOG_VALUE)

    # ========== (4) 损失计算阶段 ==========
    # 构造有效性掩码，过滤掉无效或越界数据
    valid_mask = (
        torch.isfinite(log_dP_dtau) &
        torch.isfinite(log_g_kappa) &
        (log_dP_dtau > MIN_LOG_VALUE) &
        (log_dP_dtau < MAX_LOG_VALUE) &
        (log_g_kappa > MIN_LOG_VALUE) &
        (log_g_kappa < MAX_LOG_VALUE)
    )

    if valid_mask.any():
        mse_loss = F.mse_loss(log_dP_dtau[valid_mask], log_g_kappa[valid_mask])
    else:
        # 如果整批数据都无效，返回一个极小损失，避免 NaN
        mse_loss = torch.tensor(1e-30, device=device, requires_grad=True)

    # 物理符号一致性约束：
    #   dP/dtau 与 g/kappa 同号 => log_dP_dtau - log_g_kappa >= 0
    #   若 < 0，则用 ReLU 惩罚
    sign_loss = F.relu(log_g_kappa - log_dP_dtau).mean()

    # 合并损失
    # 可以根据经验/需要调节权重
    if is_training:
        total_loss = mse_loss + 0.01 * sign_loss
    else:
        # 验证阶段减少权重，防止过度惩罚
        total_loss = mse_loss * 0.1 + 0.001 * sign_loss

    # ========== (5) 数值稳定性检查 ==========
    if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
        print(f"[警告] 损失值出现 NaN/Inf: mse={mse_loss.item():.3e}, sign={sign_loss.item():.3e}")
        print(f"    log_dP_dtau范围: [{log_dP_dtau.min().item():.2f}, {log_dP_dtau.max().item():.2f}]")
        print(f"    log_g_kappa范围: [{log_g_kappa.min().item():.2f}, {log_g_kappa.max().item():.2f}]")
        print(f"    有效数据比例: {valid_mask.float().mean().item():.1%}")
        print(f"    当前模式: {'训练' if is_training else '验证'}")
        # 紧急梯度截断，避免梯度爆炸
        total_loss = torch.clamp(total_loss, -1e3, 1e3)

    return total_loss


def simplified_hydrostatic_equilibrium_loss(outputs, inputs, dataset, model, is_training=True):
    """
    改进的简化版流体静力学平衡损失函数
    
    特点:
    1. 增强的数值稳定性
    2. 改进的符号惩罚机制
    3. 完善的有效性检查
    4. 区分训练与验证模式
    
    参数:
    - outputs: 模型输出张量
    - inputs: 模型输入张量
    - dataset: 数据集对象，用于归一化/反归一化
    - model: 模型对象
    - is_training: 是否处于训练模式，影响损失计算
    """
    device = outputs.device
    
    # 安全常数
    MIN_LOG_VALUE = -15.0  # 10^-15
    MAX_LOG_VALUE = 15.0   # 10^15
    MIN_VALUE = 1e-15
    MAX_VALUE = 1e15
    
    # 参数索引
    P_idx = 2
    ABROSS_idx = 4
    TAU_idx = 4  # 在输入中的光学深度索引
    
    try:
        # 输入输出裁剪
        P_norm = outputs[:, :, P_idx].clamp(-5.0, 5.0)
        ABROSS_norm = outputs[:, :, ABROSS_idx].clamp(-5.0, 5.0)
        tau_norm = inputs[:, :, TAU_idx].clamp(-3.0, 3.0)  # 3σ原则
        
        # 反归一化（带物理约束）
        def safe_denorm(param, value):
            """安全反归一化函数，处理对数和线性尺度"""
            try:
                raw = dataset.denormalize(param, value)
                if param in dataset.norm_params and dataset.norm_params[param].get('log_scale', False):
                    return 10**torch.clamp(raw, MIN_LOG_VALUE, MAX_LOG_VALUE)
                return torch.clamp(raw, MIN_VALUE, MAX_VALUE)
            except Exception as e:
                print(f"反归一化错误 ({param}): {e}")
                # 返回安全的默认值
                return torch.ones_like(value) * MIN_VALUE
        
        # 物理量计算
        P = safe_denorm('P', P_norm)
        tau = safe_denorm('TAU', tau_norm)
        kappa = safe_denorm('ABROSS', ABROSS_norm)
        
        # 重力参数处理
        logg_norm = inputs[:, 0, 1].detach().clone()
        logg = dataset.denormalize('gravity', logg_norm)
        g = 10**torch.clamp(logg, min=1.0, max=5.0).view(-1, 1)  # logg ∈ [1,5] 物理约束
        
        # 梯度计算（带排序保护）
        sorted_tau, indices = torch.sort(tau, dim=1)  # 按深度排序
        sorted_P = torch.gather(P, 1, indices)
        
        # 数值微分
        dP_dtau = calculate_gradient_torch(sorted_tau, sorted_P)
        dP_dtau = torch.clamp(dP_dtau, -MAX_VALUE, MAX_VALUE)  # 梯度截断
        
        # 理论值计算
        g_kappa = torch.clamp(g / torch.gather(kappa, 1, indices), MIN_VALUE, MAX_VALUE)
        
        # 对数空间比较（带有效性检查）
        log_dP_dtau = torch.log10(torch.abs(dP_dtau) + MIN_VALUE)
        log_g_kappa = torch.log10(g_kappa + MIN_VALUE)
        
        # 有效性掩码
        valid_mask = (
            torch.isfinite(log_dP_dtau) &
            torch.isfinite(log_g_kappa) &
            (log_dP_dtau > MIN_LOG_VALUE) &
            (log_dP_dtau < MAX_LOG_VALUE) &
            (log_g_kappa > MIN_LOG_VALUE) &
            (log_g_kappa < MAX_LOG_VALUE)
        )
        
        # 损失计算
        if valid_mask.any():
            mse_loss = F.mse_loss(log_dP_dtau[valid_mask], log_g_kappa[valid_mask])
        else:
            print("警告: 无有效数据点，使用默认损失")
            mse_loss = torch.tensor(1e-6, device=device, requires_grad=True)
        
        # 改进的符号惩罚 - 确保梯度和理论值符号一致
        sign_penalty = torch.relu(-(dP_dtau * g_kappa)).mean()
        
        # 组合损失 - 训练和验证阶段使用不同的缩放
        if is_training:
            total_loss = mse_loss + 0.1 * sign_penalty
        else:
            total_loss = mse_loss * 0.5 + 0.05 * sign_penalty
        
        # 监控指标
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            print(f"损失值异常: mse={mse_loss.item():.4f}, sign={sign_penalty.item():.4f}")
            print(f"log_dP_dtau范围: [{log_dP_dtau.min().item():.2f}, {log_dP_dtau.max().item():.2f}]")
            print(f"log_g_kappa范围: [{log_g_kappa.min().item():.2f}, {log_g_kappa.max().item():.2f}]")
            print(f"有效数据比例: {valid_mask.float().mean().item():.1%}")
            
            # 应急处理
            total_loss = torch.clamp(total_loss, 0.0, 100.0)
            
        return total_loss
        
    except Exception as e:
        print(f"损失计算总体错误: {e}")
        return torch.tensor(1.0, device=device, requires_grad=True)

def calculate_gradient_torch(x, y):
    """
    计算y对x的梯度，使用中心差分方法
    
    参数:
    - x: 自变量 (batch_size, n_points)
    - y: 因变量 (batch_size, n_points)
    
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