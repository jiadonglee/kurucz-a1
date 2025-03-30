#!/usr/bin/env python
# train.py - Training script for Kurucz atmospheric models

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
# =============================================================================
# Training Functions
# =============================================================================
# Modify custom_loss to incorporate hydrostatic equilibrium
# 在custom_loss函数中
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
            physics_loss = hydro_equilibrium_loss(pred, inputs, dataset, model)  
            total_loss = (1.0 - physics_weight) * total_data_loss + physics_weight * physics_loss
            param_losses['physics'] = physics_loss
    else:
        total_loss = total_data_loss
    
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
   
def validate(model, dataloader, device, dataset, args, logger=None):
    """Validation loop for model evaluation"""
    model.eval()
    total_loss = 0
    param_losses = {'RHOX': 0.0, 'T': 0.0, 'P': 0.0, 
                   'XNE': 0.0, 'ABROSS': 0.0, 'ACCRAD': 0.0, 'physics': 0.0}
    
    # 添加用于调试的变量
    physics_losses = []
    batch_sizes = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # 确保使用与训练相同的物理权重
            loss, batch_param_losses = custom_loss(outputs, targets, dataset, model, inputs, True, args.physics_weight, None, False)
            
            # 记录每个批次的物理损失和批次大小
            physics_losses.append(batch_param_losses['physics'].item())
            batch_sizes.append(inputs.size(0))
            
            total_loss += loss.item()
            for param, param_loss in batch_param_losses.items():
                param_losses[param] += param_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_param_losses = {k: v / len(dataloader) for k, v in param_losses.items()}
    
    # 打印物理损失的统计信息
    if len(physics_losses) > 0 and logger is not None:
        min_physics = min(physics_losses)
        max_physics = max(physics_losses)
        mean_physics = sum(physics_losses) / len(physics_losses)
        logger.info(f"Validation physics loss stats - Min: {min_physics:.6f}, Max: {max_physics:.6f}, Mean: {mean_physics:.6f}")
        logger.info(f"Validation batch sizes - Min: {min(batch_sizes)}, Max: {max(batch_sizes)}, Mean: {sum(batch_sizes)/len(batch_sizes):.2f}")
    
    return avg_loss, avg_param_losses

# 在train函数中也添加类似的调试代码
def train(model, train_loader, val_loader, optimizer, scheduler, device, start_epoch, args, logger, dataset):    
    """Main training loop with validation and learning rate scheduling"""
    best_loss = float('inf')
    patience_counter = 0
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(os.path.join(args.log_dir, 'tensorboard'))
    
    # Log model graph
    example_input = next(iter(train_loader))[0][:1].to(device)
    writer.add_graph(model, example_input)
    
    logger.info(f"Starting training for {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        
        # 初始化每个参数的损失跟踪
        param_running_losses = {'RHOX': 0.0, 'T': 0.0, 'P': 0.0, 
                               'XNE': 0.0, 'ABROSS': 0.0, 'ACCRAD': 0.0, 'physics': 0.0}
        
        # 添加用于调试的变量
        physics_losses = []
        batch_sizes = []
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 移动数据到设备
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播和损失计算
            optimizer.zero_grad()
            outputs = model(inputs)
            # 在train函数中
            loss, param_losses = custom_loss(outputs, targets, dataset, model, inputs, True, args.physics_weight, None, True)
                        
            # 记录每个批次的物理损失和批次大小
            physics_losses.append(param_losses['physics'].item())
            batch_sizes.append(inputs.size(0))
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # Step for batch-based schedulers (like CyclicLR and OneCycleLR)
            if args.scheduler in ['cyclic', 'onecycle']:
                scheduler.step()
            
            # Update statistics
            running_loss += loss.item()
            for param, param_loss in param_losses.items():
                param_running_losses[param] += param_loss.item()
            
            # Batch logging removed to only show epoch-level metrics
        
        # Calculate training epoch statistics
        train_epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        # Calculate average parameter losses for training
        train_avg_param_losses = {k: v / len(train_loader) for k, v in param_running_losses.items()}
        
        # Calculate data loss component (excluding physics loss)
        data_loss = sum(v for k, v in train_avg_param_losses.items() if k != 'physics')
        physics_loss = train_avg_param_losses['physics']
        
        # Validation phase
        val_loss, val_param_losses = validate(model, val_loader, device, dataset, args, logger)
        
        # Calculate validation data loss component
        val_data_loss = sum(v for k, v in val_param_losses.items() if k != 'physics')
        val_physics_loss = val_param_losses['physics']
        
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
        
        # Enhanced logging
        logger.info(f'Epoch {epoch+1}, Train Loss: {train_epoch_loss:.6f}, Val Loss: {val_loss:.6f}')
        logger.info(f'Data Loss: Train={data_loss:.6f}, Val={val_data_loss:.6f}')
        logger.info(f'Physics Loss: Train={physics_loss:.6f}, Val={val_physics_loss:.6f} (weight: {args.physics_weight})')
        
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
        if patience_counter >= args.patience:
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
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--checkpoint_freq', type=int, default=100, help='Checkpoint frequency (epochs)')
    parser.add_argument('--log_freq', type=int, default=10, help='Logging frequency (batches)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--validation_split', type=float, default=0.1, help='Validation set split ratio')
    
    # Hardware parameters
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker processes for data loading')
    
    # Physics-informed loss parameters
    parser.add_argument('--physics_weight', type=float, default=1e-3, help='Weight for physics-informed loss component')
    
    # Learning rate scheduler parameters
    parser.add_argument('--scheduler', type=str, default='none', 
                        choices=['none', 'step', 'multistep', 'exponential', 'cosine', 'plateau', 'cyclic', 'onecycle'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_gamma', type=float, default=0.1, 
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--lr_step_size', type=int, default=100, 
                        help='Period of learning rate decay (epochs)')
    parser.add_argument('--lr_milestones', type=str, default='200,400,600', 
                        help='Epochs at which to decay learning rate (comma-separated)')
    parser.add_argument('--lr_min', type=float, default=1e-7, 
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
    train_loader, val_loader, dataset = create_dataloader_from_saved(
        filepath=args.dataset,
        batch_size=args.batch_size,
        num_workers=0,
        device=device,
        validation_split=args.validation_split
    )
    logger.info(f"Dataset loaded, {len(dataset)} total samples, "
               f"{len(train_loader.dataset)} training, {len(val_loader.dataset)} validation")
    
    # Get the input shape from the first batch
    sample_batch = next(iter(train_loader))
    input_shape = sample_batch[0].shape
    
    # The input now includes tau at each depth point (5 features per depth point)
    # The last dimension of the input tensor is the number of features per depth point
    input_features_per_point = input_shape[-1]

    model = AtmosphereNetMLPtau(
            stellar_embed_dim=128, tau_embed_dim=64
            ).to(device)
            
    logger.info("Created AtmosphereNetMLPtau model")
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    logger.info(f"Using Adam optimizer with initial learning rate {args.lr}")
    
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
    train(model, train_loader, val_loader, optimizer, scheduler, device, start_epoch, args, logger, dataset)

if __name__ == "__main__":
    main()
    # Example usage:
    # python train_hydro.py --dataset data/kurucz_vturb_0p5_tau_v3.pt --gpu --epochs 1000 --lr 1e-4 --batch_size 256 --output_dir ./checkpoints_v0327enc_hydro --physics_weight 1e-1 --scheduler plateau --lr_patience 10 --patience 50 --checkpoint_freq 50 --log_freq 10 --hidden_size 128 --validation_split 0.1 --num_workers 4 --log_dir ./logs_v0327enc_hydro
    # python train_hydro.py --dataset data/kurucz_vturb_0p5_tau_v3.pt --gpu --epochs 1000 --lr 1e-5 --batch_size 256 --output_dir ./checkpoints_v0327enc_hydro   --scheduler plateau --lr_patience 10 --patience 50 --checkpoint_freq 50 --log_freq 10 --hidden_size 128 --validation_split 0.1  --log_dir ./logs_v0327enc_hydro --resume ./checkpoints_v0327enc/best_model.pt --physics_weight 10.0