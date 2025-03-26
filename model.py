#!/usr/bin/env python

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Model Architecture
# =============================================================================
class AtmosphereNet(torch.nn.Module):
    def __init__(self, input_size=5, hidden_size=256, output_size=6, depth_points=80):
        super(AtmosphereNet, self).__init__()
        self.input_size = input_size  # Now includes tau (teff, gravity, feh, afe, tau)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.depth_points = depth_points
        
        # Shared feature extractor for each depth point
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01)
        )
        
        # Final prediction layers
        self.output_layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            
            torch.nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, depth_points, input_size)
        batch_size, depth_points, _ = x.shape
        
        # Process each depth point independently
        # Reshape to (batch_size * depth_points, input_size)
        x_reshaped = x.reshape(-1, self.input_size)
        
        # Extract features for each depth point
        features = self.feature_extractor(x_reshaped)
        
        # Generate predictions for each depth point
        outputs = self.output_layers(features)
        
        # Reshape back to (batch_size, depth_points, output_size)
        outputs = outputs.view(batch_size, depth_points, self.output_size)
        
        return outputs

class AtmosphereNetMLP(torch.nn.Module):
    def __init__(self, input_size=5, hidden_size=256, output_size=6, depth_points=80):
        super(AtmosphereNetMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.depth_points = depth_points
        self.flattened_input_size = input_size * depth_points
        self.flattened_output_size = output_size * depth_points
        
        # MLP architecture
        self.layers = torch.nn.Sequential(
            # Input layer
            torch.nn.Linear(self.flattened_input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            
            # Hidden layers
            torch.nn.Linear(hidden_size, hidden_size*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            
            torch.nn.Linear(hidden_size*2, hidden_size*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            
            # Output layer
            torch.nn.Linear(hidden_size*2, self.flattened_output_size)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, depth_points, input_size)
        batch_size = x.size(0)
        
        # Flatten input
        x_flat = x.view(batch_size, -1)
        
        # Pass through MLP
        output_flat = self.layers(x_flat)
        
        # Reshape to (batch_size, depth_points, output_size)
        outputs = output_flat.view(batch_size, self.depth_points, self.output_size)
        
        return outputs


class AtmosphereNetMLPtau(torch.nn.Module):
    def __init__(self, input_size=5, hidden_size=256, output_size=6, depth_points=80):
        super(AtmosphereNetMLPtau, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.depth_points = depth_points
        self.flattened_input_size = input_size * depth_points
        self.flattened_output_size = output_size * depth_points
        
        # 新增：tau特征增强层
        self.tau_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, 32),  # 单独处理tau特征
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32)
        )
        
        # 主网络架构
        self.layers = torch.nn.Sequential(
            # 输入层
            torch.nn.Linear(self.flattened_input_size + 32 * depth_points, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            
            # 隐藏层，增加深度以捕捉连续变化
            torch.nn.Linear(hidden_size, hidden_size*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            
            torch.nn.Linear(hidden_size*2, hidden_size*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            
            # 输出层
            torch.nn.Linear(hidden_size*2, self.flattened_output_size)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, depth_points, input_size)
        batch_size = x.size(0)
        
        # 提取tau特征
        tau = x[..., 4:5]  # 假设tau是第5个特征
        tau_features = self.tau_encoder(tau.view(-1, 1))  # 处理每个tau值
        tau_features = tau_features.view(batch_size, self.depth_points, -1)
        
        # 拼接原始输入和tau特征
        x_flat = x.view(batch_size, -1)
        tau_flat = tau_features.view(batch_size, -1)
        combined = torch.cat([x_flat, tau_flat], dim=1)
        
        # 通过MLP
        output_flat = self.layers(combined)
        
        # Reshape to (batch_size, depth_points, output_size)
        outputs = output_flat.view(batch_size, self.depth_points, self.output_size)
        
        # 对P和kappa应用连续性约束
        P = outputs[..., 2]  # P是第3个输出
        kappa = outputs[..., 4]  # kappa是第5个输出
        
        # 确保P和kappa随tau单调变化
        P = self.apply_monotonic_constraint(P, tau)
        kappa = self.apply_monotonic_constraint(kappa, tau)
        
        # 将约束后的值放回输出
        outputs[..., 2] = P
        outputs[..., 4] = kappa
        
        return outputs
    
    def apply_monotonic_constraint(self, y, x):
        """
        应用单调性约束，确保y随x单调变化
        """
        # 计算差分
        diff = y[:, 1:] - y[:, :-1]
        
        # 应用ReLU约束
        constrained_diff = torch.nn.functional.relu(diff)
        
        # 重建y值
        y_constrained = torch.cat([
            y[:, :1],  # 保持第一个值不变
            y[:, :1] + torch.cumsum(constrained_diff, dim=1)
        ], dim=1)
        
        return y_constrained

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
#         super(ResidualBlock, self).__init__()
#         self.conv_block = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size, 
#                      padding=kernel_size//2 + (kernel_size//2)*(dilation-1), 
#                      dilation=dilation),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(),
#             nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm1d(out_channels)
#         )
        
#         # 1x1 conv for matching dimensions in skip connection if needed
#         self.skip_connection = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size=1),
#             nn.BatchNorm1d(out_channels)
#         ) if in_channels != out_channels else None
        
#         self.relu = nn.ReLU()
        
#     def forward(self, x):
#         identity = x
#         out = self.conv_block(x)
        
#         if self.skip_connection:
#             identity = self.skip_connection(identity)
            
#         out += identity
#         out = self.relu(out)
#         return out


# class AtmosphereConvNet(nn.Module):
#     def __init__(self, input_size=5, hidden_size=128, output_size=6, depth_points=80):
#         super(AtmosphereConvNet, self).__init__()
#         self.depth_points = depth_points
#         self.input_size = input_size
        
#         # Position encoding layer
#         self.position_emb = nn.Sequential(
#             nn.Embedding(depth_points, hidden_size//2),  # Reduced embedding size
#             nn.LayerNorm(hidden_size//2)
#         )
        
#         # Input projection layer
#         self.input_proj = nn.Sequential(
#             nn.Conv1d(input_size, hidden_size//2, kernel_size=1),
#             nn.BatchNorm1d(hidden_size//2),
#             nn.ReLU()
#         )
        
#         # Combined input size after concatenation
#         combined_size = hidden_size  # hidden_size//2 + hidden_size//2
        
#         # Convolutional blocks with residual connections
#         self.conv_blocks = nn.Sequential(
#             ResidualBlock(combined_size, hidden_size, dilation=1),
#             ResidualBlock(hidden_size, hidden_size, dilation=2),
#             ResidualBlock(hidden_size, hidden_size, dilation=4),
#             ResidualBlock(hidden_size, hidden_size, dilation=8)
#         )
        
#         # Output projection
#         self.output_proj = nn.Sequential(
#             nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
#             nn.BatchNorm1d(hidden_size),
#             nn.ReLU(),
#             nn.Conv1d(hidden_size, output_size, kernel_size=1)
#         )
        
#         # Skip connection directly from input to output
#         self.global_skip = nn.Conv1d(input_size, output_size, kernel_size=1)
        
#         # Initialize weights properly
#         self._init_weights()
        
#     def _init_weights(self):
#         """Initialize weights using Kaiming initialization"""
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         batch_size = x.size(0)
        
#         # Save original input for global skip connection
#         x_orig = x.transpose(1, 2)  # (batch_size, input_size, depth_points)
        
#         # Generate position embeddings
#         positions = torch.arange(self.depth_points, device=x.device).unsqueeze(0)
#         pos_emb = self.position_emb(positions)  # (1, depth_points, hidden_size//2)
#         pos_emb = pos_emb.expand(batch_size, -1, -1)  # (batch_size, depth_points, hidden_size//2)
#         pos_emb = pos_emb.transpose(1, 2)  # (batch_size, hidden_size//2, depth_points)
        
#         # Project input features
#         x_proj = self.input_proj(x_orig)  # (batch_size, hidden_size//2, depth_points)
        
#         # Concatenate projected input with position embeddings
#         x_combined = torch.cat([x_proj, pos_emb], dim=1)  # (batch_size, hidden_size, depth_points)
        
#         # Process through convolutional blocks
#         features = self.conv_blocks(x_combined)
        
#         # Project to output dimensions
#         out = self.output_proj(features)
        
#         # Apply global skip connection
#         global_skip = self.global_skip(x_orig)
#         out = out + global_skip
        
#         # Return in the expected format (batch_size, depth_points, output_size)
#         return out.transpose(1, 2)


class AtmosphereConvNet(nn.Module):
    """
    A basic CNN model with minimal components but enough expressivity to learn:
    1. Uses a few convolutional layers
    2. Includes batch normalization for stability
    3. Uses ReLU activations for non-linearity
    4. Includes proper initialization
    """
    def __init__(self, input_size=5, hidden_size=32, output_size=6, depth_points=80):
        super(AtmosphereConvNet, self).__init__()
        
        # First layer projects to hidden dimension
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        # Second layer maintains hidden dimension
        self.layer2 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        # Output projection
        self.output = nn.Conv1d(hidden_size, output_size, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input comes as [batch_size, depth_points, input_size]
        # Convert to [batch_size, input_size, depth_points] for Conv1d
        x = x.transpose(1, 2)
        
        # Apply layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output(x)
        
        # Convert back to [batch_size, depth_points, output_size]
        x = x.transpose(1, 2)
        
        return x

# =============================================================================
# Dataset Loader
# =============================================================================

class KuruczDataset(Dataset):
    """
    Dataset for Kurucz stellar atmosphere models with standardized [-1, 1] normalization.
    """
    def __init__(self, data_dir, file_pattern='*.atm', max_depth_points=80, device='cpu'):
        self.data_dir = data_dir
        self.file_paths = glob.glob(os.path.join(data_dir, file_pattern))
        self.max_depth_points = max_depth_points
        self.device = device
        self.norm_params = {}
        
        # Load and process all files
        self.models = []
        for file_path in self.file_paths:
            try:
                model = read_kurucz_model(file_path)
                # Calculate optical depth (tau) for each model
                self.calculate_tau(model)
                self.models.append(model)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if len(self.models) == 0:
            raise ValueError(f"No valid model files found in {data_dir} with pattern {file_pattern}")
        
        # Extract input and output features
        self.prepare_data()
        
        # Calculate normalization parameters
        self.setup_normalization()
        
        # Apply normalization to all data
        self.normalize_all_data()

    def calculate_tau(self, model):
        """
        Calculate optical depth (tau) for a model by integrating opacity over mass column density.
        
        Parameters:
            model (dict): The model dictionary containing RHOX and ABROSS arrays
            
        Returns:
            None: Adds 'TAU' key to the model dictionary
        """
        # Get the mass column density (RHOX) and opacity (ABROSS)
        rhox = np.array(model['RHOX'])
        abross = np.array(model['ABROSS'])
        
        # Initialize tau array
        tau = np.zeros_like(rhox)
        
        # The Kurucz models are typically ordered from the outer atmosphere (low density)
        # to the inner atmosphere (high density), so we integrate from outside in
        for i in range(1, len(rhox)):
            # Calculate the increment in tau using the trapezoidal rule
            delta_rhox = rhox[i] - rhox[i-1]
            avg_opacity = (abross[i] + abross[i-1]) / 2.0
            delta_tau = avg_opacity * delta_rhox
            
            # Add to the cumulative tau
            tau[i] = tau[i-1] + delta_tau
        
        # Store tau in the model
        model['TAU'] = tau.tolist()
        
        return None

    def prepare_data(self):
        """Prepare data tensors from the loaded models"""
        # Input features: teff, gravity, feh, afe
        self.teff_list = []
        self.gravity_list = []
        self.feh_list = []
        self.afe_list = []
        
        # Output features: RHOX, T, P, XNE, ABROSS, ACCRAD, TAU
        self.rhox_list = []
        self.t_list = []
        self.p_list = []
        self.xne_list = []
        self.abross_list = []
        self.accrad_list = []
        self.tau_list = []  # Add tau list
        
        # Process each model
        for model in self.models:
            # Pad or truncate depth profiles to max_depth_points
            rhox = self.pad_sequence(model['RHOX'], self.max_depth_points)
            t = self.pad_sequence(model['T'], self.max_depth_points)
            p = self.pad_sequence(model['P'], self.max_depth_points)
            xne = self.pad_sequence(model['XNE'], self.max_depth_points)
            abross = self.pad_sequence(model['ABROSS'], self.max_depth_points)
            accrad = self.pad_sequence(model['ACCRAD'], self.max_depth_points)
            tau = self.pad_sequence(model['TAU'], self.max_depth_points)  # Add tau
            
            # Store input parameters
            self.teff_list.append(model['teff'])
            self.gravity_list.append(model['gravity'])
            self.feh_list.append(model['feh'] if model['feh'] is not None else 0.0)
            self.afe_list.append(model['afe'] if model['afe'] is not None else 0.0)
            
            # Store output parameters
            self.rhox_list.append(rhox)
            self.t_list.append(t)
            self.p_list.append(p)
            self.xne_list.append(xne)
            self.abross_list.append(abross)
            self.accrad_list.append(accrad)
            self.tau_list.append(tau)  # Add tau
        
        # Convert to tensors
        self.original = {
            'teff': torch.tensor(self.teff_list, dtype=torch.float32, device=self.device).unsqueeze(1),
            'gravity': torch.tensor(self.gravity_list, dtype=torch.float32, device=self.device).unsqueeze(1),
            'feh': torch.tensor(self.feh_list, dtype=torch.float32, device=self.device).unsqueeze(1),
            'afe': torch.tensor(self.afe_list, dtype=torch.float32, device=self.device).unsqueeze(1),
            'RHOX': torch.tensor(self.rhox_list, dtype=torch.float32, device=self.device),
            'T': torch.tensor(self.t_list, dtype=torch.float32, device=self.device),
            'P': torch.tensor(self.p_list, dtype=torch.float32, device=self.device),
            'XNE': torch.tensor(self.xne_list, dtype=torch.float32, device=self.device),
            'ABROSS': torch.tensor(self.abross_list, dtype=torch.float32, device=self.device),
            'ACCRAD': torch.tensor(self.accrad_list, dtype=torch.float32, device=self.device),
            'TAU': torch.tensor(self.tau_list, dtype=torch.float32, device=self.device)  # Add tau
        }

    def setup_normalization(self):
        """Calculate normalization parameters for each feature"""
        log_params = ['teff', 'RHOX', 'P', 'XNE', 'ABROSS', 'ACCRAD', 'TAU']
        
        for param_name, data in self.original.items():
            if param_name in log_params:
                transformed_data = torch.log10(data + 1e-30)
                log_scale = True
            else:
                transformed_data = data
                log_scale = False
            
            # 新增 scale 计算
            if transformed_data.dim() > 2:
                param_min = transformed_data.min(dim=0).values
                param_max = transformed_data.max(dim=0).values
            else:
                param_min = transformed_data.min()
                param_max = transformed_data.max()
            
            # 显式存储 scale = (max - min)/2
            self.norm_params[param_name] = {
                'min': param_min,
                'max': param_max,
                'scale': (param_max - param_min) / 2.0,  # 新增关键字段
                'log_scale': log_scale
            }

    def normalize_all_data(self):
        """Apply normalization to all data tensors"""
        # Normalize input parameters
        self.teff = self.normalize('teff', self.original['teff'])
        self.gravity = self.normalize('gravity', self.original['gravity'])
        self.feh = self.normalize('feh', self.original['feh'])
        self.afe = self.normalize('afe', self.original['afe'])
        
        # Normalize output parameters
        self.RHOX = self.normalize('RHOX', self.original['RHOX'])
        self.T = self.normalize('T', self.original['T'])
        self.P = self.normalize('P', self.original['P'])
        self.XNE = self.normalize('XNE', self.original['XNE'])
        self.ABROSS = self.normalize('ABROSS', self.original['ABROSS'])
        self.ACCRAD = self.normalize('ACCRAD', self.original['ACCRAD'])
        self.TAU = self.normalize('TAU', self.original['TAU'])  # Add TAU normalization

    def normalize(self, param_name, data):
        """
        Normalize data to [-1, 1] range with optional log transform.

        Parameters:
            param_name (str): Name of the parameter to normalize
            data (torch.Tensor): Data to normalize

        Returns:
            torch.Tensor: Normalized data in [-1, 1] range
        """
        params = self.norm_params[param_name]
        
        # Apply log transform if needed
        if params['log_scale']:
            transformed_data = torch.log10(data + 1e-30)
        else:
            transformed_data = data
        
        # Apply min-max scaling to [-1, 1] range
        normalized = 2.0 * (transformed_data - params['min']) / (params['max'] - params['min']) - 1.0
        
        return normalized

    def denormalize(self, param_name, normalized_data):
        """
        Denormalize data from [-1, 1] range back to original scale.

        Parameters:
            param_name (str): Name of the parameter to denormalize
            normalized_data (torch.Tensor): Normalized data to convert back

        Returns:
            torch.Tensor: Denormalized data with gradients preserved
        """
        params = self.norm_params[param_name]
        
        # Reverse min-max scaling from [-1, 1] range
        transformed_data = (normalized_data + 1.0) / 2.0 * (params['max'] - params['min']) + params['min']
        
        # Reverse log transform if needed
        if params['log_scale']:
            denormalized_data = torch.pow(10.0, transformed_data) - 1e-30
        else:
            denormalized_data = transformed_data
        
        # Ensure gradient propagation is maintained and data stays on the correct device
        return denormalized_data.to(self.device)

    def pad_sequence(self, values, target_length):
        """Pad a sequence to the target length"""
        if len(values) >= target_length:
            return values[:target_length]
        
        # Pad with the last value
        padding_needed = target_length - len(values)
        return values + [values[-1]] * padding_needed
    
    def __len__(self):
        return len(self.teff)
    
    def __getitem__(self, idx):
        """Return normalized input features and output features"""
        # Get the normalized tau for this model
        tau_features = self.TAU[idx]
        
        # Create stellar parameter features (already normalized)
        stellar_params = torch.cat([
            self.teff[idx],
            self.gravity[idx],
            self.feh[idx],
            self.afe[idx]
        ], dim=0)
        
        # Input features: stellar parameters and tau at each depth point
        # For each depth point, we'll have [teff, gravity, feh, afe, tau]
        # Expand stellar parameters to match the depth dimension
        batch_size = 1 if stellar_params.dim() == 1 else stellar_params.shape[0]
        depth_points = tau_features.shape[0]
        
        # Reshape stellar parameters to [batch, 1, params] and expand to [batch, depth, params]
        expanded_params = stellar_params.unsqueeze(1).expand(-1, depth_points, -1) if batch_size > 1 else \
                          stellar_params.unsqueeze(0).expand(depth_points, -1)
        
        # Add tau as an additional input feature [batch, depth, params+1]
        tau_reshaped = tau_features.unsqueeze(-1) if tau_features.dim() == 1 else tau_features
        
        # Combine stellar parameters with tau
        input_features = torch.cat([expanded_params, tau_reshaped], dim=-1)
        
        # Output features (already normalized) - now without TAU
        output_features = torch.stack([
            self.RHOX[idx],
            self.T[idx],
            self.P[idx],
            self.XNE[idx],
            self.ABROSS[idx],
            self.ACCRAD[idx]
        ], dim=1)
        
        return input_features, output_features
    
    def inverse_transform_inputs(self, inputs):
        """Transform normalized inputs back to physical units"""
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)  # Add batch dimension if needed
            
        # Split the input tensor back into individual parameters
        teff = inputs[:, 0:1]
        gravity = inputs[:, 1:2]
        feh = inputs[:, 2:3]
        afe = inputs[:, 3:4]
        
        # Denormalize each parameter
        teff_denorm = self.denormalize('teff', teff)
        gravity_denorm = self.denormalize('gravity', gravity)
        feh_denorm = self.denormalize('feh', feh)
        afe_denorm = self.denormalize('afe', afe)
        
        # Return as a dictionary for clarity
        return {
            'teff': teff_denorm,
            'gravity': gravity_denorm,
            'feh': feh_denorm,
            'afe': afe_denorm
        }
    
    def inverse_transform_outputs(self, outputs):
        """Transform normalized outputs back to physical units"""
        batch_size, depth_points, num_features = outputs.shape
        
        # Initialize containers for denormalized data
        columns = ['RHOX', 'T', 'P', 'XNE', 'ABROSS', 'ACCRAD', 'TAU']  # Add TAU
        result = {param: None for param in columns}
        
        # Denormalize each feature
        for i, param_name in enumerate(columns):
            # Extract the i-th feature for all samples and depths
            feature_data = outputs[:, :, i]
            # Denormalize
            result[param_name] = self.denormalize(param_name, feature_data)
        
        return result
    
    def save_dataset(self, filepath):
        """
        Save the dataset, including all normalized data and normalization parameters.
        
        Parameters:
            filepath (str): Path to save the dataset
        """
        # Create a dictionary with all necessary data
        save_dict = {
            'norm_params': self.norm_params,
            'max_depth_points': self.max_depth_points,
            'data': {
                # Input features
                'teff': self.teff,
                'gravity': self.gravity,
                'feh': self.feh,
                'afe': self.afe,
                # Output features
                'RHOX': self.RHOX,
                'T': self.T,
                'P': self.P,
                'XNE': self.XNE,
                'ABROSS': self.ABROSS,
                'ACCRAD': self.ACCRAD,
                'TAU': self.TAU,  # Add TAU
                # Original data (optional, can be commented out to save space)
                'original': self.original
            }
        }
        
        # Save to file
        torch.save(save_dict, filepath)
        print(f"Dataset saved to {filepath}")

    @classmethod
    def load_dataset(cls, filepath, device='cpu'):
        """
        Load a saved dataset.
        
        Parameters:
            filepath (str): Path to the saved dataset
            device (str): Device to load the data to ('cpu' or 'cuda')
            
        Returns:
            KuruczDataset: Loaded dataset
        """
        # Create an empty dataset
        dataset = cls.__new__(cls)
        
        # Load the saved data
        save_dict = torch.load(filepath, map_location=device)
        
        # Restore attributes
        dataset.norm_params = save_dict['norm_params']
        dataset.max_depth_points = save_dict['max_depth_points']
        dataset.device = device
        
        # Move data to the specified device
        for key, tensor in save_dict['data'].items():
            if isinstance(tensor, dict):
                # Handle nested dictionaries (like 'original')
                setattr(dataset, key, {k: v.to(device) for k, v in tensor.items()})
            else:
                setattr(dataset, key, tensor.to(device))
        
        # Initialize empty models list (not needed after loading)
        dataset.models = []
        
        return dataset