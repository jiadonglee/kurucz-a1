import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np
import re

def read_kurucz_model(file_path):
    """
    Reads a Kurucz stellar atmospheric model grid file and returns a dictionary with parsed data.
    Extracts [Fe/H] and [α/Fe] values from the filename.
    
    Parameters:
    file_path (str): Path to the Kurucz model file.
    
    Returns:
    dict: A dictionary containing the parsed header, abundance data, and atmospheric structure.
    """
    # Parse [Fe/H] and [α/Fe] from filename
    filename = file_path.split('/')[-1]
    
    # More robust regex patterns for metal content values
    feh_pattern = r'feh([+-]?\d+\.?\d*)'
    afe_pattern = r'afe([+-]?\d+\.?\d*)'
    
    feh_match = re.search(feh_pattern, filename)
    afe_match = re.search(afe_pattern, filename)
    
    feh = float(feh_match.group(1)) if feh_match else None
    afe = float(afe_match.group(1)) if afe_match else None
    
    model = {
        'teff': None,
        'gravity': None,
        'title': None,
        'feh': feh,  # Add [Fe/H] from filename
        'afe': afe,  # Add [α/Fe] from filename
        'filename': filename,
        'opacity_ifop': [],
        'convection': {'status': 'OFF', 'parameters': []},
        'turbulence': {'status': 'OFF', 'parameters': []},
        'abundance_scale': None,
        'abundance_changes': {},
        # Directly index the columns we're interested in
        'RHOX': [],
        'T': [],
        'P': [],
        'XNE': [],
        'ABROSS': [],
        'ACCRAD': [],
        'VTURB': [],
        # Store original data columns for reference
        'data_columns': []
    }

    with open(file_path, 'r') as f:
        # Read lines until TEFF line is found
        line = ''
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('TEFF'):
                break
        
        # Parse TEFF and GRAVITY
        parts = line.split()
        model['teff'] = float(parts[1])
        model['gravity'] = float(parts[3].strip())
        
        # Read TITLE
        line = f.readline().strip()
        model['title'] = line.split('TITLE')[1].strip() if 'TITLE' in line else line
        
        # Parse OPACITY line
        line = f.readline().strip()
        parts = line.split()
        model['opacity_ifop'] = list(map(int, parts[2:22]))  # 20 opacity flags
        
        # Parse CONVECTION and TURBULENCE line
        line = f.readline().strip()
        parts = line.split()
        conv_idx = parts.index('CONVECTION')
        turb_idx = parts.index('TURBULENCE')
        
        model['convection']['status'] = parts[conv_idx + 1]
        model['convection']['parameters'] = list(map(float, parts[conv_idx + 2:turb_idx]))
        
        model['turbulence']['status'] = parts[turb_idx + 1]
        model['turbulence']['parameters'] = list(map(float, parts[turb_idx + 2:]))
        
        # Parse ABUNDANCE SCALE
        line = f.readline().strip()
        parts = line.split()
        model['abundance_scale'] = float(parts[2])
        
        # Parse ABUNDANCE CHANGE lines
        line = f.readline().strip()
        while line.startswith('ABUNDANCE CHANGE'):
            parts = line.split()
            for i in range(2, len(parts), 2):
                elem_num = int(parts[i])
                abundance = float(parts[i+1])
                model['abundance_changes'][elem_num] = abundance
            line = f.readline().strip()
        
        # Skip until READ DECK6 line
        while not line.startswith('READ DECK6'):
            line = f.readline().strip()
        
        # Parse READ DECK6 parameters
        parts = line.split()
        num_depth_points = int(parts[2])
        model['data_columns'] = parts[3:]  # Expected columns
        
        # Define the target columns we're interested in
        target_columns = ['RHOX', 'T', 'P', 'XNE', 'ABROSS', 'ACCRAD', 'VTURB']
        
        # The READ DECK6 line specifies 7 columns, but 
        # the data rows might contain additional values (extra columns)
        
        # Read atmospheric structure data
        for _ in range(num_depth_points):
            line = f.readline()
            values = list(map(float, line.strip().split()))
            
            # The first 7 values correspond to RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB
            # regardless of how many extra columns there are
            if len(values) >= 7:
                model['RHOX'].append(values[0])
                model['T'].append(values[1])
                model['P'].append(values[2])
                model['XNE'].append(values[3])
                model['ABROSS'].append(values[4])
                model['ACCRAD'].append(values[5])
                model['VTURB'].append(values[6])
                
                # Store any extra columns as a list
                if len(values) > 7:
                    if 'extra_columns' not in model:
                        model['extra_columns'] = []
                    model['extra_columns'].append(values[7:])
    
    return model


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
        """
        Return normalized input features and output features
        
        Returns:
            tuple: (input_features, output_features)
                input_features: tensor of shape (4+max_depth_points,) containing
                                [teff, logg, feh, afe, tau_1, tau_2, ..., tau_80]
                output_features: tensor of shape (max_depth_points, 6) containing
                                atmospheric parameters [RHOX, T, P, XNE, ABROSS, ACCRAD]
                                for each depth point
        """
        # Get stellar parameters
        stellar_params = torch.cat([
            self.teff[idx],
            self.gravity[idx],
            self.feh[idx],
            self.afe[idx]
        ], dim=0).flatten()  # Flatten to handle both single item and batch cases
        
        # Get the tau values
        tau = self.TAU[idx]
        
        # Concatenate stellar parameters with tau values to form input
        input_features = torch.cat([stellar_params, tau])
        
        # Output features - atmospheric parameters for each depth point
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
        tau = inputs[:, 4:]
        
        # Denormalize each parameter
        teff_denorm = self.denormalize('teff', teff)
        gravity_denorm = self.denormalize('gravity', gravity)
        feh_denorm = self.denormalize('feh', feh)
        afe_denorm = self.denormalize('afe', afe)
        tau_denorm = self.denormalize('TAU', tau)
        
        # Return as a dictionary for clarity
        return {
            'teff': teff_denorm,
            'gravity': gravity_denorm,
            'feh': feh_denorm,
            'afe': afe_denorm,
            'tau': tau_denorm
        }
    
    def inverse_transform_outputs(self, outputs):
        """Transform normalized outputs back to physical units"""
        batch_size, num_features = outputs.shape[:2]
        
        # Initialize containers for denormalized data
        result = {
            'RHOX': None,
            'T': None,
            'P': None,
            'XNE': None,
            'ABROSS': None,
            'ACCRAD': None
        }
        
        # Get column names
        columns = list(result.keys())
        
        # Denormalize each feature
        for i, param_name in enumerate(columns):
            # Extract the i-th feature for all samples and depths
            feature_data = outputs[:, :, i] if outputs.dim() > 2 else outputs[:, i]
            # Denormalize
            result[param_name] = self.denormalize(param_name, feature_data)
        
        return result


def save_dataset(dataset, filepath):
    """
    Save the KuruczDataset to a file.
    
    Parameters:
        dataset: KuruczDataset instance to save
        filepath (str): Path where to save the dataset
    """
    save_dict = {
        'norm_params': dataset.norm_params,
        'max_depth_points': dataset.max_depth_points,
        'data': {
            'teff': dataset.teff,
            'gravity': dataset.gravity,
            'feh': dataset.feh,
            'afe': dataset.afe,
            'RHOX': dataset.RHOX,
            'T': dataset.T,
            'P': dataset.P,
            'XNE': dataset.XNE,
            'ABROSS': dataset.ABROSS,
            'ACCRAD': dataset.ACCRAD,
            'TAU': dataset.TAU,
            'original': dataset.original
        }
    }
    torch.save(save_dict, filepath)
    print(f"Dataset saved to {filepath}")


def load_dataset_file(filepath, device='cpu'):
    """
    Load a saved dataset.
    
    Parameters:
        filepath (str): Path to the saved dataset
        device (str): Device to load the data to ('cpu' or 'cuda')
        
    Returns:
        KuruczDataset: Loaded dataset
    """
    # Create an empty dataset
    dataset = KuruczDataset.__new__(KuruczDataset)
    
    # Load the saved data
    save_dict = torch.load(filepath, map_location=device)
    
    # Restore attributes
    dataset.norm_params = save_dict['norm_params']
    dataset.max_depth_points = save_dict['max_depth_points']
    dataset.device = device
    
    # Move data to the specified device
    for key, value in save_dict['data'].items():
        if key == 'original':
            # Handle the 'original' dictionary specially
            original_dict = {}
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    original_dict[k] = v.to(device)
                else:
                    original_dict[k] = v
            setattr(dataset, key, original_dict)
        else:
            # Normal tensor values
            setattr(dataset, key, value.to(device))
    
    # Initialize empty models list (not needed after loading)
    dataset.models = []
    
    return dataset


def create_dataloader_from_saved(filepath, batch_size=32, num_workers=4, device='cpu', validation_split=0.1):
    """
    Create train and validation DataLoaders from a saved dataset.
    
    Parameters:
        filepath (str): Path to the saved dataset
        batch_size (int): Batch size for the DataLoader
        num_workers (int): Number of worker processes for data loading
        device (str): Device to load the data to ('cpu' or 'cuda')
        validation_split (float): Fraction of data to use for validation
        
    Returns:
        tuple: (train_loader, val_loader, dataset)
    """
    from torch.utils.data import DataLoader, random_split
    
    # Load the dataset
    dataset = load_dataset_file(filepath, device)
    
    # Calculate split sizes
    dataset_size = len(dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    # Split dataset
    generator = torch.Generator().manual_seed(42)  # For reproducibility
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device=='cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device=='cuda' else False
    )
    
    return train_loader, val_loader, dataset
