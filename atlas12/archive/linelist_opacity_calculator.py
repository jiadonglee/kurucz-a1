#!/usr/bin/env python3
"""
Linelist-based Line Opacity Calculator

This module shows how to use real astronomical linelists (VALD, Kurucz, NIST)
to calculate line opacity using the ATLAS12 methodology.

Supports multiple linelist formats:
- VALD (Vienna Atomic Line Database)
- Kurucz linelists
- NIST Atomic Spectra Database
- Custom formats
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from line_opacity_calculator import LineOpacityCalculator, SpectralLine

class LinelistReader:
    """Read and parse different linelist formats"""
    
    def __init__(self):
        self.supported_formats = ['vald', 'kurucz', 'nist', 'custom']
    
    def read_vald_linelist(self, filename: str, wavelength_range: Tuple[float, float] = None) -> pd.DataFrame:
        """
        Read VALD (Vienna Atomic Line Database) format linelist
        
        VALD format example:
        'Fe 1',        5250.2084,  0.000,  -0.004, 1.00e+08, 1.00e-31, 0.085, 'E',  12600.0, 'E',  32600.0
        
        Args:
            filename: Path to VALD linelist file
            wavelength_range: Optional (min_wave, max_wave) in Angstroms
            
        Returns:
            DataFrame with standardized column names
        """
        print(f"Reading VALD linelist: {filename}")
        
        # VALD column names
        columns = [
            'species', 'wavelength', 'excitation_lower', 'log_gf', 
            'gamma_rad', 'gamma_stark', 'gamma_vdw', 'flag_lower',
            'energy_lower', 'flag_upper', 'energy_upper'
        ]
        
        lines = []
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                
                # Skip comments and headers
                if line.startswith('#') or line.startswith('*') or not line:
                    continue
                
                try:
                    # Parse VALD format (handle quoted strings)
                    parts = self._parse_vald_line(line)
                    if len(parts) >= 11:
                        species = parts[0].strip("'\"")
                        wavelength = float(parts[1])
                        
                        # Apply wavelength filter
                        if wavelength_range:
                            if wavelength < wavelength_range[0] or wavelength > wavelength_range[1]:
                                continue
                        
                        data = {
                            'species': species,
                            'wavelength': wavelength,
                            'excitation_lower': float(parts[2]),
                            'log_gf': float(parts[3]),
                            'gamma_rad': float(parts[4]),
                            'gamma_stark': float(parts[5]),
                            'gamma_vdw': float(parts[6]),
                            'flag_lower': parts[7].strip("'\""),
                            'energy_lower': float(parts[8]),
                            'flag_upper': parts[9].strip("'\""),
                            'energy_upper': float(parts[10])
                        }
                        lines.append(data)
                        
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line {line_num}: {line[:50]}...")
                    continue
        
        df = pd.DataFrame(lines)
        print(f"Successfully read {len(df)} lines from VALD format")
        return df
    
    def _parse_vald_line(self, line: str) -> List[str]:
        """Parse a VALD format line handling quoted strings"""
        # Split by comma, but handle quoted strings
        parts = []
        current_part = ""
        in_quotes = False
        quote_char = None
        
        for char in line:
            if char in ("'", '"') and not in_quotes:
                in_quotes = True
                quote_char = char
                current_part += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                current_part += char
                quote_char = None
            elif char == ',' and not in_quotes:
                parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
        
        if current_part.strip():
            parts.append(current_part.strip())
        
        return parts
    
    def read_kurucz_linelist(self, filename: str, wavelength_range: Tuple[float, float] = None) -> pd.DataFrame:
        """
        Read Kurucz format linelist
        
        Kurucz format (fixed width):
        Columns 1-11: wavelength (Å)
        Columns 12-17: log gf
        Columns 18-25: element code
        Columns 26-37: excitation potential (eV)
        etc.
        
        Args:
            filename: Path to Kurucz linelist file
            wavelength_range: Optional (min_wave, max_wave) in Angstroms
            
        Returns:
            DataFrame with standardized column names
        """
        print(f"Reading Kurucz linelist: {filename}")
        
        lines = []
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f):
                if len(line) < 50:  # Skip short lines
                    continue
                
                try:
                    # Parse fixed-width Kurucz format
                    wavelength = float(line[0:11])
                    log_gf = float(line[11:17])
                    element_code = float(line[17:25])
                    excitation_lower = float(line[25:37])
                    
                    # Apply wavelength filter
                    if wavelength_range:
                        if wavelength < wavelength_range[0] or wavelength > wavelength_range[1]:
                            continue
                    
                    # Parse element information
                    species, ion_stage = self._parse_kurucz_element_code(element_code)
                    
                    # Extract additional parameters if available
                    gamma_rad = 1e8  # Default radiative damping
                    gamma_stark = 1e-15  # Default Stark broadening
                    gamma_vdw = 1e-7  # Default van der Waals
                    
                    if len(line) > 50:
                        try:
                            gamma_rad = float(line[50:60]) if line[50:60].strip() else 1e8
                        except:
                            pass
                    
                    data = {
                        'species': f"{species} {ion_stage}",
                        'wavelength': wavelength,
                        'excitation_lower': excitation_lower,
                        'log_gf': log_gf,
                        'gamma_rad': gamma_rad,
                        'gamma_stark': gamma_stark,
                        'gamma_vdw': gamma_vdw,
                        'flag_lower': 'E',
                        'energy_lower': excitation_lower * 8065.54,  # Convert eV to cm⁻¹
                        'flag_upper': 'E',
                        'energy_upper': (excitation_lower + 1.24e-4 / wavelength * 1e8) * 8065.54
                    }
                    lines.append(data)
                    
                except (ValueError, IndexError) as e:
                    continue
        
        df = pd.DataFrame(lines)
        print(f"Successfully read {len(df)} lines from Kurucz format")
        return df
    
    def _parse_kurucz_element_code(self, code: float) -> Tuple[str, str]:
        """Parse Kurucz element code to get species and ionization"""
        # Kurucz code: element_number.ionization (e.g., 26.0 = Fe I, 26.1 = Fe II)
        element_num = int(code)
        ion_stage = int((code - element_num) * 10) + 1
        
        # Element symbol lookup
        elements = {
            1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
            9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
            16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
            23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
            30: 'Zn'
        }
        
        element_symbol = elements.get(element_num, f"El{element_num}")
        ion_string = ['I', 'II', 'III', 'IV', 'V'][min(ion_stage-1, 4)]
        
        return element_symbol, ion_string
    
    def read_custom_linelist(self, filename: str, format_dict: Dict[str, int],
                           wavelength_range: Tuple[float, float] = None) -> pd.DataFrame:
        """
        Read custom format linelist
        
        Args:
            filename: Path to linelist file
            format_dict: Dictionary mapping column names to column indices
            wavelength_range: Optional (min_wave, max_wave) in Angstroms
            
        Returns:
            DataFrame with standardized column names
        """
        print(f"Reading custom linelist: {filename}")
        
        # Read as space/tab separated
        try:
            df = pd.read_csv(filename, sep=r'\s+', comment='#', header=None)
        except:
            df = pd.read_csv(filename, sep=',', comment='#', header=None)
        
        # Map columns according to format_dict
        mapped_data = {}
        for col_name, col_idx in format_dict.items():
            if col_idx < len(df.columns):
                mapped_data[col_name] = df.iloc[:, col_idx]
        
        result_df = pd.DataFrame(mapped_data)
        
        # Apply wavelength filter
        if wavelength_range and 'wavelength' in result_df.columns:
            mask = ((result_df['wavelength'] >= wavelength_range[0]) & 
                   (result_df['wavelength'] <= wavelength_range[1]))
            result_df = result_df[mask]
        
        print(f"Successfully read {len(result_df)} lines from custom format")
        return result_df

class LinelistOpacityCalculator:
    """Calculate line opacity using real astronomical linelists"""
    
    def __init__(self):
        self.reader = LinelistReader()
        self.calculator = LineOpacityCalculator()
        self.atomic_masses = self._load_atomic_masses()
        self.linelist_data = None
    
    def _load_atomic_masses(self) -> Dict[str, float]:
        """Load atomic masses for common elements"""
        return {
            'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811,
            'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
            'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
            'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
            'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
            'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.380
        }
    
    def load_linelist(self, filename: str, format_type: str = 'vald',
                     wavelength_range: Tuple[float, float] = None,
                     custom_format: Dict[str, int] = None) -> pd.DataFrame:
        """
        Load linelist from file
        
        Args:
            filename: Path to linelist file
            format_type: 'vald', 'kurucz', 'nist', or 'custom'
            wavelength_range: Optional (min_wave, max_wave) filter
            custom_format: For custom format, mapping of column names to indices
            
        Returns:
            DataFrame with loaded line data
        """
        if format_type == 'vald':
            self.linelist_data = self.reader.read_vald_linelist(filename, wavelength_range)
        elif format_type == 'kurucz':
            self.linelist_data = self.reader.read_kurucz_linelist(filename, wavelength_range)
        elif format_type == 'custom':
            if custom_format is None:
                raise ValueError("custom_format dict required for custom format")
            self.linelist_data = self.reader.read_custom_linelist(filename, custom_format, wavelength_range)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        print(f"Loaded {len(self.linelist_data)} lines")
        return self.linelist_data
    
    def convert_to_spectral_lines(self, selection_criteria: Dict = None,
                                max_lines: int = None) -> List[SpectralLine]:
        """
        Convert linelist DataFrame to SpectralLine objects
        
        Args:
            selection_criteria: Dict with selection parameters
            max_lines: Maximum number of lines to include
            
        Returns:
            List of SpectralLine objects
        """
        if self.linelist_data is None:
            raise ValueError("No linelist loaded. Call load_linelist() first.")
        
        df = self.linelist_data.copy()
        
        # Apply selection criteria
        if selection_criteria:
            if 'min_log_gf' in selection_criteria:
                df = df[df['log_gf'] >= selection_criteria['min_log_gf']]
            
            if 'species_filter' in selection_criteria:
                species_list = selection_criteria['species_filter']
                df = df[df['species'].isin(species_list)]
            
            if 'max_excitation' in selection_criteria:
                df = df[df['excitation_lower'] <= selection_criteria['max_excitation']]
        
        # Sort by line strength (log_gf) and take strongest lines
        df = df.sort_values('log_gf', ascending=False)
        if max_lines:
            df = df.head(max_lines)
        
        # Convert to SpectralLine objects
        spectral_lines = []
        for _, row in df.iterrows():
            try:
                # Parse species to get element and ionization
                species_parts = row['species'].split()
                element = species_parts[0]
                ion_stage = species_parts[1] if len(species_parts) > 1 else 'I'
                
                # Get atomic mass
                atomic_mass = self.atomic_masses.get(element, 55.845)  # Default to Fe
                
                # Create ion ID (simplified)
                ion_id = self._create_ion_id(element, ion_stage)
                
                # Convert log_gf to oscillator strength
                oscillator_strength = 10**row['log_gf']
                
                # Get broadening parameters
                gamma_rad = row.get('gamma_rad', 1e8)
                gamma_stark = row.get('gamma_stark', 1e-15)
                gamma_vdw = row.get('gamma_vdw', 1e-7)
                
                # Create SpectralLine object
                line = SpectralLine(
                    wavelength=row['wavelength'],
                    ion_id=ion_id,
                    oscillator_strength=oscillator_strength,
                    lower_energy=row['energy_lower'],
                    upper_energy=row['energy_upper'],
                    gamma_rad=gamma_rad,
                    gamma_stark=gamma_stark,
                    gamma_vdw=gamma_vdw,
                    atomic_mass=atomic_mass
                )
                spectral_lines.append(line)
                
            except Exception as e:
                print(f"Warning: Could not convert line {row['wavelength']:.2f}: {e}")
                continue
        
        print(f"Converted {len(spectral_lines)} lines to SpectralLine objects")
        return spectral_lines
    
    def _create_ion_id(self, element: str, ion_stage: str) -> int:
        """Create numerical ion ID from element and ionization stage"""
        # Simplified mapping: element_number * 100 + ionization
        element_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
            'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30
        }
        
        ion_numbers = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5}
        
        element_num = element_numbers.get(element, 26)  # Default to Fe
        ion_num = ion_numbers.get(ion_stage, 1)  # Default to neutral
        
        return element_num * 100 + ion_num
    
    def calculate_linelist_opacity(self, wavelength_grid: np.ndarray,
                                 atmosphere_params: Dict,
                                 abundance_dict: Dict[int, float],
                                 selection_criteria: Dict = None,
                                 max_lines: int = 10000) -> Tuple[np.ndarray, Dict]:
        """
        Calculate line opacity from loaded linelist
        
        Args:
            wavelength_grid: Wavelength array (Angstroms)
            atmosphere_params: Dictionary with atmospheric parameters
            abundance_dict: Element abundances by ion_id
            selection_criteria: Line selection criteria
            max_lines: Maximum number of lines to include
            
        Returns:
            Tuple of (opacity_array, statistics_dict)
        """
        # Convert linelist to SpectralLine objects
        spectral_lines = self.convert_to_spectral_lines(selection_criteria, max_lines)
        
        # Add lines to calculator
        self.calculator.lines = []  # Clear existing lines
        for line in spectral_lines:
            self.calculator.add_line(line)
        
        print(f"Calculating opacity for {len(spectral_lines)} lines...")
        
        # Calculate total line opacity
        opacity = self.calculator.compute_total_line_opacity(
            wavelength_grid=wavelength_grid,
            temperature=atmosphere_params['temperature'],
            electron_density=atmosphere_params['electron_density'],
            neutral_density=atmosphere_params['neutral_density'],
            mass_density=atmosphere_params['mass_density'],
            abundance_dict=abundance_dict,
            turbulent_velocity=atmosphere_params.get('turbulent_velocity', 2e5),
            opacity_threshold=atmosphere_params.get('opacity_threshold', 1e-12)
        )
        
        # Calculate statistics
        stats = {
            'total_lines_loaded': len(self.linelist_data),
            'lines_used': len(spectral_lines),
            'wavelength_range': (wavelength_grid.min(), wavelength_grid.max()),
            'max_opacity': np.max(opacity),
            'mean_opacity': np.mean(opacity[opacity > 0]),
            'integrated_opacity': np.trapz(opacity, wavelength_grid)
        }
        
        return opacity, stats

def create_sample_linelist_file():
    """Create a sample linelist file for demonstration"""
    sample_data = """# Sample VALD format linelist
# Species, Wavelength(A), Excit(eV), log(gf), Rad.damp, Stark.damp, vdW.damp, Lower.flag, Lower.energy, Upper.flag, Upper.energy
'Fe 1',      5250.2084,  0.000,  -0.004, 1.00e+08, 1.00e-15, 1.20e-07, 'E',      0.0, 'E',  19020.0
'Fe 1',      5269.5370,  0.859,  -1.321, 1.00e+08, 1.20e-15, 1.00e-07, 'E',    416.0, 'E',  19415.0
'Fe 1',      5283.6210,  3.241,  -0.432, 1.00e+08, 1.50e-15, 1.50e-07, 'E',   2616.0, 'E',  21530.0
'Ca 1',      4226.7280,  0.000,   0.244, 2.50e+08, 5.00e-16, 2.50e-07, 'E',      0.0, 'E',  23652.0
'Mg 1',      5183.6040,  2.717,  -0.239, 1.50e+08, 1.00e-16, 1.50e-07, 'E',   2195.0, 'E',  21480.0
'Mg 1',      5172.6840,  2.712,  -0.402, 1.50e+08, 1.00e-16, 1.50e-07, 'E',   2187.0, 'E',  21519.0
'Ti 1',      4981.7310,  0.848,   0.504, 2.00e+08, 2.00e-15, 2.00e-07, 'E',    684.0, 'E',  20753.0
'Cr 1',      5247.5650,  0.961,  -1.640, 1.80e+08, 1.80e-15, 1.80e-07, 'E',    775.0, 'E',  19810.0
'Ni 1',      5476.9040,  1.826,  -0.890, 1.70e+08, 1.60e-15, 1.60e-07, 'E',   1472.0, 'E',  19730.0
'Si 1',      5948.5410,  5.082,  -1.230, 1.30e+08, 8.00e-16, 1.30e-07, 'E',   4102.0, 'E',  20900.0
"""
    
    filename = "/Users/jdli/Desktop/AI4astro/RT/atlas12/sample_linelist.vald"
    with open(filename, 'w') as f:
        f.write(sample_data)
    
    print(f"Created sample linelist: {filename}")
    return filename

def main():
    """Demonstration of linelist-based opacity calculation"""
    
    print("=== Linelist-based Line Opacity Calculation ===\n")
    
    # Create sample linelist file
    linelist_file = create_sample_linelist_file()
    
    # Initialize calculator
    calc = LinelistOpacityCalculator()
    
    # Load linelist
    print("Loading linelist...")
    wavelength_range = (4000, 6000)  # Angstroms
    linelist_df = calc.load_linelist(
        linelist_file, 
        format_type='vald',
        wavelength_range=wavelength_range
    )
    
    # Display linelist information
    print(f"\nLinelist summary:")
    print(f"  Wavelength range: {linelist_df['wavelength'].min():.1f} - {linelist_df['wavelength'].max():.1f} Å")
    print(f"  Species: {sorted(linelist_df['species'].unique())}")
    print(f"  log(gf) range: {linelist_df['log_gf'].min():.2f} to {linelist_df['log_gf'].max():.2f}")
    
    # Selection criteria for lines
    selection_criteria = {
        'min_log_gf': -2.0,  # Only include lines with log(gf) > -2
        'species_filter': ['Fe 1', 'Ca 1', 'Mg 1', 'Ti 1'],  # Only these species
        'max_excitation': 5.0  # Maximum excitation energy (eV)
    }
    
    # Atmospheric parameters
    atmosphere_params = {
        'temperature': 5777.0,      # K
        'electron_density': 1.4e14, # cm⁻³
        'neutral_density': 2.8e17,  # cm⁻³
        'mass_density': 2.3e-7,     # g/cm³
        'turbulent_velocity': 1.2e5, # cm/s
        'opacity_threshold': 1e-12   # Minimum opacity to compute
    }
    
    # Element abundances (by ion_id: element_number*100 + ionization)
    abundance_dict = {
        1201: 3.8e-5,   # Mg I
        2001: 2.3e-6,   # Ca I  
        2201: 1.0e-7,   # Ti I
        2401: 5.0e-7,   # Cr I
        2601: 3.2e-5,   # Fe I
        2801: 1.8e-6,   # Ni I
        1401: 3.6e-5    # Si I
    }
    
    print(f"\nAtmospheric parameters:")
    for key, value in atmosphere_params.items():
        print(f"  {key}: {value}")
    
    # Create wavelength grid
    wave_min, wave_max = 5200, 5300  # Focus on Fe I region
    n_points = 2000
    wavelength_grid = np.linspace(wave_min, wave_max, n_points)
    
    print(f"\nCalculating opacity...")
    print(f"  Wavelength grid: {wave_min}-{wave_max} Å")
    print(f"  Resolution: {(wave_max-wave_min)/n_points:.3f} Å/pixel")
    
    # Calculate line opacity
    opacity, stats = calc.calculate_linelist_opacity(
        wavelength_grid=wavelength_grid,
        atmosphere_params=atmosphere_params,
        abundance_dict=abundance_dict,
        selection_criteria=selection_criteria,
        max_lines=5000
    )
    
    # Print results
    print(f"\nResults:")
    for key, value in stats.items():
        if isinstance(value, tuple):
            print(f"  {key}: {value[0]:.1f} - {value[1]:.1f}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value}")
    
    # Create plots
    print(f"\nCreating plots...")
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Full opacity spectrum
    plt.subplot(2, 3, 1)
    plt.plot(wavelength_grid, opacity, 'b-', linewidth=0.8)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Line Opacity (cm²/g)')
    plt.title('Line Opacity from Linelist')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Linelist statistics
    plt.subplot(2, 3, 2)
    species_counts = calc.linelist_data['species'].value_counts()
    plt.bar(range(len(species_counts)), species_counts.values)
    plt.xticks(range(len(species_counts)), species_counts.index, rotation=45)
    plt.ylabel('Number of Lines')
    plt.title('Lines per Species')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Line strength distribution
    plt.subplot(2, 3, 3)
    plt.hist(calc.linelist_data['log_gf'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('log(gf)')
    plt.ylabel('Number of Lines')
    plt.title('Line Strength Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Zoom on specific lines
    plt.subplot(2, 3, 4)
    zoom_mask = (wavelength_grid >= 5248) & (wavelength_grid <= 5252)
    plt.plot(wavelength_grid[zoom_mask], opacity[zoom_mask], 'r-', linewidth=2)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Line Opacity (cm²/g)')
    plt.title('Fe I 5250 Å Region')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Excitation vs wavelength
    plt.subplot(2, 3, 5)
    plt.scatter(calc.linelist_data['wavelength'], calc.linelist_data['excitation_lower'], 
                c=calc.linelist_data['log_gf'], cmap='viridis', alpha=0.6)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Excitation Energy (eV)')
    plt.title('Line Excitation vs Wavelength')
    plt.colorbar(label='log(gf)')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Cumulative opacity
    plt.subplot(2, 3, 6)
    cumulative_opacity = np.cumsum(opacity) * (wavelength_grid[1] - wavelength_grid[0])
    plt.plot(wavelength_grid, cumulative_opacity, 'g-', linewidth=2)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Cumulative Opacity (cm²/g·Å)')
    plt.title('Cumulative Line Opacity')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save results
    output_file = "/Users/jdli/Desktop/AI4astro/RT/atlas12/linelist_opacity_results.txt"
    with open(output_file, 'w') as f:
        f.write("# Line Opacity Results from Linelist\n")
        f.write("# Wavelength(A)  Opacity(cm²/g)\n")
        for wave, opa in zip(wavelength_grid, opacity):
            f.write(f"{wave:.4f}  {opa:.6e}\n")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Analysis complete!")

if __name__ == "__main__":
    main()