#!/usr/bin/env python3
"""
ATLAS12 Default Linelist Demonstration

This script shows how to create and use ATLAS12-style default linelists
for different stellar types, implementing the temperature-dependent line
selection approach used in the original ATLAS12 code.
"""

import numpy as np
import matplotlib.pyplot as plt
from line_opacity_calculator import LineOpacityCalculator, SpectralLine

def create_atlas12_default_lines(stellar_type='solar'):
    """
    Create ATLAS12-style default linelist for different stellar types
    
    Args:
        stellar_type: 'hot', 'solar', 'cool', or 'm_dwarf'
        
    Returns:
        List of SpectralLine objects
    """
    lines = []
    
    # Always include key hydrogen lines
    hydrogen_lines = [
        # Balmer series (visible)
        SpectralLine(6562.79, 101, 0.6407, 82259.0, 97492.0, 4.7e8, 1.0e-14, 1.0e-8, 1.008),  # H α
        SpectralLine(4861.35, 101, 0.1193, 82259.0, 102824.0, 4.7e8, 1.0e-14, 1.0e-8, 1.008), # H β
        SpectralLine(4340.47, 101, 0.0447, 82259.0, 105292.0, 4.7e8, 1.0e-14, 1.0e-8, 1.008), # H γ
        SpectralLine(4101.74, 101, 0.0213, 82259.0, 106632.0, 4.7e8, 1.0e-14, 1.0e-8, 1.008)  # H δ
    ]
    
    lines.extend(hydrogen_lines)
    
    if stellar_type in ['hot']:
        # Hot star lines (O/B/A stars)
        hot_lines = [
            # He I lines
            SpectralLine(5875.62, 201, 0.318, 166272.0, 183237.0, 1.8e8, 1.0e-15, 1.0e-7, 4.003),  # He I D3
            SpectralLine(4471.48, 201, 0.020, 166272.0, 188665.0, 1.8e8, 1.0e-15, 1.0e-7, 4.003),  # He I 4471
            
            # Strong Ca II lines
            SpectralLine(3933.66, 2002, 0.108, 0.0, 25414.0, 1.4e8, 1.0e-15, 1.0e-7, 40.078),     # Ca II K
            SpectralLine(3968.47, 2002, 0.681, 0.0, 25192.0, 1.4e8, 1.0e-15, 1.0e-7, 40.078),     # Ca II H
            
            # Na I D lines
            SpectralLine(5889.95, 1101, 0.117, 0.0, 16973.0, 6.3e7, 1.0e-15, 1.0e-7, 22.990),     # Na I D1
            SpectralLine(5895.92, 1101, 0.684, 0.0, 16956.0, 6.3e7, 1.0e-15, 1.0e-7, 22.990)      # Na I D2
        ]
        lines.extend(hot_lines)
    
    elif stellar_type in ['solar', 'cool']:
        # Solar-type and cool star lines (F/G/K stars)
        metal_lines = [
            # Strong Fe I lines (opacity dominators)
            SpectralLine(5250.21, 2601, 0.001, 0.0, 19020.0, 1.2e8, 1.5e-15, 1.2e-7, 55.845),    # Fe I 5250
            SpectralLine(5269.54, 2601, 0.0005, 415.9, 19415.0, 1.0e8, 1.2e-15, 1.0e-7, 55.845), # Fe I 5269
            SpectralLine(5324.18, 2601, 0.0008, 3037.0, 21222.0, 1.1e8, 1.3e-15, 1.1e-7, 55.845), # Fe I 5324
            SpectralLine(5328.04, 2601, 0.0003, 415.9, 19201.0, 1.0e8, 1.2e-15, 1.0e-7, 55.845),  # Fe I 5328
            
            # Ca I lines (abundance indicators)
            SpectralLine(4226.73, 2001, 2.0, 0.0, 23652.0, 2.5e8, 5.0e-16, 2.5e-7, 40.078),       # Ca I 4227
            SpectralLine(5588.76, 2001, 0.358, 2523.0, 20335.0, 1.8e8, 4.0e-16, 2.0e-7, 40.078),  # Ca I 5589
            
            # Mg I lines  
            SpectralLine(5172.68, 1201, 0.18, 2.72, 19285.0, 1.5e8, 1.0e-16, 1.5e-7, 24.305),     # Mg I 5173
            SpectralLine(5183.60, 1201, 0.18, 2.72, 19285.0, 1.5e8, 1.0e-16, 1.5e-7, 24.305),     # Mg I 5184
            
            # Si I lines
            SpectralLine(5948.54, 1401, 0.059, 5.082, 21812.0, 1.3e8, 8.0e-16, 1.3e-7, 28.086),   # Si I 5949
            
            # Ti I lines (alpha element)
            SpectralLine(4981.73, 2201, 0.504, 0.848, 20753.0, 2.0e8, 2.0e-15, 2.0e-7, 47.867),   # Ti I 4982
            SpectralLine(5039.96, 2201, 0.130, 0.021, 19842.0, 1.9e8, 1.9e-15, 1.9e-7, 47.867)    # Ti I 5040
        ]
        lines.extend(metal_lines)
        
        if stellar_type == 'cool':
            # Additional lines for K stars
            cool_lines = [
                # More Fe I lines (become more important in cooler stars)
                SpectralLine(5371.49, 2601, 0.023, 0.958, 20874.0, 1.1e8, 1.4e-15, 1.1e-7, 55.845), # Fe I 5371
                SpectralLine(5383.37, 2601, 0.645, 4.312, 22838.0, 1.6e8, 1.8e-15, 1.4e-7, 55.845), # Fe I 5383
                
                # Cr I lines
                SpectralLine(5247.56, 2401, 0.023, 0.961, 19810.0, 1.8e8, 1.8e-15, 1.8e-7, 51.996), # Cr I 5248
                
                # Ni I lines  
                SpectralLine(5476.90, 2801, 0.129, 1.826, 19730.0, 1.7e8, 1.6e-15, 1.6e-7, 58.693)  # Ni I 5477
            ]
            lines.extend(cool_lines)
    
    elif stellar_type == 'm_dwarf':
        # M dwarf stars - include molecular features
        molecular_lines = [
            # Simplified TiO features (in reality these are complex band systems)
            SpectralLine(5167.0, 9901, 0.1, 0.0, 19354.0, 3.0e7, 1.0e-17, 1.0e-5, 63.866),       # TiO γ system
            SpectralLine(6159.0, 9901, 0.063, 0.0, 16237.0, 3.0e7, 1.0e-17, 1.0e-5, 63.866),     # TiO α system
            SpectralLine(7054.0, 9901, 0.158, 0.0, 14178.0, 3.0e7, 1.0e-17, 1.0e-5, 63.866),     # TiO β system
            
            # Some remaining atomic lines (weaker but still present)
            SpectralLine(5250.21, 2601, 0.001, 0.0, 19020.0, 1.2e8, 1.5e-15, 1.2e-7, 55.845),    # Fe I 5250
            SpectralLine(4226.73, 2001, 2.0, 0.0, 23652.0, 2.5e8, 5.0e-16, 2.5e-7, 40.078)       # Ca I 4227
        ]
        lines.extend(molecular_lines)
    
    print(f"Created {len(lines)} default lines for {stellar_type} star")
    return lines

def get_default_abundances(metallicity=0.0, alpha_enhancement=0.0):
    """
    Get ATLAS12-style default abundances
    
    Args:
        metallicity: [Fe/H] value
        alpha_enhancement: [α/Fe] value
        
    Returns:
        Dictionary of ion_id: abundance pairs
    """
    # Solar abundances (log scale, H = 12)
    solar_log = {
        'Mg': 7.60, 'Al': 6.45, 'Si': 7.51, 'Ca': 6.34, 'Ti': 4.95,
        'Cr': 5.64, 'Mn': 5.43, 'Fe': 7.50, 'Ni': 6.22, 'Na': 6.24
    }
    
    alpha_elements = ['Mg', 'Si', 'Ca', 'Ti']
    
    # Ion ID mapping (element_number * 100 + ionization_stage)
    ion_ids = {
        'Na': 1101, 'Mg': 1201, 'Al': 1301, 'Si': 1401, 'Ca': 2001,
        'Ti': 2201, 'Cr': 2401, 'Mn': 2501, 'Fe': 2601, 'Ni': 2801,
        'Ca2': 2002  # Ca II
    }
    
    abundances = {}
    
    for element, solar_abund in solar_log.items():
        # Apply metallicity scaling
        abund = solar_abund + metallicity
        
        # Apply alpha enhancement for alpha elements
        if element in alpha_elements:
            abund += alpha_enhancement
        
        # Convert to number fraction (relative to total number density)
        number_fraction = 10**(abund - 12.0)
        
        # Neutral species
        if element in ion_ids:
            abundances[ion_ids[element]] = number_fraction
        
        # Add some ionized species for hotter stars
        if element == 'Ca':
            abundances[ion_ids['Ca2']] = number_fraction * 0.1  # 10% ionized
    
    # Add some special cases
    abundances[101] = 1e-4    # H (relative to neutrals)
    abundances[201] = 1e-5    # He I
    abundances[9901] = 1e-6   # TiO (for M dwarfs)
    
    return abundances

def demonstrate_stellar_types():
    """Demonstrate opacity calculation for different stellar types"""
    
    print("=== ATLAS12 Default Linelist Demonstration ===\n")
    
    # Stellar parameters for different types
    stellar_params = {
        'hot': {'teff': 15000, 'type': 'A star'},
        'solar': {'teff': 5777, 'type': 'G star (Sun)'},
        'cool': {'teff': 4500, 'type': 'K star'},
        'm_dwarf': {'teff': 3500, 'type': 'M dwarf'}
    }
    
    # Create calculator
    calc = LineOpacityCalculator()
    
    # Wavelength grid (focus on visible)
    wavelengths = np.linspace(5200, 5300, 1000)  # Fe I region
    
    results = {}
    
    for stellar_type, params in stellar_params.items():
        print(f"Processing {params['type']} (T_eff = {params['teff']} K)...")
        
        # Create default linelist for this stellar type
        default_lines = create_atlas12_default_lines(stellar_type)
        
        # Add lines to calculator
        calc.lines = default_lines
        
        # Get appropriate abundances
        if stellar_type == 'hot':
            metallicity = 0.0  # Solar metallicity
            alpha_enh = 0.0
        elif stellar_type == 'solar':
            metallicity = 0.0  # Solar
            alpha_enh = 0.0
        elif stellar_type == 'cool':
            metallicity = -0.3  # Slightly metal-poor
            alpha_enh = 0.2     # Alpha-enhanced
        else:  # m_dwarf
            metallicity = -0.5  # Metal-poor
            alpha_enh = 0.3     # Alpha-enhanced
        
        abundances = get_default_abundances(metallicity, alpha_enh)
        
        # Atmospheric parameters (scale with temperature)
        temperature = params['teff']
        atmosphere_params = {
            'temperature': temperature,
            'electron_density': 1.4e14 * (temperature/5777)**0.5,  # Scale with sqrt(T)
            'neutral_density': 2.8e17,
            'mass_density': 2.3e-7,
            'turbulent_velocity': 1.2e5
        }
        
        # Calculate opacity
        opacity = calc.compute_total_line_opacity(
            wavelength_grid=wavelengths,
            temperature=atmosphere_params['temperature'],
            electron_density=atmosphere_params['electron_density'],
            neutral_density=atmosphere_params['neutral_density'],
            mass_density=atmosphere_params['mass_density'],
            abundance_dict=abundances,
            turbulent_velocity=atmosphere_params['turbulent_velocity']
        )
        
        results[stellar_type] = {
            'wavelengths': wavelengths,
            'opacity': opacity,
            'params': params,
            'max_opacity': np.max(opacity),
            'n_lines': len(default_lines)
        }
        
        print(f"  Lines: {len(default_lines)}")
        print(f"  Max opacity: {np.max(opacity):.2e} cm²/g")
        print(f"  [Fe/H] = {metallicity:.1f}, [α/Fe] = {alpha_enh:.1f}\n")
    
    # Create comparison plots
    print("Creating comparison plots...")
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Opacity comparison
    plt.subplot(2, 3, 1)
    colors = {'hot': 'blue', 'solar': 'orange', 'cool': 'red', 'm_dwarf': 'purple'}
    
    for stellar_type, data in results.items():
        plt.plot(data['wavelengths'], data['opacity'], 
                color=colors[stellar_type], 
                label=f"{data['params']['type']}", linewidth=2)
    
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Line Opacity (cm²/g)')
    plt.title('Line Opacity vs Stellar Type')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Maximum opacity vs temperature
    plt.subplot(2, 3, 2)
    temps = [data['params']['teff'] for data in results.values()]
    max_opas = [data['max_opacity'] for data in results.values()]
    labels = [data['params']['type'] for data in results.values()]
    
    plt.scatter(temps, max_opas, s=100, c=[colors[st] for st in results.keys()])
    for i, label in enumerate(labels):
        plt.annotate(label, (temps[i], max_opas[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('T_eff (K)')
    plt.ylabel('Max Line Opacity (cm²/g)')
    plt.title('Peak Opacity vs Temperature')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Number of lines vs stellar type
    plt.subplot(2, 3, 3)
    n_lines = [data['n_lines'] for data in results.values()]
    stellar_labels = [data['params']['type'] for data in results.values()]
    
    bars = plt.bar(stellar_labels, n_lines, color=[colors[st] for st in results.keys()])
    plt.ylabel('Number of Lines')
    plt.title('Default Lines per Stellar Type')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, n in zip(bars, n_lines):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(n), ha='center', va='bottom')
    
    # Plot 4-6: Individual stellar type details
    plot_positions = [(2, 3, 4), (2, 3, 5), (2, 3, 6)]
    stellar_types_detail = ['solar', 'cool', 'm_dwarf']
    
    for i, stellar_type in enumerate(stellar_types_detail):
        plt.subplot(*plot_positions[i])
        data = results[stellar_type]
        
        plt.plot(data['wavelengths'], data['opacity'], 
                color=colors[stellar_type], linewidth=2)
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Line Opacity (cm²/g)')
        plt.title(f"{data['params']['type']} Detail")
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary
    print("Summary of ATLAS12 default linelist approach:")
    print("=" * 50)
    for stellar_type, data in results.items():
        print(f"{data['params']['type']:12} | {data['params']['teff']:5.0f} K | "
              f"{data['n_lines']:2d} lines | Max opacity: {data['max_opacity']:.1e}")
    
    return results

if __name__ == "__main__":
    results = demonstrate_stellar_types()