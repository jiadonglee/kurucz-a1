# ATLAS12 Default Linelist Guide

## Overview

ATLAS12 uses a sophisticated system of multiple linelist sources with intelligent temperature-dependent selection rather than a single "default" linelist. This guide explains the default approach and provides Python implementations.

## ATLAS12 Default Linelist Sources

### 1. Primary Atomic Line Databases

| Source | Content | Coverage | File Format |
|--------|---------|----------|-------------|
| **CD-ROM 1** | Main atomic lines | H, He, metals up to Z~30 | `LOWLINES.DAT`, `HIGHLINES.DAT` |
| **CD-ROM 15** | Diatomic molecules | CO, CN, CH, OH for G/K stars | `DIATOMICS.PCK` |
| **CD-ROM 24** | TiO molecular lines | Titanium oxide for M dwarfs | `TIOSCHWENKE.BIN` |
| **CD-ROM 26** | H2O molecular lines | Water vapor for cool stars | `H2OFAST.BIN` |

### 2. Special Treatment Lines

- **GFALLNLTE.DAT**: Lines requiring NLTE treatment (H, He, selected metals)
- **BNLTELINES.DAT**: Binary processed NLTE lines (created by RNLTEALL)

### 3. Temperature-Dependent Line Selection

ATLAS12 automatically selects different line sets based on stellar temperature:

```fortran
IF(TEFF.LT.30000.)NUSTART=NUHEII    ! He II lines for hot stars
IF(TEFF.LT.13000.)NUSTART=NUHEI     ! He I lines for B/A stars  
IF(TEFF.LT.7250.)NUSTART=NULYMAN    ! Lyman lines for F/G stars
IF(TEFF.LT.4500.)NUSTART=NUCI       ! Full metal lines for K/M stars
```

## Default Line Parameters

When specific broadening data is unavailable:

- **Radiative damping**: `γ_rad = 1.0×10⁸ s⁻¹`
- **Stark broadening**: `γ_stark = 1.0×10⁻¹⁵`
- **van der Waals**: `γ_vdw = 1.0×10⁻⁷`

## Element Coverage

### Major Contributors
- **H, He**: Complete coverage with special treatment
- **CNO group**: C, N, O fundamental lines
- **α-elements**: Mg, Si, S, Ca, Ti (important for abundance analysis)
- **Iron peak**: Cr, Mn, Fe, Co, Ni (dominant opacity sources)
- **Alkalis**: Na, K (strong resonance lines)

### Typical Line Counts
- **Available in databases**: ~10⁶ lines
- **Pre-selected by temperature**: ~10⁵ lines
- **Actually computed**: ~10⁴ lines above threshold
- **Dominant contributors**: ~10³ strongest lines

## Python Implementation

### 1. Default Linelist Creator

```python
import numpy as np
import pandas as pd
from pathlib import Path
from line_opacity_calculator import LineOpacityCalculator, SpectralLine

class ATLAS12DefaultLinelist:
    """Create ATLAS12-style default linelist based on stellar parameters"""
    
    def __init__(self):
        self.temperature_ranges = {
            'hot_star': (30000, np.inf),      # O/B stars
            'a_star': (13000, 30000),         # A stars  
            'fg_star': (7250, 13000),         # F/G stars
            'cool_star': (4500, 7250),        # K stars
            'm_dwarf': (2000, 4500)           # M dwarfs
        }
        
        # Default broadening parameters
        self.default_broadening = {
            'gamma_rad': 1.0e8,      # s⁻¹
            'gamma_stark': 1.0e-15,  # Stark parameter
            'gamma_vdw': 1.0e-7      # van der Waals parameter
        }
    
    def get_stellar_type(self, teff: float) -> str:
        """Determine stellar type from effective temperature"""
        for stellar_type, (tmin, tmax) in self.temperature_ranges.items():
            if tmin <= teff < tmax:
                return stellar_type
        return 'm_dwarf'  # Default for very cool
    
    def create_default_linelist(self, teff: float, 
                              wavelength_range: tuple = (3000, 10000)) -> pd.DataFrame:
        """
        Create ATLAS12-style default linelist for given stellar parameters
        
        Args:
            teff: Effective temperature (K)
            wavelength_range: Wavelength range in Angstroms
            
        Returns:
            DataFrame with default linelist
        """
        stellar_type = self.get_stellar_type(teff)
        
        # Base linelist with most important lines
        lines = []
        
        # Always include hydrogen lines
        lines.extend(self._get_hydrogen_lines(wavelength_range))
        
        # Add lines based on stellar type
        if stellar_type in ['hot_star', 'a_star']:
            lines.extend(self._get_hot_star_lines(wavelength_range))
        elif stellar_type == 'fg_star':
            lines.extend(self._get_solar_type_lines(wavelength_range))
        elif stellar_type == 'cool_star':
            lines.extend(self._get_cool_star_lines(wavelength_range))
        else:  # m_dwarf
            lines.extend(self._get_m_dwarf_lines(wavelength_range))
        
        # Convert to DataFrame
        df = pd.DataFrame(lines)
        
        # Apply wavelength filter
        mask = ((df['wavelength'] >= wavelength_range[0]) & 
                (df['wavelength'] <= wavelength_range[1]))
        df = df[mask].copy()
        
        # Sort by wavelength
        df = df.sort_values('wavelength').reset_index(drop=True)
        
        print(f"Created default linelist for {stellar_type} (T_eff = {teff} K)")
        print(f"Total lines: {len(df)}")
        print(f"Wavelength range: {df['wavelength'].min():.1f} - {df['wavelength'].max():.1f} Å")
        
        return df
    
    def _get_hydrogen_lines(self, wave_range: tuple) -> list:
        """Get fundamental hydrogen lines"""
        lines = []
        
        # Lyman series (UV)
        lyman_lines = [
            (1215.67, 2, 1, 0.4162),     # Lyman α
            (1025.72, 3, 1, 0.0791),     # Lyman β  
            (972.54, 4, 1, 0.0290),      # Lyman γ
            (949.74, 5, 1, 0.0139)       # Lyman δ
        ]
        
        # Balmer series (visible)
        balmer_lines = [
            (6562.79, 3, 2, 0.6407),     # H α
            (4861.35, 4, 2, 0.1193),     # H β
            (4340.47, 5, 2, 0.0447),     # H γ
            (4101.74, 6, 2, 0.0213),     # H δ
            (3970.07, 7, 2, 0.0118)      # H ε
        ]
        
        # Paschen series (near-IR)
        paschen_lines = [
            (18751.0, 4, 3, 0.8419),     # Pa α
            (12818.1, 5, 3, 0.1506),     # Pa β
            (10938.1, 6, 3, 0.0532)      # Pa γ
        ]
        
        all_h_lines = lyman_lines + balmer_lines + paschen_lines
        
        for wave, upper, lower, f_lu in all_h_lines:
            if wave_range[0] <= wave <= wave_range[1]:
                lines.append({
                    'species': 'H 1',
                    'wavelength': wave,
                    'excitation_lower': 13.6 * (1/lower**2 - 1/upper**2),  # eV
                    'log_gf': np.log10(f_lu),
                    'gamma_rad': 4.7e8,      # Higher for hydrogen
                    'gamma_stark': 1.0e-14,  # Strong Stark broadening
                    'gamma_vdw': 1.0e-8,
                    'flag_lower': 'E',
                    'energy_lower': 109678.8 * (1 - 1/lower**2),  # cm⁻¹
                    'flag_upper': 'E', 
                    'energy_upper': 109678.8 * (1 - 1/upper**2)
                })
        
        return lines
    
    def _get_solar_type_lines(self, wave_range: tuple) -> list:
        """Get lines appropriate for solar-type stars (F/G)"""
        lines = []
        
        # Strong Fe I lines (most important opacity contributors)
        fe_lines = [
            (5250.21, 0.12, -0.004, 'Fe 1'),   # Very strong Fe I
            (5269.54, 0.86, -1.321, 'Fe 1'),
            (5283.62, 3.24, -0.432, 'Fe 1'),
            (5324.18, 3.21, -0.103, 'Fe 1'),
            (5328.04, 0.92, -1.466, 'Fe 1'),
            (5371.49, 0.96, -1.645, 'Fe 1'),
            (5383.37, 4.31, 0.645, 'Fe 1'),
            (5397.13, 0.91, -1.993, 'Fe 1'),
            (5405.77, 0.99, -1.844, 'Fe 1'),
            (5429.70, 0.96, -1.881, 'Fe 1')
        ]
        
        # Ca I lines (important for abundance analysis)
        ca_lines = [
            (4226.73, 0.00, 0.244, 'Ca 1'),    # Very strong Ca I
            (5588.76, 2.53, 0.358, 'Ca 1'),
            (6102.72, 1.88, -0.793, 'Ca 1'),
            (6122.22, 1.89, -0.316, 'Ca 1'),
            (6162.17, 1.90, -0.090, 'Ca 1')
        ]
        
        # Mg I lines
        mg_lines = [
            (5172.68, 2.71, -0.402, 'Mg 1'),
            (5183.60, 2.72, -0.239, 'Mg 1'),
            (4571.10, 0.00, -5.623, 'Mg 1'),   # Mg b triplet
            (5528.41, 4.35, -0.620, 'Mg 1')
        ]
        
        # Si I lines  
        si_lines = [
            (5948.54, 5.08, -1.230, 'Si 1'),
            (6145.02, 5.62, -1.560, 'Si 1'),
            (6155.13, 5.62, -0.754, 'Si 1')
        ]
        
        # Ti I lines
        ti_lines = [
            (4981.73, 0.85, 0.504, 'Ti 1'),
            (5039.96, 0.02, -1.130, 'Ti 1'),
            (5173.75, 0.00, -1.118, 'Ti 1')
        ]
        
        all_metal_lines = fe_lines + ca_lines + mg_lines + si_lines + ti_lines
        
        for wave, excit_eV, log_gf, species in all_metal_lines:
            if wave_range[0] <= wave <= wave_range[1]:
                element = species.split()[0]
                atomic_mass = {'Fe': 55.845, 'Ca': 40.078, 'Mg': 24.305, 
                              'Si': 28.086, 'Ti': 47.867}.get(element, 55.845)
                
                lines.append({
                    'species': species,
                    'wavelength': wave,
                    'excitation_lower': excit_eV,
                    'log_gf': log_gf,
                    'gamma_rad': self.default_broadening['gamma_rad'],
                    'gamma_stark': self.default_broadening['gamma_stark'],
                    'gamma_vdw': self.default_broadening['gamma_vdw'],
                    'flag_lower': 'E',
                    'energy_lower': excit_eV * 8065.54,  # Convert eV to cm⁻¹
                    'flag_upper': 'E',
                    'energy_upper': (excit_eV + 1.24e-4 / wave * 1e8) * 8065.54
                })
        
        return lines
    
    def _get_hot_star_lines(self, wave_range: tuple) -> list:
        """Lines for hot stars (O/B/A type)"""
        lines = []
        
        # He I lines
        he_lines = [
            (5875.62, 20.96, 0.318, 'He 1'),   # He I D3
            (4471.48, 20.96, -0.020, 'He 1'),  # He I 4471
            (4026.19, 20.96, -0.400, 'He 1'),
            (4713.14, 20.96, -0.710, 'He 1')
        ]
        
        # He II lines (for very hot stars)
        he2_lines = [
            (4685.68, 40.81, 0.200, 'He 2'),   # He II 4686
            (5411.52, 40.81, -0.300, 'He 2')
        ]
        
        # Strong metal lines for A stars
        metal_lines = [
            (4226.73, 0.00, 0.244, 'Ca 1'),    # Ca I
            (3933.66, 0.00, 0.108, 'Ca 2'),    # Ca II K
            (3968.47, 0.00, -0.181, 'Ca 2'),   # Ca II H
            (5889.95, 0.00, 0.117, 'Na 1'),    # Na I D1
            (5895.92, 0.00, -0.184, 'Na 1')    # Na I D2
        ]
        
        all_hot_lines = he_lines + he2_lines + metal_lines
        
        for wave, excit_eV, log_gf, species in all_hot_lines:
            if wave_range[0] <= wave <= wave_range[1]:
                lines.append({
                    'species': species,
                    'wavelength': wave,
                    'excitation_lower': excit_eV,
                    'log_gf': log_gf,
                    'gamma_rad': self.default_broadening['gamma_rad'],
                    'gamma_stark': self.default_broadening['gamma_stark'],
                    'gamma_vdw': self.default_broadening['gamma_vdw'],
                    'flag_lower': 'E',
                    'energy_lower': excit_eV * 8065.54,
                    'flag_upper': 'E',
                    'energy_upper': (excit_eV + 1.24e-4 / wave * 1e8) * 8065.54
                })
        
        return lines
    
    def _get_cool_star_lines(self, wave_range: tuple) -> list:
        """Lines for cool stars (K type)"""
        # Similar to solar but with more molecular features
        lines = self._get_solar_type_lines(wave_range)
        
        # Add molecular lines (simplified)
        molecular_lines = [
            (4300.0, 0.0, -2.0, 'CH 1'),    # CH G-band
            (3883.0, 0.0, -1.5, 'CN 1'),    # CN violet
            (4215.0, 0.0, -2.5, 'CO 1')     # CO bandhead
        ]
        
        for wave, excit_eV, log_gf, species in molecular_lines:
            if wave_range[0] <= wave <= wave_range[1]:
                lines.append({
                    'species': species,
                    'wavelength': wave,
                    'excitation_lower': excit_eV,
                    'log_gf': log_gf,
                    'gamma_rad': 5.0e7,  # Lower for molecules
                    'gamma_stark': 1.0e-16,
                    'gamma_vdw': 1.0e-6,  # Higher for molecules
                    'flag_lower': 'E',
                    'energy_lower': excit_eV * 8065.54,
                    'flag_upper': 'E',
                    'energy_upper': (excit_eV + 1.24e-4 / wave * 1e8) * 8065.54
                })
        
        return lines
    
    def _get_m_dwarf_lines(self, wave_range: tuple) -> list:
        """Lines for M dwarf stars"""
        lines = self._get_cool_star_lines(wave_range)
        
        # Add TiO bands (dominant in M dwarfs)
        tio_lines = [
            (5167.0, 0.0, -1.0, 'TiO 1'),    # TiO γ system
            (6159.0, 0.0, -1.2, 'TiO 1'),    # TiO α system
            (7054.0, 0.0, -0.8, 'TiO 1'),    # TiO β system
            (8432.0, 0.0, -1.5, 'TiO 1')     # TiO δ system
        ]
        
        for wave, excit_eV, log_gf, species in tio_lines:
            if wave_range[0] <= wave <= wave_range[1]:
                lines.append({
                    'species': species,
                    'wavelength': wave,
                    'excitation_lower': excit_eV,
                    'log_gf': log_gf,
                    'gamma_rad': 3.0e7,
                    'gamma_stark': 1.0e-17,
                    'gamma_vdw': 1.0e-5,
                    'flag_lower': 'E',
                    'energy_lower': excit_eV * 8065.54,
                    'flag_upper': 'E',
                    'energy_upper': (excit_eV + 1.24e-4 / wave * 1e8) * 8065.54
                })
        
        return lines

def get_atlas12_abundances(metallicity: float = 0.0, 
                          alpha_enhancement: float = 0.0) -> dict:
    """
    Get ATLAS12-style element abundances
    
    Args:
        metallicity: [Fe/H] metallicity
        alpha_enhancement: [α/Fe] enhancement
        
    Returns:
        Dictionary with ion_id: abundance pairs
    """
    # Solar abundances (Asplund et al. 2009) - log10(N/N_H) + 12
    solar_log_abundances = {
        'H': 12.00, 'He': 10.93, 'C': 8.43, 'N': 7.83, 'O': 8.69,
        'Na': 6.24, 'Mg': 7.60, 'Al': 6.45, 'Si': 7.51, 'Ca': 6.34,
        'Ti': 4.95, 'Cr': 5.64, 'Mn': 5.43, 'Fe': 7.50, 'Ni': 6.22
    }
    
    # Alpha elements
    alpha_elements = ['O', 'Mg', 'Si', 'Ca', 'Ti']
    
    # Ion ID mapping (element_number * 100 + ionization_stage)
    ion_mapping = {
        'H': 101, 'He': 201, 'C': 601, 'N': 701, 'O': 801,
        'Na': 1101, 'Mg': 1201, 'Al': 1301, 'Si': 1401, 'Ca': 2001,
        'Ti': 2201, 'Cr': 2401, 'Mn': 2501, 'Fe': 2601, 'Ni': 2801
    }
    
    abundances = {}
    
    for element, solar_abund in solar_log_abundances.items():
        if element == 'H':
            continue  # Hydrogen is reference
            
        # Apply metallicity scaling
        abund = solar_abund + metallicity
        
        # Apply alpha enhancement
        if element in alpha_elements:
            abund += alpha_enhancement
        
        # Convert to number fraction relative to hydrogen  
        number_fraction = 10**(abund - 12.0)
        
        # Map to ion ID (neutral species)
        if element in ion_mapping:
            ion_id = ion_mapping[element]
            abundances[ion_id] = number_fraction
    
    return abundances

def example_usage():
    """Example of using ATLAS12 default linelist"""
    from linelist_opacity_calculator import LinelistOpacityCalculator
    
    # Create default linelist generator
    atlas12_default = ATLAS12DefaultLinelist()
    
    # Stellar parameters
    teff = 5777.0  # Solar temperature
    wavelength_range = (5200, 5300)  # Focus on Fe I region
    
    # Generate default linelist
    print("=== ATLAS12 Default Linelist Example ===\n")
    default_linelist = atlas12_default.create_default_linelist(
        teff=teff,
        wavelength_range=wavelength_range
    )
    
    # Display linelist info
    print(f"\nLinelist composition:")
    species_counts = default_linelist['species'].value_counts()
    for species, count in species_counts.items():
        print(f"  {species}: {count} lines")
    
    # Convert to opacity calculator format
    calc = LinelistOpacityCalculator()
    calc.linelist_data = default_linelist
    
    # Get ATLAS12-style abundances
    abundances = get_atlas12_abundances(metallicity=0.0, alpha_enhancement=0.0)
    print(f"\nUsing abundances for {len(abundances)} elements")
    
    # Atmospheric parameters (solar photosphere)
    atmosphere_params = {
        'temperature': teff,
        'electron_density': 1.4e14,
        'neutral_density': 2.8e17,
        'mass_density': 2.3e-7,
        'turbulent_velocity': 1.2e5
    }
    
    # Calculate opacity
    wavelengths = np.linspace(*wavelength_range, 1000)
    opacity, stats = calc.calculate_linelist_opacity(
        wavelength_grid=wavelengths,
        atmosphere_params=atmosphere_params,
        abundance_dict=abundances
    )
    
    print(f"\nOpacity calculation results:")
    print(f"  Max opacity: {stats['max_opacity']:.2e} cm²/g")
    print(f"  Lines computed: {stats['lines_used']}")
    
    # Plot results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(wavelengths, opacity, 'b-', linewidth=1)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Line Opacity (cm²/g)')
    plt.title('ATLAS12 Default Linelist Opacity')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.hist(default_linelist['log_gf'], bins=15, alpha=0.7, edgecolor='black')
    plt.xlabel('log(gf)')
    plt.ylabel('Number of Lines')
    plt.title('Line Strength Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return default_linelist, opacity, stats

if __name__ == "__main__":
    example_usage()