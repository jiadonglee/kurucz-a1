# Using Real Linelists for Line Opacity Calculation

## Overview

This guide shows how to use real astronomical linelists (VALD, Kurucz, NIST) with the ATLAS12-style line opacity calculator. It covers different linelist formats, selection criteria, and practical examples.

## Supported Linelist Formats

### 1. VALD (Vienna Atomic Line Database)

**Format Example:**
```
'Fe 1',      5250.2084,  0.000,  -0.004, 1.00e+08, 1.00e-15, 1.20e-07, 'E',      0.0, 'E',  19020.0
'Ca 1',      4226.7280,  0.000,   0.244, 2.50e+08, 5.00e-16, 2.50e-07, 'E',      0.0, 'E',  23652.0
```

**Columns:**
- Species (element + ionization)
- Wavelength (Å, vacuum)
- Lower level excitation energy (eV)
- log(gf) (oscillator strength)
- Radiative damping constant
- Stark broadening parameter
- van der Waals broadening parameter
- Lower/upper level flags and energies

### 2. Kurucz Format

**Fixed-width format:**
```
 5250.208 -0.004  26.0     0.000     0.0    1.00e+08
 4226.728  0.244  20.0     0.000     0.0    2.50e+08
```

**Columns (fixed positions):**
- 1-11: Wavelength (Å)
- 12-17: log(gf)
- 18-25: Element code (element.ionization)
- 26-37: Excitation potential (eV)

### 3. Custom Format

User-defined column mapping for any ASCII format.

## Basic Usage

### 1. Simple Linelist Loading

```python
from linelist_opacity_calculator import LinelistOpacityCalculator

# Initialize calculator
calc = LinelistOpacityCalculator()

# Load VALD format linelist
linelist_df = calc.load_linelist(
    filename='my_linelist.vald',
    format_type='vald',
    wavelength_range=(5000, 6000)  # Optional filter
)

print(f"Loaded {len(linelist_df)} lines")
```

### 2. Line Selection Criteria

```python
# Define selection criteria
selection_criteria = {
    'min_log_gf': -1.5,  # Only strong lines
    'species_filter': ['Fe 1', 'Ca 1', 'Mg 1'],  # Specific elements
    'max_excitation': 4.0  # Low excitation lines only
}

# Convert to SpectralLine objects
spectral_lines = calc.convert_to_spectral_lines(
    selection_criteria=selection_criteria,
    max_lines=1000  # Limit number of lines
)
```

### 3. Opacity Calculation

```python
# Atmospheric parameters
atmosphere_params = {
    'temperature': 5777.0,      # K
    'electron_density': 1.4e14, # cm⁻³
    'neutral_density': 2.8e17,  # cm⁻³
    'mass_density': 2.3e-7,     # g/cm³
    'turbulent_velocity': 1.2e5 # cm/s
}

# Element abundances (ion_id: abundance)
abundances = {
    2601: 3.2e-5,   # Fe I
    2001: 2.3e-6,   # Ca I
    1201: 3.8e-5    # Mg I
}

# Wavelength grid
wavelengths = np.linspace(5200, 5300, 2000)

# Calculate opacity
opacity, stats = calc.calculate_linelist_opacity(
    wavelength_grid=wavelengths,
    atmosphere_params=atmosphere_params,
    abundance_dict=abundances,
    selection_criteria=selection_criteria
)
```

## Advanced Examples

### 1. Working with VALD Linelists

```python
def process_vald_linelist(vald_file, output_wavelength_range):
    """Process a VALD linelist for specific wavelength range"""
    
    calc = LinelistOpacityCalculator()
    
    # Load with wavelength filtering
    linelist_df = calc.load_linelist(
        filename=vald_file,
        format_type='vald',
        wavelength_range=output_wavelength_range
    )
    
    # Analyze line distribution
    print("Species distribution:")
    species_counts = linelist_df['species'].value_counts()
    for species, count in species_counts.head(10).items():
        print(f"  {species}: {count} lines")
    
    print(f"\nlog(gf) range: {linelist_df['log_gf'].min():.2f} to {linelist_df['log_gf'].max():.2f}")
    print(f"Excitation range: {linelist_df['excitation_lower'].min():.2f} to {linelist_df['excitation_lower'].max():.2f} eV")
    
    return linelist_df

# Example usage
vald_data = process_vald_linelist('extract_all.vald', (5000, 5500))
```

### 2. Custom Format Linelist

```python
def load_custom_linelist(filename):
    """Load custom format linelist"""
    
    # Define column mapping
    custom_format = {
        'wavelength': 0,        # Column 0 = wavelength
        'species': 1,           # Column 1 = species
        'log_gf': 2,           # Column 2 = log(gf)
        'excitation_lower': 3,  # Column 3 = excitation
        'gamma_rad': 4,        # Column 4 = radiative damping
        'gamma_stark': 5,      # Column 5 = Stark damping
        'gamma_vdw': 6,        # Column 6 = vdW damping
        'energy_lower': 7,     # Column 7 = lower energy (cm⁻¹)
        'energy_upper': 8      # Column 8 = upper energy (cm⁻¹)
    }
    
    calc = LinelistOpacityCalculator()
    
    linelist_df = calc.load_linelist(
        filename=filename,
        format_type='custom',
        custom_format=custom_format,
        wavelength_range=(4000, 7000)
    )
    
    return linelist_df
```

### 3. Multi-Element Abundance Patterns

```python
def calculate_stellar_abundances(metallicity=0.0, alpha_enhancement=0.0):
    """Calculate stellar abundance pattern"""
    
    # Solar abundances (Asplund et al. 2009)
    solar_abundances = {
        'H': 12.0, 'He': 10.93, 'Li': 1.05, 'C': 8.43, 'N': 7.83, 'O': 8.69,
        'Ne': 7.93, 'Na': 6.24, 'Mg': 7.60, 'Al': 6.45, 'Si': 7.51, 'P': 5.41,
        'S': 7.12, 'Ar': 6.40, 'K': 5.03, 'Ca': 6.34, 'Sc': 3.15, 'Ti': 4.95,
        'V': 3.93, 'Cr': 5.64, 'Mn': 5.43, 'Fe': 7.50, 'Co': 4.99, 'Ni': 6.22
    }
    
    # Alpha elements
    alpha_elements = ['O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Ti']
    
    abundances = {}
    for element, solar_abund in solar_abundances.items():
        # Apply metallicity scaling
        abund = solar_abund + metallicity
        
        # Apply alpha enhancement
        if element in alpha_elements:
            abund += alpha_enhancement
        
        # Convert to number fraction (relative to H)
        number_fraction = 10**(abund - 12.0)
        
        # Convert to ion IDs for neutral species
        element_numbers = {
            'Mg': 12, 'Al': 13, 'Si': 14, 'Ca': 20, 'Ti': 22,
            'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28
        }
        
        if element in element_numbers:
            ion_id = element_numbers[element] * 100 + 1  # Neutral
            abundances[ion_id] = number_fraction
    
    return abundances

# Example: Metal-poor star with alpha enhancement
abundances_metal_poor = calculate_stellar_abundances(
    metallicity=-1.0,  # [Fe/H] = -1.0
    alpha_enhancement=0.4  # [α/Fe] = +0.4
)
```

### 4. Opacity Comparison Between Different Abundance Patterns

```python
def compare_abundance_patterns(wavelength_range, atmosphere_params):
    """Compare opacity for different abundance patterns"""
    
    # Load linelist
    calc = LinelistOpacityCalculator()
    calc.load_linelist('sample_linelist.vald', 'vald', wavelength_range)
    
    # Different abundance patterns
    patterns = {
        'Solar': calculate_stellar_abundances(0.0, 0.0),
        'Metal-poor': calculate_stellar_abundances(-1.0, 0.0),
        'Alpha-enhanced': calculate_stellar_abundances(-1.0, 0.4)
    }
    
    wavelengths = np.linspace(*wavelength_range, 1000)
    results = {}
    
    for pattern_name, abundances in patterns.items():
        opacity, stats = calc.calculate_linelist_opacity(
            wavelength_grid=wavelengths,
            atmosphere_params=atmosphere_params,
            abundance_dict=abundances,
            max_lines=5000
        )
        results[pattern_name] = opacity
        print(f"{pattern_name}: max opacity = {stats['max_opacity']:.2e}")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    for pattern_name, opacity in results.items():
        plt.plot(wavelengths, opacity, label=pattern_name, linewidth=1.5)
    
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Line Opacity (cm²/g)')
    plt.title('Line Opacity vs Abundance Pattern')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return results
```

## Practical Workflows

### 1. High-Resolution Spectrum Synthesis

```python
def synthesize_high_resolution_spectrum():
    """Synthesize high-resolution spectrum using linelist"""
    
    # Load comprehensive linelist
    calc = LinelistOpacityCalculator()
    calc.load_linelist(
        'comprehensive_linelist.vald',
        format_type='vald',
        wavelength_range=(5000, 5100)  # 100 Å region
    )
    
    # High-resolution wavelength grid
    R = 100000  # Resolving power
    central_wave = 5050.0
    delta_wave = central_wave / R
    n_points = int(100 / delta_wave)
    wavelengths = np.linspace(5000, 5100, n_points)
    
    print(f"Resolution: {R}")
    print(f"Wavelength sampling: {delta_wave:.4f} Å")
    print(f"Number of points: {n_points}")
    
    # Solar atmosphere parameters
    atmosphere_params = {
        'temperature': 5777.0,
        'electron_density': 1.4e14,
        'neutral_density': 2.8e17,
        'mass_density': 2.3e-7,
        'turbulent_velocity': 1.0e5  # 1 km/s
    }
    
    # Solar abundances
    abundances = calculate_stellar_abundances(0.0, 0.0)
    
    # Use only strong lines for efficiency
    selection_criteria = {
        'min_log_gf': -2.0,
        'max_excitation': 5.0
    }
    
    # Calculate opacity
    opacity, stats = calc.calculate_linelist_opacity(
        wavelength_grid=wavelengths,
        atmosphere_params=atmosphere_params,
        abundance_dict=abundances,
        selection_criteria=selection_criteria,
        max_lines=20000
    )
    
    return wavelengths, opacity, stats
```

### 2. Temperature Sensitivity Analysis

```python
def analyze_temperature_sensitivity(line_species='Fe 1'):
    """Analyze how line opacity changes with temperature"""
    
    calc = LinelistOpacityCalculator()
    calc.load_linelist('sample_linelist.vald', 'vald')
    
    # Filter for specific species
    selection_criteria = {
        'species_filter': [line_species],
        'min_log_gf': -2.0
    }
    
    # Temperature range
    temperatures = np.array([4000, 5000, 5777, 6500, 7500])  # K
    wavelengths = np.linspace(5200, 5300, 1000)
    
    results = {}
    
    for temp in temperatures:
        atmosphere_params = {
            'temperature': temp,
            'electron_density': 1.4e14 * (temp/5777)**0.5,  # Scale with T
            'neutral_density': 2.8e17,
            'mass_density': 2.3e-7,
            'turbulent_velocity': 1.2e5
        }
        
        abundances = {2601: 3.2e-5}  # Fe I
        
        opacity, stats = calc.calculate_linelist_opacity(
            wavelength_grid=wavelengths,
            atmosphere_params=atmosphere_params,
            abundance_dict=abundances,
            selection_criteria=selection_criteria
        )
        
        results[temp] = opacity
        print(f"T = {temp} K: max opacity = {stats['max_opacity']:.2e}")
    
    # Plot temperature dependence
    plt.figure(figsize=(12, 8))
    
    for temp, opacity in results.items():
        plt.plot(wavelengths, opacity, label=f'T = {temp} K', linewidth=1.5)
    
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Line Opacity (cm²/g)')
    plt.title(f'{line_species} Line Opacity vs Temperature')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return results
```

## Performance Optimization Tips

### 1. Line Selection Strategy

```python
# For computational efficiency, use aggressive line selection:
selection_criteria = {
    'min_log_gf': -1.0,      # Only very strong lines
    'max_excitation': 3.0,    # Only low excitation lines
    'species_filter': ['Fe 1', 'Ca 1', 'Mg 1', 'Si 1']  # Main contributors
}

# Limit total number of lines
max_lines = 1000  # Adjust based on computational resources
```

### 2. Wavelength Grid Optimization

```python
# Use appropriate resolution for your application
R_low = 1000      # For broad-band studies
R_medium = 10000  # For line identification
R_high = 100000   # For detailed line profile studies

# Calculate optimal grid
central_wave = 5500.0
delta_wave = central_wave / R_medium
wave_range = 100.0  # Å
n_points = int(wave_range / delta_wave)

wavelengths = np.linspace(5450, 5550, n_points)
```

### 3. Memory Management for Large Linelists

```python
def process_large_linelist_in_chunks(linelist_file, chunk_size=10000):
    """Process large linelist in chunks to manage memory"""
    
    calc = LinelistOpacityCalculator()
    
    # Load full linelist
    full_linelist = calc.load_linelist(linelist_file, 'vald')
    
    # Process in chunks
    total_opacity = None
    
    for i in range(0, len(full_linelist), chunk_size):
        chunk = full_linelist.iloc[i:i+chunk_size]
        calc.linelist_data = chunk
        
        # Convert chunk and calculate
        opacity, _ = calc.calculate_linelist_opacity(
            wavelength_grid=wavelengths,
            atmosphere_params=atmosphere_params,
            abundance_dict=abundances,
            max_lines=chunk_size
        )
        
        if total_opacity is None:
            total_opacity = opacity
        else:
            total_opacity += opacity
        
        print(f"Processed chunk {i//chunk_size + 1}, lines {i} to {min(i+chunk_size, len(full_linelist))}")
    
    return total_opacity
```

## Validation and Quality Control

### 1. Compare with Observational Data

```python
def validate_against_observations(observed_spectrum_file):
    """Compare calculated opacity with observed spectrum"""
    
    # Load observed spectrum
    obs_data = np.loadtxt(observed_spectrum_file)
    obs_wavelength = obs_data[:, 0]
    obs_flux = obs_data[:, 1]
    
    # Calculate line opacity
    calc_opacity, _ = calc.calculate_linelist_opacity(...)
    
    # Convert opacity to absorption (simplified)
    calculated_absorption = 1 - np.exp(-calc_opacity)
    
    # Interpolate to common grid
    common_wave = obs_wavelength
    calc_interp = np.interp(common_wave, wavelengths, calculated_absorption)
    
    # Compare
    plt.figure(figsize=(12, 6))
    plt.plot(common_wave, obs_flux, 'k-', label='Observed', alpha=0.7)
    plt.plot(common_wave, 1 - calc_interp, 'r--', label='Calculated', alpha=0.7)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Normalized Flux')
    plt.title('Observed vs Calculated Spectrum')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

### 2. Line Identification

```python
def identify_strong_lines(opacity_array, wavelength_array, threshold=0.1):
    """Identify strong absorption lines in calculated opacity"""
    
    from scipy.signal import find_peaks
    
    # Find peaks in opacity
    peaks, properties = find_peaks(opacity_array, height=threshold)
    
    peak_wavelengths = wavelength_array[peaks]
    peak_opacities = opacity_array[peaks]
    
    # Sort by strength
    sorted_indices = np.argsort(peak_opacities)[::-1]
    
    print("Strongest absorption lines:")
    for i in sorted_indices[:10]:
        wave = peak_wavelengths[i]
        opacity = peak_opacities[i]
        print(f"  {wave:.2f} Å: opacity = {opacity:.2e} cm²/g")
    
    return peak_wavelengths, peak_opacities
```

This comprehensive guide provides practical methods for using real astronomical linelists with the ATLAS12-style line opacity calculator, covering everything from basic usage to advanced applications and optimization strategies.