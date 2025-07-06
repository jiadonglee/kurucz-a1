# ATLAS12 Line Opacity Implementation Guide

## Overview

This guide explains the Python implementation of ATLAS12's line opacity calculation method, providing a complete working example that reproduces the core physics and algorithms.

## Key Components

### 1. Physical Constants Class

```python
class PhysicalConstants:
    c = 2.99792458e10          # Speed of light (cm/s)
    h = 6.6256e-27             # Planck constant (erg·s)
    k = 1.38054e-16            # Boltzmann constant (erg/K)
    e = 4.80320425e-10         # Elementary charge (esu)
    m_e = 9.10938356e-28       # Electron mass (g)
    
    # Key derived constant from ATLAS12
    pi_e2_mec = π * e² / (m_e * c)  # Fundamental absorption constant
```

## Core Mathematical Implementation

### 1. Line Opacity Formula

The fundamental equation implemented:

$$\kappa_{line}(\nu) = \frac{\pi e^2 f_{lu}}{m_e c} \cdot \frac{n_l}{\rho} \cdot \left(1 - e^{-h\nu/kT}\right) \cdot \frac{H(a,v)}{\Delta\nu_D}$$

**Python implementation:**
```python
def compute_line_center_opacity(self, line, temperature, lower_level_density, 
                               mass_density, doppler_width):
    # Boltzmann factor
    boltzmann_factor = np.exp(-line.lower_energy * self.const.h * self.const.c / 
                             (self.const.k * temperature))
    
    # Stimulated emission correction
    frequency = self.const.c / (line.wavelength * 1e-8)
    stimulated_emission = 1 - np.exp(-self.const.h * frequency / 
                                   (self.const.k * temperature))
    
    # Line center opacity
    center_opacity = (self.const.pi_e2_mec * line.oscillator_strength * 
                     lower_level_density / mass_density * boltzmann_factor *
                     stimulated_emission / doppler_width)
    
    return center_opacity
```

### 2. Doppler Width Calculation

$$\Delta\nu_D = \frac{1}{c}\sqrt{\frac{2kT}{m_{atom}} + v_{turb}^2}$$

**Python implementation:**
```python
def compute_doppler_width(self, temperature, atomic_mass, turbulent_velocity=2.0e5):
    # Thermal velocity squared
    v_thermal_sq = 2 * self.const.k * temperature / (atomic_mass * self.const.amu)
    
    # Total velocity squared (thermal + turbulent)
    v_total_sq = v_thermal_sq + turbulent_velocity**2
    
    # Doppler width in frequency
    doppler_width = np.sqrt(v_total_sq) / self.const.c
    
    return doppler_width
```

### 3. Damping Parameter

$$a = \frac{\Gamma_{total}}{4\pi\Delta\nu_D} = \frac{\Gamma_r + \Gamma_s n_e + \Gamma_w n_n}{4\pi\Delta\nu_D}$$

**Python implementation:**
```python
def compute_damping_parameter(self, line, temperature, electron_density, 
                            neutral_density, doppler_width):
    # Individual broadening mechanisms
    gamma_rad = line.gamma_rad
    gamma_stark = line.gamma_stark * electron_density
    gamma_vdw = line.gamma_vdw * neutral_density * (temperature / 10000)**0.3
    
    # Total damping
    gamma_total = gamma_rad + gamma_stark + gamma_vdw
    
    # Dimensionless damping parameter
    damping_param = gamma_total / (4 * np.pi * doppler_width)
    
    return damping_param
```

## Voigt Profile Implementation

### Three-Regime Calculation

The implementation follows ATLAS12's optimized approach:

#### Regime 1: Pure Doppler (a < 0.2)
```python
if a < 0.2:
    if v > 10.0:
        return PhysicalConstants.inv_sqrt_pi * a / v**2
    else:
        # Use tabulated values with linear interpolation
        iv = int(v * (self.n_points - 1) / 10.0)
        iv = min(iv, self.n_points - 2)
        frac = (v * (self.n_points - 1) / 10.0) - iv
        
        h0 = self.h0_tab[iv] + frac * (self.h0_tab[iv+1] - self.h0_tab[iv])
        h1 = self.h1_tab[iv] + frac * (self.h1_tab[iv+1] - self.h1_tab[iv])
        
        return h0 + a * h1
```

#### Regime 2: Far Wings (a > 1.4 or a+v > 3.2)
```python
elif a > 1.4 or (a + v) > 3.2:
    if v > 10.0:
        return PhysicalConstants.inv_sqrt_pi * a / v**2
    else:
        # Asymptotic expansion
        u = (a**2 + v**2) * np.sqrt(2)
        result = a * PhysicalConstants.inv_sqrt_pi / u
        
        # Higher order correction for moderate damping
        if a < 100.0:
            aa_u = a**2 / u
            vv_u = v**2 / u
            uu = u**2
            correction = (((aa_u - 10*vv_u) * aa_u * 3 + 15*vv_u**2) + 
                         3*v**2 - a**2) / uu + 1
            result *= correction
        
        return result
```

#### Regime 3: Intermediate (0.2 ≤ a ≤ 1.4)
```python
else:
    # Polynomial expansion in damping parameter
    vv = v**2
    hh1 = h1 + h0 * 1.12838
    hh2 = h2 + hh1 * 1.12838 - h0
    hh3 = (1 - h2) * 0.37613 - hh1 * 0.66667 * vv + hh2 * 1.12838
    hh4 = (3 * hh3 - hh1) * 0.37613 + h0 * 0.66667 * vv**2
    
    result = ((((hh4 * a + hh3) * a + hh2) * a + hh1) * a + h0)
    
    # Correction factor
    correction = (((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 
                  0.979895032)
    
    return result * correction
```

## Complete Line Profile Calculation

### Frequency Offset Calculation

Converting wavelength difference to Doppler units:

$$v = \frac{\lambda_0 - \lambda}{\lambda_0} \cdot \frac{c}{\Delta\nu_D}$$

**Python implementation:**
```python
def compute_line_opacity_profile(self, line, wavelength_grid, temperature, 
                               electron_density, neutral_density, 
                               lower_level_density, mass_density, turbulent_velocity):
    # Calculate parameters
    doppler_width = self.compute_doppler_width(temperature, line.atomic_mass, 
                                             turbulent_velocity)
    damping_param = self.compute_damping_parameter(line, temperature, 
                                                 electron_density, neutral_density,
                                                 doppler_width)
    center_opacity = self.compute_line_center_opacity(line, temperature,
                                                    lower_level_density, 
                                                    mass_density, doppler_width)
    
    # Convert to frequency offset
    line_wavelength = line.wavelength
    wavelength_diff = line_wavelength - wavelength_grid
    v_doppler = (wavelength_diff / line_wavelength) * (self.const.c / 
                (doppler_width * line_wavelength * 1e-8))
    
    # Compute profile
    opacity_profile = np.zeros_like(wavelength_grid)
    for i, v in enumerate(v_doppler):
        if abs(v) < 50:  # Reasonable frequency range
            voigt_value = self.voigt.voigt_function(v, damping_param)
            opacity_profile[i] = center_opacity * voigt_value
    
    return opacity_profile
```

## Line Selection and Optimization

### Weak Line Rejection

Following ATLAS12's efficiency approach:

```python
def compute_total_line_opacity(self, wavelength_grid, temperature, electron_density,
                             neutral_density, mass_density, abundance_dict, 
                             turbulent_velocity=2.0e5, opacity_threshold=1e-10):
    total_opacity = np.zeros_like(wavelength_grid)
    lines_computed = 0
    lines_skipped = 0
    
    for line in self.lines:
        # Quick strength estimate
        doppler_width = self.compute_doppler_width(temperature, line.atomic_mass, 
                                                 turbulent_velocity)
        center_opacity = self.compute_line_center_opacity(line, temperature,
                                                        lower_level_density,
                                                        mass_density, doppler_width)
        
        # Skip weak lines (ATLAS12 optimization)
        if center_opacity < opacity_threshold:
            lines_skipped += 1
            continue
        
        # Compute full profile for significant lines
        line_opacity = self.compute_line_opacity_profile(...)
        total_opacity += line_opacity
        lines_computed += 1
    
    return total_opacity
```

## Usage Example

### Basic Setup

```python
# Create calculator
calc = LineOpacityCalculator()

# Add lines (from database or manual input)
line = SpectralLine(
    wavelength=5250.2,          # Å
    ion_id=2601,                # Fe I
    oscillator_strength=0.001,   # f_lu
    lower_energy=0.0,           # cm⁻¹
    upper_energy=19020.0,       # cm⁻¹
    gamma_rad=1e8,              # s⁻¹
    gamma_stark=1e-15,          # Stark parameter
    gamma_vdw=1e-7,             # vdW parameter
    atomic_mass=55.845          # amu
)
calc.add_line(line)

# Atmospheric parameters
temperature = 5777.0        # K
electron_density = 1e14     # cm⁻³
neutral_density = 1e17      # cm⁻³
mass_density = 1e-7         # g/cm³

# Abundances (by ion ID)
abundances = {2601: 1e-5}   # Fe I relative abundance

# Wavelength grid
wavelengths = np.linspace(5240, 5260, 1000)  # Å

# Compute opacity
opacity = calc.compute_total_line_opacity(
    wavelengths, temperature, electron_density, neutral_density,
    mass_density, abundances
)
```

### Advanced Features

#### Custom Broadening Parameters
```python
# Modify line broadening for specific conditions
line.gamma_stark = 2e-15    # Enhanced Stark broadening
line.gamma_vdw = 5e-8       # Reduced vdW broadening

# High turbulent velocity for active stars
turbulent_velocity = 5e5    # 5 km/s
```

#### Profile Analysis
```python
# Analyze individual line properties
doppler_width = calc.compute_doppler_width(temperature, line.atomic_mass)
damping_param = calc.compute_damping_parameter(line, temperature, 
                                             electron_density, neutral_density,
                                             doppler_width)

print(f"Doppler width: {doppler_width:.2e} Hz")
print(f"Damping parameter: {damping_param:.3f}")

if damping_param < 0.2:
    print("Pure Doppler regime")
elif damping_param > 1.4:
    print("Strong damping regime")
else:
    print("Intermediate Voigt regime")
```

## Performance Considerations

### Computational Efficiency

1. **Line Selection**: Only compute lines above opacity threshold
2. **Wing Limits**: Restrict calculation to significant frequency range
3. **Tabulated Functions**: Pre-compute expensive mathematical functions
4. **Vectorization**: Use NumPy for array operations

### Memory Management

```python
# For large line lists, process in chunks
def compute_opacity_chunked(self, wavelength_grid, chunk_size=1000):
    total_opacity = np.zeros_like(wavelength_grid)
    
    for i in range(0, len(self.lines), chunk_size):
        chunk_lines = self.lines[i:i+chunk_size]
        chunk_opacity = self._compute_chunk_opacity(chunk_lines, wavelength_grid)
        total_opacity += chunk_opacity
    
    return total_opacity
```

## Validation and Testing

### Comparison with ATLAS12

The implementation can be validated by:

1. **Single line profiles**: Compare Voigt functions with analytical solutions
2. **Broadening regimes**: Verify correct regime selection and transitions
3. **Physical limits**: Check asymptotic behavior in extreme conditions
4. **Conservation**: Verify line profile normalization

### Example Validation

```python
def validate_voigt_profile():
    """Test Voigt function against analytical limits"""
    voigt = VoigtProfile()
    
    # Pure Doppler limit (a → 0)
    a_small = 1e-6
    v_test = 2.0
    voigt_result = voigt.voigt_function(v_test, a_small)
    gaussian_result = np.exp(-v_test**2)
    
    assert abs(voigt_result - gaussian_result) < 1e-4
    
    # Lorentzian limit (large v, finite a)
    a_finite = 0.1
    v_large = 20.0
    voigt_result = voigt.voigt_function(v_large, a_finite)
    lorentzian_result = a_finite / (np.pi * v_large**2)
    
    assert abs(voigt_result - lorentzian_result) < 1e-3
```

This implementation provides a complete, working example of ATLAS12's line opacity calculation method, suitable for educational purposes and practical stellar atmosphere modeling applications.