#!/usr/bin/env python3
"""
ATLAS12-style Line Opacity Calculator

This module implements the line opacity calculation method used in ATLAS12,
including Voigt profile computation, line broadening mechanisms, and 
efficient line selection algorithms.

Author: Based on ATLAS12 FORTRAN implementation by Robert Kurucz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings

# Physical constants (CGS units)
class PhysicalConstants:
    """Physical constants in CGS units as used in ATLAS12"""
    c = 2.99792458e10          # Speed of light (cm/s)
    h = 6.6256e-27             # Planck constant (erg·s)
    k = 1.38054e-16            # Boltzmann constant (erg/K)
    e = 4.80320425e-10         # Elementary charge (esu)
    m_e = 9.10938356e-28       # Electron mass (g)
    m_H = 1.6735575e-24        # Hydrogen mass (g)
    amu = 1.660539040e-24      # Atomic mass unit (g)
    
    # Derived constants
    pi_e2_mec = np.pi * e**2 / (m_e * c)  # π e²/(m_e c)
    inv_sqrt_pi = 1.0 / np.sqrt(np.pi)    # 1/√π = 0.5642

@dataclass
class SpectralLine:
    """Represents a single spectral line with all necessary parameters"""
    wavelength: float      # Wavelength in Å (vacuum)
    ion_id: int           # Ion identification number
    oscillator_strength: float  # Oscillator strength (f_lu)
    lower_energy: float   # Lower level energy (cm⁻¹)
    upper_energy: float   # Upper level energy (cm⁻¹)
    gamma_rad: float      # Radiative damping constant (s⁻¹)
    gamma_stark: float    # Stark broadening parameter
    gamma_vdw: float      # van der Waals broadening parameter
    atomic_mass: float    # Atomic mass (amu)
    
class VoigtProfile:
    """Fast Voigt profile calculation using ATLAS12 method"""
    
    def __init__(self, n_points: int = 2001):
        """Initialize with tabulated values for fast computation"""
        self.n_points = n_points
        self._setup_tables()
    
    def _setup_tables(self):
        """Setup tabulated Voigt function values for fast computation"""
        # Create tables similar to H0TAB, H1TAB, H2TAB in ATLAS12
        v_max = 10.0
        self.v_grid = np.linspace(0, v_max, self.n_points)
        
        # Pre-compute pure Doppler profiles for different damping values
        self.h0_tab = np.exp(-self.v_grid**2)  # Pure Gaussian
        
        # First derivative correction
        self.h1_tab = 2 * self.v_grid**2 * self.h0_tab - 1/np.sqrt(np.pi)
        
        # Second derivative correction  
        self.h2_tab = (4 * self.v_grid**4 - 2) * self.h0_tab
    
    def voigt_function(self, v: float, a: float) -> float:
        """
        Compute Voigt function H(a,v) using ATLAS12 three-regime method
        
        Args:
            v: Frequency offset in Doppler units
            a: Damping parameter
            
        Returns:
            Voigt function value
        """
        v = abs(v)  # Symmetric function
        
        # Regime 1: Pure Doppler (small damping)
        if a < 0.2:
            if v > 10.0:
                return PhysicalConstants.inv_sqrt_pi * a / v**2
            else:
                # Linear interpolation in pre-computed table
                iv = int(v * (self.n_points - 1) / 10.0)
                iv = min(iv, self.n_points - 2)
                frac = (v * (self.n_points - 1) / 10.0) - iv
                
                h0 = self.h0_tab[iv] + frac * (self.h0_tab[iv+1] - self.h0_tab[iv])
                h1 = self.h1_tab[iv] + frac * (self.h1_tab[iv+1] - self.h1_tab[iv])
                
                return h0 + a * h1
        
        # Regime 2: Large damping or far wings
        elif a > 1.4 or (a + v) > 3.2:
            # Asymptotic Lorentzian form
            if v > 10.0:
                return PhysicalConstants.inv_sqrt_pi * a / v**2
            else:
                # More accurate asymptotic expansion
                u = (a**2 + v**2) * np.sqrt(2)
                result = a * PhysicalConstants.inv_sqrt_pi / u
                
                if a < 100.0:  # Higher order correction
                    aa_u = a**2 / u
                    vv_u = v**2 / u
                    uu = u**2
                    correction = (((aa_u - 10*vv_u) * aa_u * 3 + 15*vv_u**2) + 3*v**2 - a**2) / uu + 1
                    result *= correction
                
                return result
        
        # Regime 3: Intermediate regime - polynomial approximation
        else:
            if v > 10.0:
                return PhysicalConstants.inv_sqrt_pi * a / v**2
            
            # Interpolate in table
            iv = int(v * (self.n_points - 1) / 10.0)
            iv = min(iv, self.n_points - 2)
            frac = (v * (self.n_points - 1) / 10.0) - iv
            
            h0 = self.h0_tab[iv] + frac * (self.h0_tab[iv+1] - self.h0_tab[iv])
            h1 = self.h1_tab[iv] + frac * (self.h1_tab[iv+1] - self.h1_tab[iv])
            h2 = self.h2_tab[iv] + frac * (self.h2_tab[iv+1] - self.h2_tab[iv])
            
            # Polynomial expansion in damping parameter
            vv = v**2
            hh1 = h1 + h0 * 1.12838
            hh2 = h2 + hh1 * 1.12838 - h0
            hh3 = (1 - h2) * 0.37613 - hh1 * 0.66667 * vv + hh2 * 1.12838
            hh4 = (3 * hh3 - hh1) * 0.37613 + h0 * 0.66667 * vv**2
            
            result = ((((hh4 * a + hh3) * a + hh2) * a + hh1) * a + h0)
            
            # Correction factor for intermediate regime
            correction = (((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 0.979895032)
            
            return result * correction

class LineOpacityCalculator:
    """Main class for calculating line opacity following ATLAS12 methodology"""
    
    def __init__(self):
        self.voigt = VoigtProfile()
        self.lines = []
        self.const = PhysicalConstants()
    
    def add_line(self, line: SpectralLine):
        """Add a spectral line to the line list"""
        self.lines.append(line)
    
    def compute_doppler_width(self, temperature: float, atomic_mass: float, 
                            turbulent_velocity: float = 2.0e5) -> float:
        """
        Compute Doppler width in frequency units
        
        Args:
            temperature: Temperature in K
            atomic_mass: Atomic mass in amu
            turbulent_velocity: Turbulent velocity in cm/s
            
        Returns:
            Doppler width in Hz
        """
        # Thermal velocity squared
        v_thermal_sq = 2 * self.const.k * temperature / (atomic_mass * self.const.amu)
        
        # Total velocity squared (thermal + turbulent)
        v_total_sq = v_thermal_sq + turbulent_velocity**2
        
        # Doppler width in frequency
        doppler_width = np.sqrt(v_total_sq) / self.const.c
        
        return doppler_width
    
    def compute_damping_parameter(self, line: SpectralLine, temperature: float,
                                electron_density: float, neutral_density: float,
                                doppler_width: float) -> float:
        """
        Compute total damping parameter
        
        Args:
            line: Spectral line object
            temperature: Temperature in K
            electron_density: Electron number density (cm⁻³)
            neutral_density: Neutral number density (cm⁻³)
            doppler_width: Doppler width in Hz
            
        Returns:
            Damping parameter (dimensionless)
        """
        # Radiative damping
        gamma_rad = line.gamma_rad
        
        # Stark damping (proportional to electron density)
        gamma_stark = line.gamma_stark * electron_density
        
        # van der Waals damping (proportional to neutral density and temperature)
        gamma_vdw = line.gamma_vdw * neutral_density * (temperature / 10000)**0.3
        
        # Total damping
        gamma_total = gamma_rad + gamma_stark + gamma_vdw
        
        # Convert to dimensionless damping parameter
        # a = Γ_total / (4π Δν_D)
        damping_param = gamma_total / (4 * np.pi * doppler_width)
        
        return damping_param
    
    def compute_line_center_opacity(self, line: SpectralLine, temperature: float,
                                  lower_level_density: float, mass_density: float,
                                  doppler_width: float) -> float:
        """
        Compute line center opacity coefficient
        
        Args:
            line: Spectral line object
            temperature: Temperature in K
            lower_level_density: Number density of lower level (cm⁻³)
            mass_density: Mass density (g/cm³)
            doppler_width: Doppler width in Hz
            
        Returns:
            Line center opacity coefficient
        """
        # Frequency of line center
        frequency = self.const.c / (line.wavelength * 1e-8)  # Convert Å to cm
        
        # Boltzmann factor for lower level
        boltzmann_factor = np.exp(-line.lower_energy * self.const.h * self.const.c / 
                                (self.const.k * temperature))
        
        # Stimulated emission correction
        stimulated_emission = 1 - np.exp(-self.const.h * frequency / 
                                       (self.const.k * temperature))
        
        # Line center opacity
        # κ₀ = (π e² / m_e c) × f_lu × (n_l / ρ) × (1 - exp(-hν/kT)) / Δν_D
        center_opacity = (self.const.pi_e2_mec * line.oscillator_strength * 
                         lower_level_density / mass_density * boltzmann_factor *
                         stimulated_emission / doppler_width)
        
        return center_opacity
    
    def compute_line_opacity_profile(self, line: SpectralLine, wavelength_grid: np.ndarray,
                                   temperature: float, electron_density: float,
                                   neutral_density: float, lower_level_density: float,
                                   mass_density: float, turbulent_velocity: float = 2.0e5) -> np.ndarray:
        """
        Compute line opacity profile across wavelength grid
        
        Args:
            line: Spectral line object
            wavelength_grid: Wavelength grid in Å
            temperature: Temperature in K
            electron_density: Electron density (cm⁻³)
            neutral_density: Neutral density (cm⁻³)
            lower_level_density: Lower level density (cm⁻³)
            mass_density: Mass density (g/cm³)
            turbulent_velocity: Turbulent velocity (cm/s)
            
        Returns:
            Line opacity profile (cm²/g)
        """
        # Compute Doppler width
        doppler_width = self.compute_doppler_width(temperature, line.atomic_mass, 
                                                 turbulent_velocity)
        
        # Compute damping parameter
        damping_param = self.compute_damping_parameter(line, temperature, 
                                                     electron_density, neutral_density,
                                                     doppler_width)
        
        # Compute line center opacity
        center_opacity = self.compute_line_center_opacity(line, temperature,
                                                        lower_level_density, 
                                                        mass_density, doppler_width)
        
        # Convert wavelength grid to frequency offset in Doppler units
        line_wavelength = line.wavelength  # Å
        
        # v = (λ₀ - λ) / λ₀ × c / Δν_D
        wavelength_diff = line_wavelength - wavelength_grid
        v_doppler = (wavelength_diff / line_wavelength) * (self.const.c / 
                    (doppler_width * line_wavelength * 1e-8))
        
        # Compute Voigt profile for each wavelength point
        opacity_profile = np.zeros_like(wavelength_grid)
        
        for i, v in enumerate(v_doppler):
            if abs(v) < 50:  # Only compute for reasonable frequency offsets
                voigt_value = self.voigt.voigt_function(v, damping_param)
                opacity_profile[i] = center_opacity * voigt_value
        
        return opacity_profile
    
    def compute_total_line_opacity(self, wavelength_grid: np.ndarray,
                                 temperature: float, electron_density: float,
                                 neutral_density: float, mass_density: float,
                                 abundance_dict: dict, turbulent_velocity: float = 2.0e5,
                                 opacity_threshold: float = 1e-10) -> np.ndarray:
        """
        Compute total line opacity from all lines in the line list
        
        Args:
            wavelength_grid: Wavelength grid in Å
            temperature: Temperature in K
            electron_density: Electron density (cm⁻³)
            neutral_density: Neutral density (cm⁻³)
            mass_density: Mass density (g/cm³)
            abundance_dict: Dictionary of abundances by ion_id
            turbulent_velocity: Turbulent velocity (cm/s)
            opacity_threshold: Minimum opacity to include
            
        Returns:
            Total line opacity profile (cm²/g)
        """
        total_opacity = np.zeros_like(wavelength_grid)
        lines_computed = 0
        lines_skipped = 0
        
        for line in self.lines:
            # Get abundance for this ion
            if line.ion_id not in abundance_dict:
                continue
                
            abundance = abundance_dict[line.ion_id]
            lower_level_density = abundance * neutral_density  # Simplified
            
            # Quick check: estimate line strength
            doppler_width = self.compute_doppler_width(temperature, line.atomic_mass, 
                                                     turbulent_velocity)
            center_opacity = self.compute_line_center_opacity(line, temperature,
                                                            lower_level_density,
                                                            mass_density, doppler_width)
            
            # Skip weak lines (ATLAS12 optimization)
            if center_opacity < opacity_threshold:
                lines_skipped += 1
                continue
            
            # Compute full line profile
            line_opacity = self.compute_line_opacity_profile(
                line, wavelength_grid, temperature, electron_density,
                neutral_density, lower_level_density, mass_density, turbulent_velocity
            )
            
            total_opacity += line_opacity
            lines_computed += 1
        
        print(f"Computed {lines_computed} lines, skipped {lines_skipped} weak lines")
        return total_opacity

def create_sample_line_list() -> List[SpectralLine]:
    """Create a sample line list for demonstration"""
    
    lines = [
        # Fe I lines (iron neutral)
        SpectralLine(
            wavelength=5250.2, ion_id=2601, oscillator_strength=0.001,
            lower_energy=0.0, upper_energy=19020.0, gamma_rad=1e8,
            gamma_stark=1e-15, gamma_vdw=1e-7, atomic_mass=55.845
        ),
        SpectralLine(
            wavelength=5269.5, ion_id=2601, oscillator_strength=0.0005,
            lower_energy=415.9, upper_energy=19415.0, gamma_rad=1e8,
            gamma_stark=1e-15, gamma_vdw=1e-7, atomic_mass=55.845
        ),
        # Ca I lines (calcium neutral)
        SpectralLine(
            wavelength=4226.7, ion_id=2001, oscillator_strength=2.0,
            lower_energy=0.0, upper_energy=23652.0, gamma_rad=2e8,
            gamma_stark=5e-16, gamma_vdw=2e-7, atomic_mass=40.078
        ),
        # Mg I line (magnesium neutral)
        SpectralLine(
            wavelength=5183.6, ion_id=1201, oscillator_strength=0.18,
            lower_energy=2.72, upper_energy=19285.0, gamma_rad=1e8,
            gamma_stark=1e-16, gamma_vdw=1e-7, atomic_mass=24.305
        ),
        # H alpha (hydrogen)
        SpectralLine(
            wavelength=6562.8, ion_id=101, oscillator_strength=0.64,
            lower_energy=82259.0, upper_energy=97492.0, gamma_rad=4.7e8,
            gamma_stark=1e-14, gamma_vdw=1e-8, atomic_mass=1.008
        )
    ]
    
    return lines

def main():
    """Demonstration of line opacity calculation"""
    
    # Create calculator and add sample lines
    calc = LineOpacityCalculator()
    sample_lines = create_sample_line_list()
    
    for line in sample_lines:
        calc.add_line(line)
    
    # Atmospheric parameters (typical solar photosphere)
    temperature = 5777.0      # K
    electron_density = 1e14   # cm⁻³
    neutral_density = 1e17    # cm⁻³
    mass_density = 1e-7       # g/cm³
    turbulent_velocity = 1e5  # cm/s (1 km/s)
    
    # Element abundances (simplified)
    abundances = {
        101: 1e-5,    # H (relative to neutrals)
        1201: 1e-5,   # Mg I
        2001: 1e-6,   # Ca I
        2601: 1e-5    # Fe I
    }
    
    # Wavelength grid
    wavelength_min, wavelength_max = 4000.0, 7000.0  # Å
    n_wavelengths = 3000
    wavelength_grid = np.linspace(wavelength_min, wavelength_max, n_wavelengths)
    
    print("Computing line opacity...")
    print(f"Temperature: {temperature} K")
    print(f"Electron density: {electron_density:.1e} cm⁻³")
    print(f"Number of lines: {len(sample_lines)}")
    print(f"Wavelength range: {wavelength_min}-{wavelength_max} Å")
    
    # Compute total line opacity
    line_opacity = calc.compute_total_line_opacity(
        wavelength_grid, temperature, electron_density, neutral_density,
        mass_density, abundances, turbulent_velocity
    )
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.semilogy(wavelength_grid, line_opacity + 1e-15)  # Add small value to avoid log(0)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Line Opacity (cm²/g)')
    plt.title('ATLAS12-style Line Opacity Calculation')
    plt.grid(True, alpha=0.3)
    
    # Individual line demonstration
    plt.subplot(2, 1, 2)
    
    # Show H alpha profile in detail
    h_alpha = sample_lines[4]  # H alpha line
    wave_range = np.linspace(6560, 6566, 1000)
    
    h_alpha_opacity = calc.compute_line_opacity_profile(
        h_alpha, wave_range, temperature, electron_density,
        neutral_density, abundances[101] * neutral_density, 
        mass_density, turbulent_velocity
    )
    
    plt.plot(wave_range, h_alpha_opacity)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('H α Opacity (cm²/g)')
    plt.title('H α Line Profile (Voigt)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    max_opacity = np.max(line_opacity)
    total_opacity_integral = np.trapz(line_opacity, wavelength_grid)
    
    print(f"\nResults:")
    print(f"Maximum line opacity: {max_opacity:.2e} cm²/g")
    print(f"Integrated line opacity: {total_opacity_integral:.2e} cm²/g·Å")
    
    # Compare different broadening regimes
    print(f"\nBroadening analysis for H alpha:")
    doppler_width = calc.compute_doppler_width(temperature, h_alpha.atomic_mass, turbulent_velocity)
    damping_param = calc.compute_damping_parameter(h_alpha, temperature, 
                                                 electron_density, neutral_density, doppler_width)
    print(f"Doppler width: {doppler_width:.2e} Hz")
    print(f"Damping parameter: {damping_param:.3f}")
    
    if damping_param < 0.2:
        print("Regime: Pure Doppler (Gaussian core)")
    elif damping_param > 1.4:
        print("Regime: Strong damping (Lorentzian wings)")
    else:
        print("Regime: Intermediate (Voigt profile)")

if __name__ == "__main__":
    main()