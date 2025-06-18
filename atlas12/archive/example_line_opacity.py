#!/usr/bin/env python3
"""
Simple example demonstrating ATLAS12-style line opacity calculation

This script shows how to use the line opacity calculator for a few representative
spectral lines in a solar-type atmosphere.
"""

import numpy as np
import matplotlib.pyplot as plt
from line_opacity_calculator import LineOpacityCalculator, SpectralLine

def main():
    """Run line opacity calculation example"""
    
    print("=== ATLAS12-Style Line Opacity Calculation Example ===\n")
    
    # Create calculator
    calc = LineOpacityCalculator()
    
    # Define a few important spectral lines
    print("Adding spectral lines...")
    
    # Iron lines (Fe I) - common in stellar spectra
    fe_line_1 = SpectralLine(
        wavelength=5250.2,          # Å (vacuum wavelength)
        ion_id=2601,                # Fe I identification
        oscillator_strength=0.001,   # Transition probability
        lower_energy=0.0,           # Lower level energy (cm⁻¹)
        upper_energy=19020.0,       # Upper level energy (cm⁻¹)
        gamma_rad=1.2e8,            # Radiative damping (s⁻¹)
        gamma_stark=1.5e-15,        # Stark broadening parameter
        gamma_vdw=1.2e-7,           # van der Waals broadening
        atomic_mass=55.845          # Atomic mass (amu)
    )
    
    fe_line_2 = SpectralLine(
        wavelength=5269.5, ion_id=2601, oscillator_strength=0.0005,
        lower_energy=415.9, upper_energy=19415.0, gamma_rad=1.0e8,
        gamma_stark=1.2e-15, gamma_vdw=1.0e-7, atomic_mass=55.845
    )
    
    # Calcium line (Ca I) - strong absorption
    ca_line = SpectralLine(
        wavelength=4226.7, ion_id=2001, oscillator_strength=2.0,
        lower_energy=0.0, upper_energy=23652.0, gamma_rad=2.5e8,
        gamma_stark=5e-16, gamma_vdw=2.5e-7, atomic_mass=40.078
    )
    
    # Magnesium line (Mg I)
    mg_line = SpectralLine(
        wavelength=5183.6, ion_id=1201, oscillator_strength=0.18,
        lower_energy=2.72, upper_energy=19285.0, gamma_rad=1.5e8,
        gamma_stark=1e-16, gamma_vdw=1.5e-7, atomic_mass=24.305
    )
    
    # Add lines to calculator
    for line in [fe_line_1, fe_line_2, ca_line, mg_line]:
        calc.add_line(line)
    
    print(f"Added {len(calc.lines)} spectral lines")
    
    # Define atmospheric parameters (solar photosphere values)
    print("\nAtmospheric parameters:")
    temperature = 5777.0        # Effective temperature (K)
    electron_density = 1.4e14   # Electron number density (cm⁻³)
    neutral_density = 2.8e17    # Neutral number density (cm⁻³)
    mass_density = 2.3e-7       # Mass density (g/cm³)
    turbulent_velocity = 1.2e5  # Turbulent velocity (cm/s) = 1.2 km/s
    
    print(f"  Temperature: {temperature} K")
    print(f"  Electron density: {electron_density:.1e} cm⁻³")
    print(f"  Neutral density: {neutral_density:.1e} cm⁻³")
    print(f"  Mass density: {mass_density:.1e} g/cm³")
    print(f"  Turbulent velocity: {turbulent_velocity/1e5:.1f} km/s")
    
    # Element abundances (number fractions relative to total neutrals)
    abundances = {
        1201: 3.8e-5,   # Mg I (log ε = -4.4)
        2001: 2.3e-6,   # Ca I (log ε = -5.6)
        2601: 3.2e-5    # Fe I (log ε = -4.5)
    }
    
    print(f"\nElement abundances:")
    for ion_id, abundance in abundances.items():
        element_names = {1201: "Mg I", 2001: "Ca I", 2601: "Fe I"}
        print(f"  {element_names[ion_id]}: {abundance:.1e}")
    
    # Create wavelength grid around Fe lines
    print(f"\nCreating wavelength grid...")
    wavelength_min, wavelength_max = 5240.0, 5280.0  # Å
    n_points = 2000
    wavelength_grid = np.linspace(wavelength_min, wavelength_max, n_points)
    print(f"  Range: {wavelength_min}-{wavelength_max} Å")
    print(f"  Resolution: {(wavelength_max-wavelength_min)/n_points:.3f} Å/pixel")
    
    # Calculate line opacity
    print(f"\nCalculating line opacity...")
    line_opacity = calc.compute_total_line_opacity(
        wavelength_grid=wavelength_grid,
        temperature=temperature,
        electron_density=electron_density,
        neutral_density=neutral_density,
        mass_density=mass_density,
        abundance_dict=abundances,
        turbulent_velocity=turbulent_velocity,
        opacity_threshold=1e-12
    )
    
    # Analyze results
    max_opacity = np.max(line_opacity)
    opacity_at_5250 = np.interp(5250.2, wavelength_grid, line_opacity)
    total_opacity_integral = np.trapz(line_opacity, wavelength_grid)
    
    print(f"\nResults:")
    print(f"  Maximum line opacity: {max_opacity:.2e} cm²/g")
    print(f"  Opacity at Fe I 5250.2 Å: {opacity_at_5250:.2e} cm²/g")
    print(f"  Integrated opacity: {total_opacity_integral:.2e} cm²/g·Å")
    
    # Analyze individual line properties
    print(f"\nLine analysis:")
    for i, line in enumerate([fe_line_1, fe_line_2]):
        doppler_width = calc.compute_doppler_width(temperature, line.atomic_mass, turbulent_velocity)
        damping_param = calc.compute_damping_parameter(
            line, temperature, electron_density, neutral_density, doppler_width
        )
        
        # Calculate line center opacity
        lower_level_density = abundances[line.ion_id] * neutral_density
        center_opacity = calc.compute_line_center_opacity(
            line, temperature, lower_level_density, mass_density, doppler_width
        )
        
        print(f"  Fe I {line.wavelength:.1f} Å:")
        print(f"    Doppler width: {doppler_width:.2e} Hz ({doppler_width*line.wavelength*1e-8/2.998e10*1e8:.3f} Å)")
        print(f"    Damping parameter: {damping_param:.4f}")
        print(f"    Center opacity: {center_opacity:.2e} cm²/g")
        
        if damping_param < 0.2:
            regime = "Pure Doppler"
        elif damping_param > 1.4:
            regime = "Strong damping"
        else:
            regime = "Intermediate Voigt"
        print(f"    Profile regime: {regime}")
    
    # Create plots
    print(f"\nCreating plots...")
    
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Full opacity spectrum
    plt.subplot(2, 2, 1)
    plt.plot(wavelength_grid, line_opacity, 'b-', linewidth=1)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Line Opacity (cm²/g)')
    plt.title('Line Opacity Spectrum (Fe I region)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Zoom on Fe I 5250.2 Å line
    plt.subplot(2, 2, 2)
    mask = (wavelength_grid >= 5249.0) & (wavelength_grid <= 5251.5)
    plt.plot(wavelength_grid[mask], line_opacity[mask], 'r-', linewidth=2)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Line Opacity (cm²/g)')
    plt.title('Fe I 5250.2 Å Line Profile')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Individual line contributions
    plt.subplot(2, 2, 3)
    
    # Calculate individual line profiles
    fe_5250_profile = calc.compute_line_opacity_profile(
        fe_line_1, wavelength_grid, temperature, electron_density,
        neutral_density, abundances[2601] * neutral_density, 
        mass_density, turbulent_velocity
    )
    
    fe_5269_profile = calc.compute_line_opacity_profile(
        fe_line_2, wavelength_grid, temperature, electron_density,
        neutral_density, abundances[2601] * neutral_density,
        mass_density, turbulent_velocity
    )
    
    plt.plot(wavelength_grid, fe_5250_profile, 'g-', label='Fe I 5250.2 Å', linewidth=1.5)
    plt.plot(wavelength_grid, fe_5269_profile, 'm--', label='Fe I 5269.5 Å', linewidth=1.5)
    plt.plot(wavelength_grid, line_opacity, 'k:', label='Total', linewidth=1)
    
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Line Opacity (cm²/g)')
    plt.title('Individual Line Contributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 4: Voigt profile demonstration
    plt.subplot(2, 2, 4)
    
    # Create detailed profile around Fe I 5250.2 Å
    wave_detail = np.linspace(5249.5, 5251.0, 500)
    fe_detail = calc.compute_line_opacity_profile(
        fe_line_1, wave_detail, temperature, electron_density,
        neutral_density, abundances[2601] * neutral_density,
        mass_density, turbulent_velocity
    )
    
    plt.plot(wave_detail, fe_detail / np.max(fe_detail), 'b-', linewidth=2)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Normalized Line Opacity')
    plt.title('Voigt Profile Shape (Fe I 5250.2 Å)')
    plt.grid(True, alpha=0.3)
    
    # Add vertical line at line center
    plt.axvline(fe_line_1.wavelength, color='r', linestyle='--', alpha=0.7, label='Line center')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nExample completed successfully!")
    print(f"The plots show:")
    print(f"  1. Full opacity spectrum showing multiple Fe I lines")
    print(f"  2. Detailed view of the strong Fe I 5250.2 Å line")
    print(f"  3. Individual line contributions to total opacity")
    print(f"  4. Normalized Voigt profile shape")

if __name__ == "__main__":
    main()