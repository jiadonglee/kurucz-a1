# ATLAS12 Energy Conservation Process

## Overview

ATLAS12 is a stellar atmosphere code that computes opacity sampling model atmospheres by enforcing energy conservation through **flux constancy**. The code iteratively solves the coupled equations of radiative transfer, hydrostatic equilibrium, and statistical equilibrium to produce self-consistent atmospheric models.

## Mathematical Framework

### Fundamental Equations

The stellar atmosphere structure is governed by three fundamental equations:

1. **Hydrostatic Equilibrium:**

   $$
   \frac{dP}{dr} = -\rho g + \frac{1}{c}\frac{dF_{rad}}{dr}
   $$
2. **Energy Conservation (Flux Constancy):**

   $$
   F_{rad} + F_{conv} = \sigma T_{eff}^4 = \text{constant}
   $$
3. **Radiative Transfer Equation:**

   $$
   \mu \frac{dI_\nu}{d\tau_\nu} = I_\nu - S_\nu
   $$

where:

- $P$ = gas pressure
- $\rho$ = density
- $g$ = surface gravity
- $F_{rad}$ = radiative flux
- $F_{conv}$ = convective flux
- $I_\nu$ = specific intensity
- $S_\nu$ = source function
- $\tau_\nu$ = optical depth
- $\mu$ = cosine of angle to normal

### Energy Conservation Implementation

The energy conservation is enforced through the **flux constancy condition**:

$$
\int_0^\infty F_\nu d\nu = \sigma T_{eff}^4
$$

The flux error at each atmospheric layer is defined as:

$$
\epsilon_F(τ) = \frac{F_{total}(τ) - \sigma T_{eff}^4}{\sigma T_{eff}^4}
$$

Temperature corrections are applied using the **Avrett-Krook method**:

$$
\Delta T = -\epsilon_F \left(\frac{\partial F}{\partial T}\right)^{-1}
$$

## Computational Algorithm

### Main Iteration Structure

The ATLAS12 calculation follows this iterative process:

```fortran
DO 100 ITERAT=1,NUMITS
  ITER=ITERAT
  [Physics calculations for each iteration]
100 CONTINUE
```

### Detailed Calculation Sequence

#### 1. Initialization Phase

**Frequency Grid Setup:**

- Wavelength range: $\lambda = 10^{1+0.0001(ν+ν_{start}-1)}$ Å
- 30,000 frequency sampling points
- Integration coefficients: $R_{co}(ν) = \frac{1}{2}(ν_{ν-1} - ν_{ν+1})$

**Integration Coefficient Calculation:**

$$
R_{co}(\nu) = \frac{c}{2}\left(\frac{1}{\lambda_{\nu-1}} - \frac{1}{\lambda_{\nu+1}}\right)
$$

#### 2. Per-Iteration Physics

##### A. Hydrostatic Equilibrium (Lines 211-232)

Update pressure structure using:

$$
P(τ) = g \int_0^τ \rho dτ' - P_{rad}(τ) - P_{turb}(τ) - P_{con}
$$

```fortran
DO 11 J=1,NRHOX
  P(J)=GRAV*RHOX(J)-PRAD(J)-PTURB(J)-PCON
11 CONTINUE
```

##### B. Statistical Equilibrium (Lines 235-236)

Solve population equations for all atomic/molecular species:

$$
\sum_j (n_i R_{ij} - n_j R_{ji}) = 0
$$

```fortran
CALL POPS(0.D0,1,XNE)     ! Ionization/excitation equilibrium
CALL POPSALL              ! All species populations  
```

##### C. Frequency Integration Loop (Lines 309-367)

For each of the 30,000 frequency points:

**Opacity Calculation:**

$$
\kappa_\nu = \kappa_{cont,\nu} + \kappa_{line,\nu} + \sigma_{scat,\nu}
$$

**Radiative Transfer Solution:**
Solve the formal solution:

$$
I_\nu(0,\mu) = \int_0^{\tau_{max}} S_\nu(τ') e^{-τ'/\mu} \frac{dτ'}{\mu}
$$

**Moment Calculations:**

- Mean intensity: $J_\nu = \frac{1}{2}\int_0^1 I_\nu(\mu) d\mu$
- Flux: $F_\nu = 2\pi \int_0^1 \mu I_\nu(\mu) d\mu$

```fortran
DO 25 NU=NULO,NUHI,NUSTEP    ! 30000 frequency points
  CALL KAPP                   ! κ_ν calculation
  CALL JOSH(IFSCAT,IFSURF)   ! Solve transfer equation
  CALL TCORR(2,RCOWT)        ! Accumulate flux integrals
  CALL RADIAP(2,RCOWT)       ! Radiation pressure
  CALL ROSS(2,RCOWT)         ! Rosseland mean opacity
25 CONTINUE
```

##### D. Radiation Field Integration

**Total Flux Integration:**

$$
F_{total} = \int_0^\infty F_\nu d\nu \approx \sum_{\nu=1}^{30000} F_\nu R_{co}(\nu)
$$

**Radiation Pressure:**

$$
P_{rad} = \frac{1}{3c} \int_0^\infty J_\nu d\nu
$$

**Rosseland Mean Opacity:**

$$
\frac{1}{\kappa_R} = \frac{\int_0^\infty \frac{1}{\kappa_\nu} \frac{\partial B_\nu}{\partial T} d\nu}{\int_0^\infty \frac{\partial B_\nu}{\partial T} d\nu}
$$

where $B_\nu$ is the Planck function:

$$
B_\nu(T) = \frac{2h\nu^3}{c^2} \frac{1}{e^{h\nu/kT} - 1}
$$

#### 3. Temperature Correction Phase (TCORR Subroutine)

##### Flux Error Calculation

For each atmospheric layer:

$$
\epsilon_F(j) = \frac{F_{rad}(j) + F_{conv}(j) - \sigma T_{eff}^4}{\sigma T_{eff}^4} \times 100
$$

```fortran
FLXERR(J)=(FLXRAD(J)+CNVFLX(J)-FLUX)/FLUX*100.
```

##### Avrett-Krook Temperature Correction

The temperature correction is computed using:

$$
\Delta T_\lambda(j) = -\frac{F_{drv}(j) \cdot \sigma T_{eff}^4}{100 \cdot R_{diag}(j)} \cdot \kappa_{Ross}(j)
$$

where:

- $F_{drv}(j)$ = flux derivative term
- $R_{diag}(j)$ = diagonal element of the correction matrix
- $\kappa_{Ross}(j)$ = Rosseland mean opacity

```fortran
DTLAMB(J)=-FLXDRV(J)*FLUX/100./RDIAGJ(J)*ABROSS(J)
```

##### Temperature Update

Apply corrections with damping for stability:

$$
T_{new}(j) = T_{old}(j) + \alpha \cdot \Delta T_\lambda(j)
$$

where $\alpha$ is a damping factor (typically 0.5-1.0).

## Key Physical Components

### 1. Continuous Opacity Sources

**Hydrogen Negative Ion (H⁻):**

- Bound-free: $\kappa_{bf}^{H-} \propto P_e T^{-2.5} f_{bf}(\lambda)$
- Free-free: $\kappa_{ff}^{H-} \propto P_e T^{-1.5} \lambda^3$

**Hydrogen Atom:**

- Bound-free: $\kappa_{bf}^H = n_n \sigma_{bf}^H(\nu)$
- Free-free: $\kappa_{ff}^H \propto n_e n_p T^{-1.5} \nu^{-3}$

**Thomson Scattering:**

$$
\sigma_T = \frac{8\pi}{3}\left(\frac{e^2}{m_e c^2}\right)^2 = 6.65 \times 10^{-25} \text{ cm}^2
$$

### 2. Line Opacity Treatment

ATLAS12 uses **opacity sampling** with precomputed line lists:

$$
\kappa_{line}(\nu) = \sum_{i} n_i \sigma_i \phi_i(\nu - \nu_0)
$$

where:

- $n_i$ = number density of absorbing level
- $\sigma_i$ = line absorption cross-section
- $\phi_i$ = line profile function (Voigt)

### 3. Convection Treatment

When convection is enabled, the **mixing length theory** is applied:

**Convective Flux:**

$$
F_{conv} = \frac{1}{2} \rho c_p T v_{conv} \nabla_{ad} \left(\nabla - \nabla_{ad}\right)
$$

**Convective Velocity:**

$$
v_{conv} = \sqrt{\frac{g \alpha H_p}{2}} \sqrt{\nabla - \nabla_{ad}}
$$

where:

- $H_p$ = pressure scale height
- $\alpha$ = mixing length parameter
- $\nabla$ = actual temperature gradient
- $\nabla_{ad}$ = adiabatic temperature gradient

## Convergence Criteria

The iteration continues until:

1. **Flux Error Convergence:**

   $$
   \max_j |\epsilon_F(j)| < \epsilon_{tol}
   $$
2. **Temperature Change Convergence:**

   $$
   \max_j \left|\frac{\Delta T(j)}{T(j)}\right| < \delta_{tol}
   $$

Typical values: $\epsilon_{tol} = 10^{-4}$, $\delta_{tol} = 10^{-4}$

## Output and Verification

### Model Verification

A converged ATLAS12 model satisfies:

1. **Flux Constancy:** $F(τ) = \sigma T_{eff}^4$ throughout the atmosphere
2. **Hydrostatic Equilibrium:** Pressure balances gravity and radiation pressure
3. **Statistical Equilibrium:** Atomic populations in detailed balance

### Typical Output

The final model provides:

- Temperature structure: $T(τ)$
- Pressure structure: $P(τ)$
- Density structure: $\rho(τ)$
- Chemical composition: $n_i(τ)$ for all species
- Opacity structure: $\kappa_\nu(τ)$
- Radiation field: $J_\nu(τ)$, $F_\nu(τ)$

## Implementation Notes

### Numerical Considerations

1. **Frequency Sampling:** 30,000 points ensure accurate integration
2. **Opacity Sampling:** Preselected lines relevant for stellar parameters
3. **Iterative Damping:** Prevents oscillations in temperature corrections
4. **Boundary Conditions:** Surface flux matches $\sigma T_{eff}^4$

### Performance Optimization

- **Line Preselection:** Only include relevant spectral lines
- **Binary Data:** Packed line data for efficient I/O
- **Vectorization:** Frequency loops optimized for performance
- **Memory Management:** Careful handling of large opacity arrays

This systematic approach ensures that ATLAS12 produces physically consistent stellar atmosphere models that rigorously conserve energy while maintaining computational efficiency.
