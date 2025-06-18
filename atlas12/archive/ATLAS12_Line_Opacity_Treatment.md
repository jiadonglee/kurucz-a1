# ATLAS12 Line Opacity Treatment

## Overview

ATLAS12 implements sophisticated line opacity calculations using **opacity sampling** with precomputed line lists. The code handles millions of spectral lines with optimized Voigt profile calculations and efficient line selection algorithms.

## Mathematical Foundation

### Basic Line Opacity Formula

The fundamental line opacity equation is:

$$\kappa_{line}(\nu) = \sum_{lines} \frac{\pi e^2 f_{lu}}{m_e c} \frac{n_l}{\rho} \left(1 - e^{-h\nu/kT}\right) H(a,v)$$

where:
- $f_{lu}$ = oscillator strength for transition from lower (l) to upper (u) level
- $n_l$ = number density of atoms in lower level
- $\rho$ = mass density
- $H(a,v)$ = Voigt profile function
- $a$ = damping parameter
- $v$ = frequency offset in Doppler units

### Individual Line Contribution

Each line contributes:

$$\kappa_{line,i}(\nu) = \frac{\pi e^2 f_{lu}}{m_e c} \cdot \frac{n_l}{\rho \Delta\nu_D} \cdot e^{-E_{low}/kT} \cdot H(a,v)$$

## Core Implementation (LINOP Subroutine)

### 1. Line Strength Calculation

**Primary formula** (atlas12.for lines 9703-9706):

```fortran
CENTER = CGF * XNFDOP(J,NELION) * FASTEX(ELO*HCKT4(J))
```

**Physical interpretation**:

$$CENTER = \frac{\pi e^2 f_{lu}}{m_e c} \cdot \frac{n_l/\rho}{\Delta\nu_D} \cdot e^{-E_{low}/kT}$$

**Components**:
- **CGF** = $\frac{\pi e^2 f_{lu}}{m_e c}$ (oscillator strength × fundamental constants)
- **XNFDOP(J,NELION)** = $\frac{n_l/\rho}{\Delta\nu_D}$ (population density per Doppler width)
- **FASTEX(ELO*HCKT4(J))** = $e^{-E_{low}/kT}$ (Boltzmann excitation factor)

### 2. Doppler Broadening

**Doppler width calculation**:

```fortran
DOPPLE(J,NELION) = SQRT(2.*TK(J)/AMASSISO(1,NELION)/1.660E-24 + VTURB(J)**2)
                   / 2.99792458E10
```

**Physical formula**:

$$\Delta\nu_D = \frac{1}{c}\sqrt{\frac{2kT}{m_{atom}} + v_{turb}^2}$$

**Population factor**:

```fortran
XNFDOP(J,NELION) = XNFP(J,NELION) / DOPPLE(J,NELION) / RHO(J)
```

## Line Profile Implementation

### 1. Damping Parameter

**Total damping** (line 9708):

```fortran
ADAMP = (GAMMAR + GAMMAS*XNE4(J) + GAMMAW*TXNXN(J)) / DOPPLE(J,NELION)
```

**Physical interpretation**:

$$a = \frac{\Gamma_{total}}{4\pi\Delta\nu_D} = \frac{\Gamma_r + \Gamma_s n_e + \Gamma_w n_n}{4\pi\Delta\nu_D}$$

**Broadening mechanisms**:
- **GAMMAR** ($\Gamma_r$): Radiative (natural) damping
- **GAMMAS** ($\Gamma_s$): Stark broadening (∝ electron density)
- **GAMMAW** ($\Gamma_w$): van der Waals broadening (∝ neutral density)

### 2. Voigt Profile Calculation

ATLAS12 uses a **three-regime Voigt function** for optimal accuracy and speed:

#### A. Pure Doppler Regime (a < 0.2)

```fortran
IF(A.LT..2) VOIGT = (H2TAB(IV)*A + H1TAB(IV))*A + H0TAB(IV)
```

**Approximation**:
$$H(a,v) \approx e^{-v^2} + \frac{a}{\sqrt{\pi}}[2v^2 e^{-v^2} - 1 + e^{-v^2}]$$

Uses **polynomial expansion** in damping parameter with tabulated Gaussian values.

#### B. Intermediate Regime (0.2 ≤ a ≤ 1.4, a+v ≤ 3.2)

```fortran
HH1 = H1TAB(IV) + H0TAB(IV)*1.12838
HH2 = H2TAB(IV) + HH1*1.12838 - H0TAB(IV)
HH3 = (1.-H2TAB(IV))*.37613 - HH1*.66667*VV + HH2*1.12838
HH4 = (3.*HH3-HH1)*.37613 + H0TAB(IV)*.66667*VV*VV
VOIGT = ((((HH4*A+HH3)*A+HH2)*A+HH1)*A+H0TAB(IV)) * correction_factor
```

Uses **high-order polynomial** in both frequency and damping with correction factors.

#### C. Far Wing Regime (large a or v)

```fortran
IF(V.GT.10.) VOIGT = .5642*A/V**2     ! Lorentzian wings
```

**Asymptotic form**:
$$H(a,v) \approx \frac{a}{\pi v^2} = \frac{0.5642 \cdot a}{v^2} \text{ for } v >> a$$

The constant 0.5642 = $1/\sqrt{\pi}$.

### 3. Frequency Offset Calculation

**Dimensionless frequency offset**:

$$v = \frac{\nu - \nu_0}{\Delta\nu_D} = \frac{\lambda_0 - \lambda}{\lambda_0} \cdot \frac{c}{\Delta\nu_D}$$

**Implementation**:
```fortran
! Red wing
VVOIGT = SNGL(WAVESET(IW)-WLVAC)/DOPWAVE

! Blue wing  
VVOIGT = SNGL(WLVAC-WAVESET(IW))/DOPWAVE
```

where `DOPWAVE = DOPPLE(J,NELION)*WLVAC` is the Doppler width in wavelength units.

## Line Wing Integration

### 1. Wing Calculation Algorithm

**Red wing** (lines 9712-9722):
```fortran
DO 557 IW=NU,MIN(NU+100,NUMNU)
  VVOIGT = SNGL(WAVESET(IW)-WLVAC)/DOPWAVE
  CV = CENTER * VOIGT_FUNCTION(VVOIGT, ADAMP)
  XLINES(J,IW) = XLINES(J,IW) + CV
  IF(CV.LT.TABCONT(J,NUCONT)) GO TO 558  ! Wing cutoff
557 CONTINUE
```

**Blue wing** (lines 9724-9736):
```fortran
DO 559 I=1,100
  IW = NU-I
  IF(IW.LE.0) GO TO 580
  VVOIGT = SNGL(WLVAC-WAVESET(IW))/DOPWAVE  
  CV = CENTER * VOIGT_FUNCTION(VVOIGT, ADAMP)
  XLINES(J,IW) = XLINES(J,IW) + CV
  IF(CV.LT.TABCONT(J,NUCONT)) GO TO 580
559 CONTINUE
```

### 2. Wing Cutoff Criteria

- **Maximum extent**: 100 frequency points per wing
- **Opacity threshold**: Stop when line opacity < continuum opacity
- **Efficiency**: Prevents unnecessary computation of negligible contributions

## Line Selection and Optimization

### 1. Line Strength Threshold

```fortran
IF(CENTER.LT.TABCONT(J,NUCONT)) GO TO 580  ! Skip weak lines
```

**Criterion**: Only compute lines where **central opacity** exceeds continuum opacity.

### 2. Computational Optimizations

#### A. Sparse Layer Processing
```fortran
DO 580 J=8,NRHOX,8    ! Process every 8th layer first
```
Process representative layers first, then interpolate for intermediate layers.

#### B. Fast Exponential Function
```fortran
FASTEX(X) = EXTAB(INT(X)+1)*EXTABF(INT((X-REAL(INT(X)))*1000.+1.5))
```
Uses **pre-tabulated exponentials** for rapid Boltzmann factor calculation.

#### C. Early Wing Termination
Stops computing line wings when opacity contribution becomes negligible.

## Line Data Format

### 1. Packed Binary Storage

Lines stored in compressed format with:
- **IWL**: Wavelength (logarithmic encoding)
- **IELION**: Ion identification  
- **IGFLOG**: Log oscillator strength
- **ELO**: Lower level energy
- **GAMMAR**: Radiative damping
- **GAMMAS**: Stark broadening coefficient
- **GAMMAW**: van der Waals coefficient

### 2. Data Reading Process

```fortran
DO 600 LINE=1,500000000
  read(13,END=601) WWWWWWW    ! Read packed line data
  ! Decode: WLVAC, NELION, CGF, ELO, GAMMAR, GAMMAS, GAMMAW
```

## Integration with Atomic Physics

### 1. Population Calculations

**XNFP(J,NELION)** contains upper level populations:

$$\frac{n_u}{Q(T)} = \frac{n_{total} \cdot A_{element} \cdot f_{ion} \cdot g_u e^{-E_u/kT}}{Q(T)}$$

### 2. Statistical Equilibrium

Population equations solved in **POPS** and **POPSALL** subroutines:

$$\sum_j (n_i R_{ij} - n_j R_{ji}) = 0$$

### 3. NLTE Effects

- **LTE assumption**: $S_\nu = B_\nu(T)$ for most lines
- **NLTE corrections**: Applied for hydrogen (HLINOP) and selected species
- **Departure coefficients**: Modify level populations when needed

## Physical Constants and Conversions

### 1. Fundamental Constants
- **π e²/(mₑc)** = 0.026538 (atomic absorption constant)
- **c** = 2.99792458×10¹⁰ cm/s (speed of light)
- **k** = 1.38054×10⁻¹⁶ erg/K (Boltzmann constant)
- **h** = 6.6256×10⁻²⁷ erg·s (Planck constant)

### 2. Conversion Factors
- **1/√π** = 0.5642 (Lorentzian normalization)
- **Wavelength factor**: 1.77245 (√π normalization)
- **Atomic mass unit**: 1.660×10⁻²⁴ g

## Performance Characteristics

### 1. Line Selection Efficiency
- **Typical dataset**: ~10⁶ lines from atomic databases
- **Selected lines**: ~10⁴ - 10⁵ lines per model (temperature dependent)
- **Active lines**: ~10³ lines per frequency point

### 2. Computational Scaling
- **Frequency grid**: 30,000 points
- **Line evaluation**: O(N_lines × N_wings × N_layers)
- **Optimization**: Smart line selection reduces computational load by 10-100×

### 3. Accuracy vs Speed Trade-offs
- **3-regime Voigt**: Maintains accuracy while optimizing speed
- **Tabulated functions**: Pre-computed exponentials and profiles
- **Wing cutoffs**: Balance between accuracy and computational efficiency

## Summary

ATLAS12's line opacity treatment represents a sophisticated implementation that:

1. **Handles millions of spectral lines** efficiently through smart selection algorithms
2. **Implements accurate Voigt profiles** using optimized 3-regime calculation
3. **Includes all major broadening mechanisms** (natural, Stark, van der Waals, thermal, turbulent)
4. **Integrates seamlessly** with atomic population calculations
5. **Balances accuracy and speed** through various optimization techniques
6. **Supports both LTE and NLTE** treatments where appropriate

This approach enables ATLAS12 to produce high-fidelity stellar atmosphere models that accurately represent the complex line-blanketing effects essential for modern astrophysical applications.