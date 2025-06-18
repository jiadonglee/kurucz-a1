# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is ATLAS12, a stellar atmosphere model atmosphere program for computing opacity sampling models. The codebase contains both the original FORTRAN implementation and a modern Python port/wrapper. ATLAS12 is designed for precise stellar atmosphere modeling, particularly for A, F, G, K, and M stars, with capabilities for handling extensive line lists and molecular data.

## Architecture

### Core Components

**Original FORTRAN Code:**
- `atlas12.for` - Main ATLAS12 program (2100+ lines of FORTRAN)
- `synthe.for` - Spectrum synthesis program 
- `xnfpelsyn.for` - Preprocessor for equation of state calculations
- `rnlteall.for` - NLTE line data preprocessor

**Python Implementation:**
- `atlas12.py` - Main Python class wrapper
- `atmosphere.py` - Atmospheric model structure and physics calculations
- `opacity.py` - Continuous opacity calculations (includes Chinese comments)
- `radiative_transfer.py` - Radiative transfer equation solver
- `chemistry.py` - Chemical abundance handler (minimal implementation)
- `convection.py` - Convective transport handler (minimal implementation)

### Data Structure

**Binary Data Files:** (`data/` directory)
- `*.bin` files contain packed line data from various CD-ROMs
- `nltelines.asc/.bin` - NLTE line data
- `diatomicsiwl.bin`, `tioschwenke.bin`, `eschwenke.bin` - Molecular line data

**Control Files:**
- `*.com` files are VMS-style command procedures for running different models
- `*.dat` files contain atmospheric models, line lists, and physical data

## Development Workflow

### Building and Running FORTRAN Code

**Compile FORTRAN programs:**
```bash
# The original uses VMS FORTRAN, modern systems need gfortran
gfortran -o atlas12 atlas12.for
gfortran -o rnlteall rnlteall.for  
gfortran -o synthe synthe.for
```

**Prepare line data:**
```bash
# Create binary NLTE line data (run once)
./rnlteall < rnlteall.com
```

**Run atmospheric model computation:**
```bash
# Select line data first
./atlas12 < at8650g40a.com
# Then compute model
./atlas12 < at8750g38i.com
```

### Python Development

**Run Python implementation:**
```python
from atlas12 import ATLAS12
model = ATLAS12()
model.initialize_model()
```

**Test continuous opacity:**
```python
from opacity import ContinuousOpacity
opacity = ContinuousOpacity()
opacity.calculate_opacity()
opacity.plot_opacity()
```

## Key Implementation Details

### FORTRAN Program Structure
- Uses extensive COMMON blocks for variable sharing
- Supports up to 30,000 wavelength sampling points
- Handles up to 72 atmospheric layers (parameter `kw=72`)
- Maximum 1006 ionic species (parameter `mion=1006`)

### Python Class Hierarchy
- Main `ATLAS12` class coordinates all components
- `AtmosphereModel` handles physical structure (temperature, pressure, density)
- `RadiativeTransfer` implements formal solution of transfer equation
- `OpacityHandler` computes continuous and line opacities

### Model Parameters
- Temperature structure: T(τ) where τ is optical depth
- Pressure structure: P(τ) from hydrostatic equilibrium  
- Chemical abundances: Solar-scaled or custom abundance tables
- Line data: Precomputed from CD-ROM sources or data files

### File Formats
- `.dat` files: Either ASCII model atmospheres or binary line data
- `.com` files: VMS command procedures (text control files)
- `.for` files: FORTRAN source code
- `.log` files: Program execution logs and convergence history

## Important Notes

- ATLAS12 requires starting models close to desired parameters (cannot scale in temperature/gravity)
- Line preselection is critical for performance - done via `at8650g40a.com` type runs
- Spectrum synthesis requires separate SYNTHE program with processed atmospheric models
- The Python implementation is a modernization effort and may not have full feature parity
- Binary data files are platform-dependent and may need regeneration on different systems

## Typical Model Computation Sequence

1. **Prepare line data:** Run `rnlteall.com` to create `BNLTELINES.DAT`
2. **Select lines:** Run line selection (e.g., `at8650g40a.com`) to create filtered line list
3. **Compute model:** Run main calculation (e.g., `at8750g38i.com`) with desired parameters
4. **Generate spectrum:** Use SYNTHE suite (`xnfpelsyn`, `synthe`, `spectrv`) for spectrum synthesis
5. **Convert output:** Use `asciitriplets` to convert binary spectrum to ASCII format