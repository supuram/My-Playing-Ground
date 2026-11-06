import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW
from gpaw.response.df import DielectricFunction
from gpaw.response.bse import BSE

# ==========================================================
# 1. Calculation Parameters
# ==========================================================
# From your G0W0 script output:
SCISSORS_SHIFT = 1.856  # eV (G0W0 Gap - PBE Gap)

# Grid for optical frequencies (eV)
omega = np.linspace(0.0, 6.0, 300)   # photon energy of the incoming light in eV
c_ang_per_fs = 2997.92458  # Speed of light in A/fs (useful for absorbance)

print("1. Loading ground-state wavefunctions...")
calc = GPAW('structures/HfS2_relaxed.gpw')
L_z = calc.atoms.cell[2, 2]  # Z-axis length including vacuum padding

# ==========================================================
# 2. Random Phase Approximation (RPA) for High Energies
# ==========================================================
print("2. Calculating RPA dielectric function (High-Energy Transitions)...")
df = DielectricFunction(
    calc='structures/HfS2_relaxed.gpw',
    txt='data/HfS2_RPA_log_12_refractive_index.txt',
    frequencies=omega,
    intraband=False,
    ecut=50,
    eta=0.1,              # Broadening
    truncation='2D',      # CRITICAL: Truncate Coulomb interaction along Z
    hilbert=False
)
# Get frequency-dependent macroscopic polarizability
'''
alpha0_2d_rpa - Bare Polarizability: This describes how the electrons would mathematically respond to an external electromagnetic field 
if we pretend electron-electron repulsions do not exist. It is a theoretical baseline. The "external field" refers exactly to the 
oscillating electric field of the incoming light wave.
alpha_2d_rpa - Interacting Polarizability: This includes "local field effects" (Hartree screening). When an external light wave moves the 
electrons, those moving electrons generate their own internal electric fields that push back on each other. This variable represents the 
true, physically measurable optical response of the material.
The five columns of the .csv file represents - Photon Energy (eV): The frequency of the incoming light. The steps (e.g., 0.000000, 
0.020067) correspond to the np.linspace(0.0, 6.0, 300), Real Bare Polarizability, Imaginary Bare Polarizability, Real Interacting 
Polarizability, Imaginary Interacting Polarizability. '''
alpha0_2d_rpa, alpha_2d_rpa = df.get_polarizability(
    direction='x',
    filename='data/HfS2_RPA_polarizability_12_refractive_index.csv'
)
print("RPA raw return shapes:",
      [np.shape(x) for x in (alpha0_2d_rpa, alpha_2d_rpa)],
      flush=True)
# ==========================================================
# 3. Bethe-Salpeter Equation (BSE) for Low Energies (Excitons)
# ==========================================================
print("3. Calculating BSE dielectric function (Low-Energy Excitons)...")
# We select 4 valence bands and 4 conduction bands for the electron-hole basis
# HfS2 VBM is 11, CBM is 12 (0-indexed)
bse = BSE(
    calc='structures/HfS2_relaxed.gpw',
    txt='data/HfS2_BSE_log_12_refractive_index.txt',
    ecut=50,
    valence_bands=[8, 9, 10, 11],
    conduction_bands=[12, 13, 14, 15],
    eshift=SCISSORS_SHIFT,
    truncation='2D'
)
# Calculate the polarizability on the requested frequency grid
bse_results = bse.get_polarizability(
    w_w=omega,
    eta=0.1,
    filename='data/HfS2_BSE_polarizability_12_refractive_index.csv'
)

print("BSE raw shape:", np.shape(bse_results), flush=True)

# Select the in-plane polarizability spectrum
alpha_2d_bse = bse_results[1]
print("BSE in-plane shape:", alpha_2d_bse.shape, flush=True)
print("BSE return shape:", np.shape(alpha_2d_bse), flush=True)
# ==========================================================
# 4. The BSE+ Stitching (Combine BSE and RPA)
# ==========================================================
# Use BSE for energies below 4.0 eV (excitonic regime) and RPA above 4.0 eV
'''RPA uses all 250 available bands. BSE is restricted to only the top 4 valence bands and bottom 4 conduction bands, but it still 
evaluates how those specific electrons respond across the entire 0 to 6 eV omega grid. 
From 0.0 to 4.0 eV: The script completely discards the RPA data. It relies only on the BSE data to accurately plot the low-energy excitons.
From 4.0 to 6.0 eV: The script completely discards the BSE data. It relies only on the RPA data to accurately plot the high-energy 
continuum transitions involving all 250 bands. '''
print("4. Stitching BSE and RPA spectra (BSE+ method)...")
stitch_energy = 4.0
alpha_2d_bse_plus = np.where(omega < stitch_energy, alpha_2d_bse, alpha_2d_rpa)

# ==========================================================
# 5. Extract 2D Absorbance and Effective Refractive Index
# ==========================================================
print("5. Deriving 2D optical observables...")

# 5a. Optical Absorbance of the 2D sheet (Dimensionless, e.g., ~0.02 - 0.15)
# A(w) = (4 * pi * w / c) * Im[alpha_2D(w)]
# Conversion factor: omega in eV to 1/fs -> omega / 0.6582
'''absorbance_2d is an array of 300 different numbers that form a continuous spectrum. Because your omega grid contains 300 distinct 
photon energies (from 0 to 6 eV), absorbance_2d calculates exactly how much light is absorbed at each of those 300 steps to build a 
complete curve. 

The Imaginary Part: Absorption (Inelastic Response)->
When light hits the material, some of the light's energy is permanently "eaten" to kick electrons from the valence band into the 
conduction band. The imaginary part of the polarizability mathematically represents this energy loss (dissipation). When you calculate 
absorbance_2d using the imaginary part, you are measuring the physical fraction of incoming light intensity that is destroyed and 
converted into excited electrons or heat at every specific frequency.

The Real Part: Refraction (Elastic Response)->
The real part represents the "springiness" of the electrons. When the electric field of the light pushes on the electrons, they 
momentarily stretch out of position and spring back without permanently absorbing the photon's energy.This interaction forces the light 
wave to interact with the material, effectively slowing the light down as it travels through. Therefore, the real part dictates the 
Refractive Index (n). It measures how much the material bends and slows down light (dispersion) without destroying the light wave.
'''
omega_fs = omega / 0.6582
absorbance_2d = (4 * np.pi * omega_fs / c_ang_per_fs) * alpha_2d_bse_plus.imag

# 5b. Effective Refractive Index of the purely 2D monolayer
# We define an intrinsic monolayer thickness t_2D (S-S distance + 2 * S vdW radius)
# S-S height ~2.87 Å, S vdW radius ~1.8 Å. Total native thickness ~6.47 Å.
t_2D = 6.47  # Å

''' Construct effective slab dielectric function. The 1.0 represents the baseline dielectric constant of a perfect, empty vacuum.
 alpha_2d_bse_plus/t_2D is the material's volume electric susceptibility. epf_eff is the dielectric function of the 3D material which 
 we made by dividing by t_2D '''
eps_eff = 1.0 + (4 * np.pi / t_2D) * alpha_2d_bse_plus

# Complex refractive index n + ik = sqrt(eps_eff)
n_complex = np.sqrt(eps_eff)
print("omega:", omega.shape, flush=True)
print("alpha_2d_rpa:", np.shape(alpha_2d_rpa), flush=True)
print("alpha_2d_bse:", np.shape(alpha_2d_bse), flush=True)
print("alpha_2d_bse_plus:", np.shape(alpha_2d_bse_plus), flush=True)
print("eps_eff:", np.shape(eps_eff), flush=True)

''' The real refractive index(n), which governs how light bends and slows down, and the imaginary extinction coefficient (k), which 
dictates how light is absorbed. '''
n_index = n_complex.real
k_extinction = n_complex.imag

# Convert energy to wavelength (nm) for plotting (ignoring omega=0)
valid = omega > 0.5
wavelength_nm = 1239.8 / omega[valid]
n_plot = n_index[valid]
k_plot = k_extinction[valid]

# ==========================================================
# 5c. Save the Numerical Data to Disk
# ==========================================================
print("Saving numerical data for future use...")

# Save the raw arrays in a compressed NumPy format (.npz)
# This allows you to load them later with np.load() without re-running GPAW
np.savez('data/HfS2_2D_Optical_Arrays_12_refractive_index.npz',
         omega=omega,
         wavelength_nm=wavelength_nm,
         alpha_2d_rpa=alpha_2d_rpa,
         alpha_2d_bse=alpha_2d_bse,
         alpha_2d_bse_plus=alpha_2d_bse_plus,
         n_index=n_index,
         k_extinction=k_extinction,
         absorbance_2d=absorbance_2d)

print("Saved raw arrays to 'data/HfS2_2D_Optical_Arrays_12_refractive_index.npz'")

# Save a clean, human-readable text file (CSV format) for Origin/Excel/Matlab
with open('data/HfS2_2D_Refractive_Index_12_refractive_index.csv', 'w') as f:
    f.write("Energy(eV),Wavelength(nm),n(In-plane),k(In-plane),Absorbance_2D\n")
    # We use zip to iterate through the arrays simultaneously, skipping the omega=0 point
    for e, wl, n, k, a in zip(omega[valid], wavelength_nm, n_plot, k_plot, absorbance_2d[valid]):
        f.write(f"{e:.4f},{wl:.2f},{n:.4f},{k:.4f},{a:.4f}\n")

print("Saved CSV spreadsheet to 'data/HfS2_2D_Refractive_Index_12_refractive_index.csv'")

# ==========================================================
# 6. Plotting the Results
# ==========================================================
fig, ax1 = plt.subplots(figsize=(7, 5))

# Plot n and k
color1 = 'tab:red'
ax1.set_xlabel('Wavelength (nm)', fontsize=12)
ax1.set_ylabel('Refractive Index, $n$', color=color1, fontsize=12)
ax1.plot(wavelength_nm, n_plot, color=color1, lw=2.5, label='n (In-plane)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xlim(300, 1000)

ax2 = ax1.twinx()  
color2 = 'tab:blue'
ax2.set_ylabel('Extinction Coefficient, $k$', color=color2, fontsize=12)
ax2.plot(wavelength_nm, k_plot, color=color2, lw=2.5, linestyle='--', label='k (In-plane)')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('BSE+ In-Plane Refractive Index of 2D 1T-HfS$_2$\n(Effective Slab Thickness = 6.47 Å)', pad=15)
fig.tight_layout()
plt.savefig('structures/HfS2_2D_Refractive_Index.png', dpi=300)
print("Saved purely 2D refractive index plot to 'structures/HfS2_2D_Refractive_Index.png'")