import numpy as np

data = np.loadtxt('data/HfS2_RPA_polarizability_12_refractive_index.csv', delimiter=',')
energy = data[:, 0]
alpha_real = data[:, 3]
alpha_imag = data[:, 4]

data = np.loadtxt('data/HfS2_BSE_polarizability_12_refractive_index.csv', delimiter=',')
energy_bse = data[:, 0]
alpha_real_bse = data[:, 1]
alpha_imag_bse = data[:, 2]

d_eff = 6.47
scissors_shift = 1.856

# Shift the RPA features to higher energies (to match the GW gap)
# np.interp(target_x, original_x_shifted, original_y)
alpha_real_rpa_shifted = np.interp(energy, energy + scissors_shift, alpha_real, left=0)
alpha_imag_rpa_shifted = np.interp(energy, energy + scissors_shift, alpha_imag, left=0)

# 3. Convert to 3D Dielectric Functions (Using the SHIFTED RPA data)
eps_1_rpa = 1 + (4 * np.pi * alpha_real_rpa_shifted) / d_eff  
eps_2_rpa = (4 * np.pi * alpha_imag_rpa_shifted) / d_eff

eps_1_bse = 1 + (4 * np.pi * alpha_real_bse) / d_eff  # Real dielectric constant
eps_2_bse = (4 * np.pi * alpha_imag_bse) / d_eff      # Imaginary dielectric constant

# 1. Stitch the Dielectric Functions (BSE below 4.0 eV, RPA above 4.0 eV)
stitch_energy = 4.0

eps_1_stitched = np.where(energy < stitch_energy, eps_1_bse, eps_1_rpa)
eps_2_stitched = np.where(energy < stitch_energy, eps_2_bse, eps_2_rpa)

# 2. Calculate Optical Constants using the STITCHED data
eps_mag_stitched = np.sqrt(eps_1_stitched**2 + eps_2_stitched**2)
n_stitched = np.sqrt((eps_mag_stitched + eps_1_stitched) / 2)
k_stitched = np.sqrt((eps_mag_stitched - eps_1_stitched) / 2)

# --- 1. EVALUATE MIE RESONATOR FIGURE OF MERIT (FOM) ---
# We evaluate the FOM in the sub-gap transparent regime (e.g., at 1.5 eV)
# Find the index in the energy array closest to 1.5 eV
target_energy = 1.5
idx_15 = (np.abs(energy - target_energy)).argmin()

n_mie = n_stitched[idx_15]
k_mie = k_stitched[idx_15]

# Prevent division by zero if k is exactly 0
k_safe = max(k_mie, 1e-10) 
fom_mie = n_mie / k_safe

print("--- MIE RESONATOR METRICS (@ 1.5 eV) ---")
print(f"Refractive Index (n): {n_mie:.3f} (Ideal is > 2.0)")
print(f"Extinction (k): {k_mie:.5f} (Ideal is ~ 0.0)")
print(f"Mie Figure of Merit (n/k): {fom_mie:.1f}\n")

# --- 2. EVALUATE THE MOSS RULE ---
# Extract the static refractive index (n at 0 eV)
idx_0 = (np.abs(energy - 0.0)).argmin()
n_0 = n_stitched[idx_0]

# Define the two gaps found in your previous studies
E_gap_GW = 3.922      # Fundamental Quasiparticle Gap
E_gap_optical = 2.625 # Optical Exciton Gap

# Calculate Moss Constants
C_moss_GW = (n_0**4) * E_gap_GW
C_moss_optical = (n_0**4) * E_gap_optical
bulk_moss_constant = 95.0

print("--- MOSS RULE EVALUATION ---")
print(f"Static Refractive Index (n_0): {n_0:.3f}")
print(f"Calculated Moss Constant (using GW Gap): {C_moss_GW:.2f} eV")
print(f"Calculated Moss Constant (using Optical Gap): {C_moss_optical:.2f} eV")
print(f"Standard Bulk Moss Constant: {bulk_moss_constant} eV")

# Calculate the percentage deviation from the bulk law
dev_GW = ((C_moss_GW - bulk_moss_constant) / bulk_moss_constant) * 100
dev_optical = ((C_moss_optical - bulk_moss_constant) / bulk_moss_constant) * 100

print(f"Deviation from bulk rule (GW): {dev_GW:.1f}%")
print(f"Deviation from bulk rule (Optical): {dev_optical:.1f}%")