import numpy as np
import matplotlib.pyplot as plt

# 1. Load the RPA polarizability data
# We only need the interacting polarizability (columns 3 and 4)
data = np.loadtxt('data/HfS2_RPA_polarizability_12_refractive_index.csv', delimiter=',')
energy = data[:, 0]
alpha_real = data[:, 3]
alpha_imag = data[:, 4]

# 2. Convert 2D polarizability to Macroscopic Dielectric Function (epsilon)
# To compare with experiment, a 2D sheet is treated as having an effective thickness (d).
# The van der Waals thickness of a 1T-HfS2 monolayer is approximately 6.2 Angstroms.
d_eff = 6.47
eps_1 = 1 + (4 * np.pi * alpha_real) / d_eff  # Real dielectric constant
eps_2 = (4 * np.pi * alpha_imag) / d_eff      # Imaginary dielectric constant

# 3. Calculate Optical Constants (Refractive Index 'n' and Extinction Coefficient 'k')
eps_mag = np.sqrt(eps_1**2 + eps_2**2)
n = np.sqrt((eps_mag + eps_1) / 2)
k = np.sqrt((eps_mag - eps_1) / 2)

# 4. Calculate Optical Reflectivity (R) at normal incidence
'''This is the Fresnel equation for normal incidence. It calculates the Reflectivity (R)—the exact percentage of light that will bounce 
off the surface of the material when shining a laser directly at it (perpendicular to the surface).'''
R = ((n - 1)**2 + k**2) / ((n + 1)**2 + k**2)

# 5. Initialize multi-panel plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# Top Panel: Refractive Index & Extinction Coefficient
ax1.plot(energy, n, label='Refractive Index ($n$)', color='tab:blue', linewidth=2.5)
ax1.plot(energy, k, label='Extinction Coefficient ($k$)', color='tab:orange', linewidth=2.5)
ax1.set_ylabel('Optical Constants', fontsize=12)
ax1.set_title('Optical Properties of 1T-HfS$_2$ from RPA', fontsize=14)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.6)

# Bottom Panel: Reflectivity
ax2.plot(energy, R * 100, label='Reflectivity ($R$)', color='tab:red', linewidth=2.5)
ax2.set_xlabel('Photon Energy (eV)', fontsize=12)
ax2.set_ylabel('Reflectivity (%)', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.6)

# Restrict x-axis to the visible/UV range where optical transitions matter
ax2.set_xlim(0, 6)

fig.tight_layout()
plt.savefig('structures/12_HfS2_Optical_Constants.png', dpi=300)
print("Saved experimental observables plot to 'structures/12_HfS2_Optical_Constants.png'")