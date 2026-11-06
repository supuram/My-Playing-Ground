import numpy as np
import matplotlib.pyplot as plt

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

# 3. Calculate True Macroscopic Reflectivity
R_stitched = ((n_stitched - 1)**2 + k_stitched**2) / ((n_stitched + 1)**2 + k_stitched**2)

# 5. Initialize multi-panel plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# Top Panel: Refractive Index & Extinction Coefficient
ax1.plot(energy, n_stitched, label='Refractive Index ($n$)', color='tab:blue', linewidth=2.5)
ax1.plot(energy, k_stitched, label='Extinction Coefficient ($k$)', color='tab:orange', linewidth=2.5)
ax1.set_ylabel('Optical Constants', fontsize=12)
ax1.set_title('Optical Properties of 1T-HfS$_2$ from RPA + BSE', fontsize=14)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.6)

# Bottom Panel: Reflectivity
ax2.plot(energy, R_stitched * 100, label='Reflectivity ($R$)', color='tab:red', linewidth=2.5)
ax2.set_xlabel('Photon Energy (eV)', fontsize=12)
ax2.set_ylabel('Reflectivity (%)', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.6)

# Restrict x-axis to the visible/UV range where optical transitions matter
ax2.set_xlim(0, 6)

fig.tight_layout()
plt.savefig('structures/12_rpa_plus_bse.png', dpi=300)
print("Saved experimental observables plot to 'structures/12_rpa_plus_bse.png'")

'''
Why such a sharp cliff at 4 eV ?
To understand this, think of a material’s ability to absorb light as a fixed financial budget. In quantum mechanics, this budget is 
governed by the f-sum rule, which dictates that a material's total probability of absorbing light—when added up across all possible 
photon energies—is a strict mathematical constant determined by the number of electrons it possesses. You cannot magically create more 
total absorption; you can only rearrange which energies get to absorb it.In your RPA calculation, the electrons and holes ignore each 
other. The material spends its absorption budget normally, spreading it out across the standard high-energy interband transitions (the 
UV continuum).In your BSE calculation, the strong electrostatic attraction between the electron and the hole pulls them together into a 
bound 1s exciton. Because this excitonic state is highly favorable, it demands a massive portion of the material's absorption budget. To 
"pay" for that towering exciton peak at 2.6 eV, the BSE mathematics must extract absorption probability away from the higher-energy 
continuum transitions, physically suppressing them.This is exactly why your 4.0 eV stitch point resulted in a vertical cliff. You 
mathematically glued the tail end of the BSE continuum (which had its budget depleted to fund the exciton) directly onto the RPA 
continuum (which was still sitting at its uncorrected, full-budget baseline). The sudden mismatch in probability creates the artifact.

1)Van Hove Singularities (Parallel Bands) - In 12_HfS2_Optical_Constants.png file in structures folder
In 2D materials, the highest valence bands and lowest conduction bands often run perfectly parallel to each other across certain regions 
of the Brillouin zone. Because they are parallel, millions of electrons across the crystal all share the exact same energy gap. When a 
photon with that exact energy arrives, a massive, concentrated wave of electrons jumps simultaneously. This creates a singularity in the 
JDOS, resulting in those towering spikes between 2 and 3 eV.

2)Orbital Overlap
Optical absorption requires the starting and ending orbitals to overlap in physical space. In HfS2, the top valence bands (Sulfur 3p 
orbitals) and bottom conduction bands (Hafnium 5d orbitals) are strongly localized and perfectly symmetric for light coupling. The 
transition probability (oscillator strength) is incredibly high.

3) Why the 4–6 eV Range is Weaker
At higher energies, the incoming light is digging into deep, low-lying valence bands and pushing electrons into highly excited conduction 
bands. These deep bands are usually highly dispersed (steep, not flat). Because the energy gap changes rapidly at every different k-point, 
the transitions are smeared out over a wide range of frequencies rather than stacking up into one giant peak. The physical spatial overlap 
between deep core-like orbitals and highly excited diffuse orbitals is poor, dropping the transition probability.

The RPA (Two Peaks): Independent Transitions - In 12_HfS2_Optical_Constants.png file in structures folder
In the RPA calculation, the material's absorption is determined purely by the independent-particle band structure. In 1T-HfS2, there are 
two distinct regions in the Brillouin zone (typically near the M and Gamma points) where the valence and conduction bands run perfectly 
parallel. The RPA treats these as two completely separate optical events, resulting in two distinct Van Hove singularity peaks between 2.5
and 3.2 eV.

The BSE+ (One Massive Peak): Exciton Mixing - In 12_rpa_plus_bse.png in structures folder
When you solve the Bethe-Salpeter Equation, the electron and hole are no longer independent; they are subjected to a strong, attractive 
Coulomb force. This attraction causes a phenomenon called k-space mixing (or transition mixing).
The attractive force is so strong that it physically couples those two separate parallel band regions together. Instead of two independent 
transitions, the electron-hole pairs form a single quantum superposition. The BSE mathematics takes the absorption probability (oscillator 
strength) from both of those separate RPA peaks, crushes them together, and funnels all that energy into one dominant, highly bound 1s 
excitonic state.
'''