import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

E_PBE = 2.066           # Uncorrected PBE Gap
E_GW = 3.922            # Corrected G0W0 Quasiparticle Gap
eb_states = [1.297, 0.530, 0.294, 0.172, 0.037]  # Binding energies
sigma = 0.05            # Broadening factor (simulating temperature/instrument resolution)

# Generate an array of photon energies (x-axis) from 1.0 eV to 5.0 eV
omega = np.linspace(1.0, 5.0, 2000)

'''
In gaussian peak, photon energy < G0W0 gap. The photon doesn't have quite enough fuel to reach the free conduction band. It creates an 
electron and a hole, but they are born trapped together, "orbiting" each other. This creates the sharp spikes. Starting point: Valence 
band. Ending point: Exciton state.
'''
# 2. Build the Spectra
def gaussian_peak(x, center, width, amplitude):
    return amplitude * np.exp(-0.5 * ((x - center) / width)**2)

'''
In step function, photon energy >= G0W0 gap. The photon creates an electron and a hole, and they immediately fly apart into "deep space" 
as free charges. This creates the continuous, flat plateau of absorption. Starting point: Valence band. Ending point: Conduction band.
'''
def step_function(x, onset, width, amplitude):
    # Models the 2D joint density of states (a step function)
    return amplitude * 0.5 * (1.0 + erf((x - onset) / (width * np.sqrt(2))))

# --- Spectrum A: The PBE "Illusion" ---
# PBE has no excitons, just a sudden onset of absorption at the band gap.
pbe_spectrum = step_function(omega, E_PBE, sigma, amplitude=1.0)

# --- Spectrum B: The True G0W0 + BSE Spectrum ---
# 1. The GW Continuum (starts at 3.922 eV)
gw_continuum = step_function(omega, E_GW, sigma, amplitude=1.0)

# 2. Add the Excitonic Peaks
true_spectrum = np.copy(gw_continuum)

# In 2D, the oscillator strength (peak height) of excitons scales roughly as 1 / (n - 0.5)^3
# We add each peak at the correct optical transition energy (E_GW - Eb)
for n, eb in enumerate(eb_states):
    peak_energy = E_GW - eb
    oscillator_strength = 0.8 / (n + 0.5)**3  # 2D scaling law for bright excitons
    
    true_spectrum += gaussian_peak(omega, peak_energy, sigma/2, oscillator_strength)

# ==========================================================
# 3. Plotting the Cancellation of Errors
# ==========================================================
plt.figure(figsize=(9, 5))

# Plot PBE (The naive approach)
plt.plot(omega, pbe_spectrum, 'k--', lw=2, label=f'Naive PBE (Gap = {E_PBE} eV)', alpha=0.6)

# Plot True Spectrum
plt.plot(omega, true_spectrum, 'r-', lw=2.5, label='True Spectrum (GW + Excitons)')

# Fill under the Exciton Peaks for visual impact
plt.fill_between(omega, gw_continuum, true_spectrum, color='red', alpha=0.3)

# Annotations
plt.axvline(E_GW, color='gray', linestyle=':', lw=1.5)
plt.text(E_GW + 0.05, 1.2, r'True GW Gap $\rightarrow$', color='gray', fontsize=11)

plt.axvline(E_GW - eb_states[0], color='darkred', linestyle=':', lw=1.5)
plt.text((E_GW - eb_states[0]) - 0.3, 1.2, r'$\leftarrow 1s$ Exciton', color='darkred', fontsize=11)

# Formatting
plt.title('Optical Absorption Spectrum of 1T-HfS$_2$\n(The Cancellation of Errors)', fontsize=14, pad=15)
plt.xlabel('Photon Energy (eV)', fontsize=12)
plt.ylabel('Arbitrary Absorbance (a.u.)', fontsize=12)
plt.xlim(1.5, 4.5)
plt.ylim(0, max(true_spectrum) * 1.1)
plt.yticks([]) # Hide y-ticks as they are arbitrary units
plt.legend(loc='upper left', fontsize=11, frameon=False)
plt.tight_layout()

# Save the figure
plt.savefig('structures/HfS2_Optical_Spectrum.png', dpi=300)
print("Saved macroscopic optical spectrum to 'structures/HfS2_Optical_Spectrum.png'")

# OAS = Optical Absorption Spectrum