import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

E_PBE = 2.066           # Uncorrected PBE Gap
E_GW = 3.922            # Corrected G0W0 Quasiparticle Gap
soc_split = 0.1431      # generated from 9_spin_orbit.py
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

# --- Spectrum B: The True G0W0 + BSE + SOC Spectrum ---
# 1. The GW Continuums
# We split the amplitude to 0.5 each because the original degenerate band is now two separate bands
continuum_A = step_function(omega, E_GW, sigma, amplitude=0.5)
continuum_B = step_function(omega, E_GW + soc_split, sigma, amplitude=0.5)
total_continuum = continuum_A + continuum_B

true_spectrum = np.copy(total_continuum)

# 2. Add the Excitonic Peaks for BOTH A and B series
for n, eb in enumerate(eb_states):
    # Oscillator strength is halved because the probability is split between two channels
    oscillator_strength = 0.4 / (n + 0.5)**3 
    
    # A Exciton (From Upper Hole)
    peak_energy_A = E_GW - eb
    true_spectrum += gaussian_peak(omega, peak_energy_A, sigma/2, oscillator_strength)
    
    # B Exciton (From Lower Hole)
    peak_energy_B = (E_GW + soc_split) - eb
    true_spectrum += gaussian_peak(omega, peak_energy_B, sigma/2, oscillator_strength)

# ==========================================================
# 3. Plotting the Spectra
# ==========================================================
plt.figure(figsize=(10, 5))

plt.plot(omega, pbe_spectrum, 'k--', lw=2, label=f'Naive PBE (Gap = {E_PBE} eV)', alpha=0.6)
plt.plot(omega, true_spectrum, 'r-', lw=2.5, label='True Spectrum (GW + Excitons + SOC)')

plt.fill_between(omega, total_continuum, true_spectrum, color='red', alpha=0.3)

# Annotations for A and B Excitons
plt.axvline(E_GW, color='gray', linestyle=':', lw=1.5)
plt.axvline(E_GW + soc_split, color='gray', linestyle=':', lw=1.5)

plt.text(E_GW - eb_states[0] - 0.25, 0.8, 'A-Exciton', color='darkred', fontsize=11)
plt.text(E_GW + soc_split - eb_states[0] + 0.05, 0.8, 'B-Exciton', color='darkred', fontsize=11)

# Formatting
plt.title('Optical Absorption Spectrum of 1T-HfS$_2$\n(Including 143 meV Spin-Orbit Splitting)', fontsize=14, pad=15)
plt.xlabel('Photon Energy (eV)', fontsize=12)
plt.ylabel('Arbitrary Absorbance (a.u.)', fontsize=12)
plt.xlim(1.5, 4.5)
plt.ylim(0, max(true_spectrum) * 1.15)
plt.yticks([]) 
plt.legend(loc='upper left', fontsize=11, frameon=False)
plt.tight_layout()

plt.savefig('structures/HfS2_Optical_Spectrum_SOC.png', dpi=300)
print("Saved macroscopic optical spectrum to 'structures/HfS2_Optical_Spectrum_SOC.png'")