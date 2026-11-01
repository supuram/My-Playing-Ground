import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.vibrations import Vibrations

# 1. Load the Hybrid Vibrations
atoms = read('structures/HfS2_Odoped_relaxed.traj')
# By passing NO indices, ASE reads ALL .pckl files in the folder (combining GPAW + MatterSim)
vib = Vibrations(atoms, name='data/vib_hybrid')
vib.read()

energies_meV = vib.get_energies() * 1000.0  # in meV
Delta_Q = np.load('data/HfS2_Odoped_Delta_Q.npy')

# 2. Calculate Hybrid Huang-Rhys Factors
Q_k, S_k = [], []
pref = 0.11961267  # Unit conversion

for k in range(len(energies_meV)):
    e_k = vib.get_mode(k)
    norm = np.linalg.norm(e_k)
    if norm > 1e-8:
        e_k = e_k / norm
    
    q = np.sum(e_k * Delta_Q)
    Q_k.append(q)
    
    if energies_meV[k].real > 0.5:
        s = pref * energies_meV[k].real * (q**2)
    else:
        s = 0.0
    S_k.append(s)

Q_k = np.array(Q_k)
S_k = np.array(S_k)
S_total = np.sum(S_k)
E_rel = np.sum(S_k * energies_meV.real) / 1000.0  # eV

print(f"--- HYBRID PHONON RESULTS ---")
print(f"Total Huang-Rhys Factor S : {S_total:.4f}")
print(f"Relaxation Energy         : {E_rel*1000:.2f} meV")

# 3. Calculate Temperature-Dependent Photoluminescence (PL)
E_abs = 1.3109  # eV (From your GPAW output)
E_zpl = E_abs - E_rel

t_max = 5000.0  # fs
N_t = 2**15
t = np.linspace(-t_max, t_max, N_t)
hbar = 658.2119514  # meV * fs

mask = energies_meV.real > 1.0
w_k = energies_meV.real[mask]
s_k = S_k[mask]

# Define the generating function for an arbitrary temperature T
def calculate_spectrum(T_Kelvin, sigma_meV=5.0):
    kB = 0.086173  # meV/K
    
    if T_Kelvin == 0:
        n_k = np.zeros_like(w_k)
    else:
        n_k = 1.0 / (np.exp(w_k / (kB * T_Kelvin)) - 1.0) # Bose-Einstein distribution
        
    phase = np.outer(w_k / hbar, t)
    
    # Finite-T generating function g(t)
    g_t = np.sum(s_k[:, np.newaxis] * (
        (n_k + 1.0)[:, np.newaxis] * np.exp(-1j * phase) + 
        n_k[:, np.newaxis] * np.exp(1j * phase) - 
        (2 * n_k + 1.0)[:, np.newaxis]
    ), axis=0)
    
    damping = np.exp(-0.5 * (sigma_meV * t / hbar)**2)
    dt = t[1] - t[0]
    
    spectrum = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(np.exp(g_t) * damping))) * dt
    energies_relative = np.fft.fftshift(np.fft.fftfreq(N_t, d=dt)) * 2.0 * np.pi * hbar
    E_photon = E_zpl + (energies_relative / 1000.0)
    
    spectrum_real = np.real(spectrum)
    return E_photon, spectrum_real / np.max(spectrum_real)

# 4. Plot 0 K vs 300 K Spectra
E_plot_0K, I_plot_0K = calculate_spectrum(T_Kelvin=0)
E_plot_300K, I_plot_300K = calculate_spectrum(T_Kelvin=300)

plt.figure(figsize=(9, 5), dpi=300)
plt.plot(E_plot_0K, I_plot_0K, color='navy', lw=1.5, label='T = 0 K')
plt.plot(E_plot_300K, I_plot_300K, color='crimson', lw=2, alpha=0.8, label='T = 300 K (Room Temp)')

plt.axvline(E_zpl, color='black', linestyle='--', alpha=0.6, label=f'ZPL = {E_zpl:.3f} eV')

plt.title(r'Hybrid DFT/MLIP Emission Spectrum of O-doped $\mathrm{HfS}_2$', fontsize=13)
plt.xlabel('Photon Energy (eV)', fontsize=12)
plt.ylabel('Normalized Intensity (arb. units)', fontsize=12)
plt.xlim(1.0, 1.35)
plt.grid(True, alpha=0.3)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig('data/HfS2_Odoped_Hybrid_PL_Spectrum.png')
print("Saved hybrid spectrum plot to data/HfS2_Odoped_Hybrid_PL_Spectrum.png")

''' Output
Total Huang-Rhys Factor S : 2.5041
Relaxation Energy         : 62.94 meV
'''
