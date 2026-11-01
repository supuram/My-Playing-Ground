import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
data = np.load('data/HfS2_Odoped_HR_spectrum.npz')
energies_meV = data['energies_meV']
S_k = data['S_k']

E_abs = 1.3109        # eV
E_rel = 0.06208       # eV
E_zpl = E_abs - E_rel # 1.2488 eV

# 2. Time-domain grid for Fourier transform
t_max = 5000.0  # fs
N_t = 2**15
t = np.linspace(-t_max, t_max, N_t)  # fs
hbar = 658.2119514  # meV * fs

# Filter non-physical acoustic drift modes
mask = energies_meV > 1.0
w_k = energies_meV[mask]
s_k = S_k[mask]

# 3. Generating function g(t) at T = 0 K
# g(t) = sum_k S_k * (exp(-i * w_k * t) - 1)
phase = np.outer(w_k / hbar, t)
g_t = np.sum(s_k[:, np.newaxis] * (np.exp(-1j * phase) - 1.0), axis=0)

# Apply Gaussian broadening (sigma ~ 5 meV)
sigma = 5.0  # meV
damping = np.exp(-0.5 * (sigma * t / hbar)**2)

# 4. Fourier Transform to obtain normalized lineshape
dt = t[1] - t[0]
spectrum = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(np.exp(g_t) * damping))) * dt
energies_relative = np.fft.fftshift(np.fft.fftfreq(N_t, d=dt)) * 2.0 * np.pi * hbar  # meV

# Shift to absolute photon energy (eV)
E_photon = E_zpl + (energies_relative / 1000.0)

# Normalize peak intensity
spectrum_real = np.real(spectrum)
spectrum_norm = spectrum_real / np.max(spectrum_real)

# 5. Crop to relevant optical window
window = (E_photon >= 1.05) & (E_photon <= 1.30)
E_plot = E_photon[window]
I_plot = spectrum_norm[window]

# Plot
plt.figure(figsize=(8, 5), dpi=300)
plt.plot(E_plot, I_plot, color='crimson', lw=2, label='PL Emission Spectrum (0 K)')
plt.axvline(E_zpl, color='navy', linestyle='--', label=f'ZPL = {E_zpl:.4f} eV')
plt.axvline(E_zpl - E_rel, color='gray', linestyle=':', label=f'Peak Max = {E_zpl - E_rel:.4f} eV')

plt.title(r'O-doped $\mathrm{HfS}_2$ Defect Emission ($S = 2.501$)', fontsize=12)
plt.xlabel('Photon Energy (eV)', fontsize=11)
plt.ylabel('Normalized Intensity (arb. units)', fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig('data/HfS2_Odoped_PL_spectrum.png')
print(f"PL spectrum successfully saved to data/HfS2_Odoped_PL_spectrum.png")