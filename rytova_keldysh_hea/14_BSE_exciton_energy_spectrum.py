import numpy as np
import matplotlib.pyplot as plt

# 1. Load the data you spent 5 days calculating
data = np.load('data/14_1s_exciton_wavefunction.npz')
energies = data['energies']  # All 2304 exciton energies!

# 2. Extract the lowest 50 excitonic states
lowest_states = energies[:50]
E_1s = lowest_states[0]

# 3. Create an Energy Level Diagram
plt.figure(figsize=(4, 6))

# Plot the 1s state in red
plt.axhline(y=E_1s, color='red', linewidth=2.5, label='1s Exciton (Dominant)')

# Plot the higher excited states (2s, 2p, etc.) and continuum in blue
for i, E in enumerate(lowest_states[1:]):
    if i == 0:
        plt.axhline(y=E, color='blue', alpha=0.4, linewidth=1.5, label='Higher Excitons / Continuum')
    else:
        plt.axhline(y=E, color='blue', alpha=0.4, linewidth=1.5)

plt.ylabel('BSE Eigenvalues (eV)', fontsize=12)
plt.title('BSE Exciton Energy Spectrum\n(Rydberg Series)', fontsize=14)
plt.xticks([])  # Hide x-axis (energy levels are just horizontal lines)
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig('structures/14_BSE_Energy_Spectrum.png', dpi=300)
print(f"Saved spectrum. 1s Energy: {E_1s:.3f} eV, 2nd State Energy: {lowest_states[1]:.3f} eV")

''' 1s Energy: 0.100 eV, 2nd State Energy: 0.101 eV '''