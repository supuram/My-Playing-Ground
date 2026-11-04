import numpy as np
import matplotlib.pyplot as plt

# 1. Load the landscape data
data = np.load('data/HfS2_Sliding_Landscape_11_bilayer.npz')
shifts = data['shifts']
energies = data['energies']
dipoles_z = data['dipoles_z']

# 2. Convert total energy to relative energy (meV)
# Subtract the lowest resting energy so the barrier starts at 0 meV
relative_energy_mev = (energies - np.min(energies)) * 1000

# 3. Initialize dual-axis plot
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot Energy Landscape on left axis
color1 = 'tab:red'
ax1.set_xlabel('Fractional Shift along a-axis', fontsize=12)
ax1.set_ylabel('Relative Energy $\Delta E$ (meV)', color=color1, fontsize=12)
ax1.plot(shifts, relative_energy_mev, marker='o', color=color1, linewidth=2.5, markersize=8, label='Energy Barrier')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot Out-of-Plane Polarization on right axis
ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('Out-of-Plane Dipole, $P_z$ (e·Å)', color=color2, fontsize=12)
ax2.plot(shifts, dipoles_z, marker='s', linestyle='--', color=color2, linewidth=2.5, markersize=8, label='Z-Dipole')
ax2.tick_params(axis='y', labelcolor=color2)

# Title and final formatting
plt.title('Sliding Energy Landscape & Polarization of 1T-HfS$_2$ Bilayer', pad=15)
fig.tight_layout()

# Save to disk
plt.savefig('structures/HfS2_11_bilayer_sliding_energy_and_polarization_Plot.png', dpi=300)
print("Saved plot to 'structures/HfS2_11_bilayer_sliding_energy_and_polarization_Plot.png'")