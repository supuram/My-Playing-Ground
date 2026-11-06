import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW

# 1. Load the converged ground state
calc = GPAW('structures/HfS2_relaxed.gpw')

# 2. Extract band structure parameters
weights = calc.get_k_point_weights()
nkpts = len(weights)  # Directly count the number of k-points from the weights array
eps_test = calc.get_eigenvalues(kpt=0)
nbands = len(eps_test)
n_electrons = calc.get_number_of_electrons()
n_occ = int(n_electrons / 2)  # Assuming a non-magnetic, spin-paired calculation

weights = calc.get_k_point_weights()

# 3. Setup the energy grid and broadening
energy_grid = np.linspace(0, 6, 500)
jdos = np.zeros_like(energy_grid)
sigma = 0.05  # 50 meV Gaussian broadening to match your optical spectra

# 4. Calculate JDOS manually from the eigenvalues
print("Calculating Joint Density of States...")
for k in range(nkpts):
    eps = calc.get_eigenvalues(kpt=k)
    weight = weights[k]
    
    # Loop over all valence (v) and conduction (c) bands
    for v in range(n_occ):
        for c in range(n_occ, nbands):
            delta_E = eps[c] - eps[v]
            
            # Add Gaussian-broadened transition to the grid
            gaussian = np.exp(-0.5 * ((energy_grid - delta_E) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
            jdos += weight * gaussian

# 5. Apply the GW Scissors Shift
# Because eigenvalues are from PBE, the transitions are too low in energy.
# We must apply the exact same 1.856 eV shift to align it with your optical plots.
scissors_shift = 1.856
shifted_energy_grid = energy_grid + scissors_shift

# Normalize JDOS for plotting alongside optical constants
jdos_normalized = jdos / np.max(jdos)

# 6. Plotting JDOS vs Energy
plt.figure(figsize=(8, 5))
plt.plot(shifted_energy_grid, jdos_normalized, 'g-', linewidth=2, label='JDOS (PBE + Scissors)')
plt.fill_between(shifted_energy_grid, jdos_normalized, color='green', alpha=0.2)

plt.xlim(1.5, 6.0)
plt.xlabel('Photon Energy (eV)', fontsize=12)
plt.ylabel('Normalized JDOS (Arb. Units)', fontsize=12)
plt.title('Joint Density of States for 1T-HfS$_2$', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig('structures/12_HfS2_JDOS.png', dpi=300)
print("Saved JDOS plot.")