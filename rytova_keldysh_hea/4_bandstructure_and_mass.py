import numpy as np
from ase.io import read
from gpaw import GPAW
import matplotlib.pyplot as plt

# 1. Load relaxed system and bandpath
atoms = read('structures/HfS2_relaxed.gpw')
lat = atoms.cell.get_bravais_lattice()
path = atoms.cell.bandpath('GMKG', npoints=100)

print("Calculating band structure along Gamma-M-K-Gamma...")
calc = GPAW('structures/HfS2_relaxed.gpw').fixed_density(
            kpts=path,
            symmetry='off',
            txt='structures/HfS2_bandpath.txt')
calc.get_potential_energy()

# Extract eigenvalues (eV)
# HfS2 has 24 valence electrons -> 12 occupied bands (0 to 11)
vbm_band = 11  # Top valence band
cbm_band = 12  # Bottom conduction band

e_kn = np.array([calc.get_eigenvalues(k) for k in range(len(path.kpts))])
e_vbm = np.max(e_kn[:, vbm_band])

# Shift VBM to 0 eV
e_kn -= e_vbm

e_cbm = np.min(e_kn[:, cbm_band])
indirect_gap = e_cbm
direct_gap = np.min(e_kn[:, cbm_band] - e_kn[:, vbm_band])

print(f"\n--- Band Gap Results ---")
print(f"VBM energy (set to 0): 0.00 eV")    # valence band maximum (VBM) and conduction band minimum (CBM)
print(f"CBM energy:           {e_cbm:.3f} eV")
print(f"Indirect Band Gap:    {indirect_gap:.3f} eV")
print(f"Direct Band Gap:      {direct_gap:.3f} eV")

# Plot Band Structure
fig, ax = plt.subplots(figsize=(6, 5))
for n in range(vbm_band - 3, cbm_band + 4):
    ax.plot(path.get_linear_kpoint_axis()[0], e_kn[:, n], color='black', lw=1.2)
ax.axhline(0, color='blue', ls='--', lw=0.8, label='VBM')
ax.axhline(e_cbm, color='red', ls='--', lw=0.8, label='CBM')
labels = [r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$']
ax.set_xticks(path.get_linear_kpoint_axis()[1])
ax.set_xticklabels(labels)
ax.set_ylabel("Energy (eV)")
ax.set_ylim(-3, 4)
ax.set_title("1T-HfS2 Monolayer Band Structure (PBE)")
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig("structures/HfS2_bandstructure.png", dpi=200)
print("Saved band structure plot to structures/HfS2_bandstructure.png")

# 2. Finite Difference for Effective Mass around Extrema
# m* / m0 = (hbar^2 / m0) / (d^2 E / dk^2)
# hbar^2 / m0 = 7.619964 eV * Å^2
HBAR2_OVER_M0 = 7.619964

# Small k-displacement dk along reciprocal vector direction
# We use Cartesian coordinates for accurate metric tensor scaling
b_recip = atoms.cell.reciprocal() * 2 * np.pi  # 1/Å units
k_gamma = np.array([0.0, 0.0, 0.0])

# Displace slightly along Gamma -> M direction
dk_frac = 0.005
k_pts = [
    k_gamma - np.array([dk_frac, 0, 0]),
    k_gamma,
    k_gamma + np.array([dk_frac, 0, 0])
]

calc_k = GPAW('structures/HfS2_relaxed.gpw').fixed_density(
              kpts=k_pts,
              symmetry='off',
              txt=None)
calc_k.get_potential_energy()

# Stencil for Hole Mass at Gamma (VBM)
dk_cart = np.linalg.norm(b_recip[0] * dk_frac)
E_minus = calc_k.get_eigenvalues(kpt=0)[vbm_band]
E_0     = calc_k.get_eigenvalues(kpt=1)[vbm_band]
E_plus  = calc_k.get_eigenvalues(kpt=2)[vbm_band]

d2E_dk2_hole = (E_plus - 2 * E_0 + E_minus) / (dk_cart**2)
# Valence band curves downward -> negative second derivative
m_h = - HBAR2_OVER_M0 / d2E_dk2_hole

# Stencil for Electron Mass (at Gamma for direct optical exciton)
E_c_minus = calc_k.get_eigenvalues(kpt=0)[cbm_band]
E_c_0     = calc_k.get_eigenvalues(kpt=1)[cbm_band]
E_c_plus  = calc_k.get_eigenvalues(kpt=2)[cbm_band]

d2E_dk2_elec = (E_c_plus - 2 * E_c_0 + E_c_minus) / (dk_cart**2)
m_e = HBAR2_OVER_M0 / d2E_dk2_elec

mu_eff = (m_e * m_h) / (m_e + m_h)

print(f"\n--- Effective Mass Results (Gamma-point) ---")
print(f"Hole effective mass (m_h*):     {m_h:.3f} m_0")
print(f"Electron effective mass (m_e*): {m_e:.3f} m_0")
print(f"Reduced exciton mass (mu_eff):  {mu_eff:.3f} m_0")

# Save values for the exciton solver
np.savez("data/HfS2_params.npz", r0_bohr=22.10, mu_eff=mu_eff, direct_gap=direct_gap)

''' Output with nbands = 36
Calculating band structure along Gamma-M-K-Gamma...

--- Band Gap Results ---
VBM energy (set to 0): 0.00 eV
CBM energy:           1.241 eV
Indirect Band Gap:    1.241 eV
Direct Band Gap:      2.066 eV
Saved band structure plot to structures/HfS2_bandstructure.png

--- Effective Mass Results (Gamma-point) ---
Hole effective mass (m_h*):     0.460 m_0
Electron effective mass (m_e*): 0.999 m_0
Reduced exciton mass (mu_eff):  0.315 m_0

Output with nbands = 250
Calculating band structure along Gamma-M-K-Gamma...

--- Band Gap Results ---
VBM energy (set to 0): 0.00 eV
CBM energy:           1.241 eV
Indirect Band Gap:    1.241 eV
Direct Band Gap:      2.066 eV
Saved band structure plot to structures/HfS2_bandstructure.png

--- Effective Mass Results (Gamma-point) ---
Hole effective mass (m_h*):     0.460 m_0
Electron effective mass (m_e*): 0.999 m_0
Reduced exciton mass (mu_eff):  0.315 m_0
'''
