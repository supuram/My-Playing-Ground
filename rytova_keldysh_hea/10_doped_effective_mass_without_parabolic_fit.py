import numpy as np
from gpaw import GPAW

# Physical constant: hbar^2 / m0 in eV * Angstrom^2
HBAR2_M0 = 7.619964

print("Loading doped wavefunctions...")
calc = GPAW('structures/HfS2_Odoped_relaxed.gpw')
atoms = calc.get_atoms()

# Extract reciprocal lattice vectors to convert Cartesian k-points to fractional
cell_cv = atoms.get_cell()
icell_cv = np.linalg.inv(cell_cv).T * 2 * np.pi 

# Define a very tight finite-difference step (dk = 0.005 Å^-1) along the x-direction
dk = 0.005 
dk_cartesian = np.array([dk, 0.0, 0.0])
dk_frac = np.dot(dk_cartesian, np.linalg.inv(icell_cv))

# We evaluate strictly at Gamma-dk, Gamma, and Gamma+dk
kpts = [
    -dk_frac,
    [0.0, 0.0, 0.0],
    dk_frac
]

print("Evaluating fixed-density Hamiltonian around Gamma...")
calc = calc.fixed_density(kpts=kpts, symmetry='off', txt=None)
bs = calc.band_structure()
energies = bs.energies[0]  # Spin 0

# Band indices for the 3x3 O-doped supercell (216 valence electrons)
vbm_band = 107
cbm_band = 108

# Extract energies for the finite difference equation
E_minus = energies[0]
E_0 = energies[1]
E_plus = energies[2]

# Central finite difference for the second derivative (d^2E/dk^2)
d2E_dk2_vbm = (E_plus[vbm_band] - 2 * E_0[vbm_band] + E_minus[vbm_band]) / (dk**2)
d2E_dk2_cbm = (E_plus[cbm_band] - 2 * E_0[cbm_band] + E_minus[cbm_band]) / (dk**2)

# m* = (hbar^2) / (d^2E/dk^2)
m_hole = HBAR2_M0 / d2E_dk2_vbm
m_elec = HBAR2_M0 / d2E_dk2_cbm

print("\n--- EXACT FINITE DIFFERENCE MASS (Units of m0) ---")
print(f"Hole Effective Mass     : {m_hole:.3f}")
print(f"Electron Effective Mass : {m_elec:.3f}")