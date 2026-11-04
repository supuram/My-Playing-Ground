from ase.io import read
from gpaw import GPAW, PW
from gpaw.spinorbit import soc_eigenstates

print("1. Loading pristine structure...")
atoms = read('structures/HfS2_relaxed.traj')

print("2. Running Scalar-Relativistic PBE calculation (Gamma point)...")
# CRITICAL: symmetry='off' must be set for the Spin-Orbit solver to work properly
calc = GPAW(
    mode=PW(600),
    xc='PBE',
    kpts=[(0.0, 0.0, 0.0)],  # We only need the Gamma point where the VBM lives
    symmetry='off',
    txt='data/HfS2_soc_scf.txt'
)
atoms.calc = calc
atoms.get_potential_energy()

# Extract scalar eigenvalues for a baseline comparison
e_scalar = calc.get_eigenvalues(kpt=0)   # (19,)
vbm_idx = 11  
cbm_idx = 12

print("\n--- SCALAR RELATIVISTIC BASELINE (No SOC) ---")
print(f"VBM   (Band {vbm_idx}): {e_scalar[vbm_idx]:.4f} eV")
print(f"VBM-1 (Band {vbm_idx-1}): {e_scalar[vbm_idx-1]:.4f} eV")
print(f"CBM   (Band {cbm_idx}): {e_scalar[cbm_idx]:.4f} eV")

print("\n3. Applying Spin-Orbit Coupling perturbation...")
# This calculates the L.S Hamiltonian and diagonalizes it
e_soc = soc_eigenstates(calc)    # (1, 38). soc_eigenstates mathematically forces the spin (S) to interact with the orbital angular momentum (L). It treats "spin-up" and "spin-down" as separate, interacting entities, it physically splits the degenerate bands.
e_km = e_soc.eigenvalues()       # (1, 38)

# e_soc is an array of shape (nkpts, 2*nbands). We only have 1 k-point.
e_soc_gamma = e_km[0]       # (38,)

# Because the total number of bands doubled, the location of your highest occupied electrons (the VBM) also shifted in the array index. 
soc_vbm_idx = (vbm_idx * 2) + 1

print("\n--- SPIN-ORBIT COUPLING RESULTS ---")
print("Top Valence Bands (The Split 'Holes'):")
for i in range(soc_vbm_idx - 3, soc_vbm_idx + 1):
    print(f"  SOC Band {i}: {e_soc_gamma[i]:.4f} eV")

upper_hole = e_soc_gamma[soc_vbm_idx]
lower_hole = e_soc_gamma[soc_vbm_idx - 2]
soc_splitting = upper_hole - lower_hole

print("\n" + "="*50)
print(f"Valence Band SOC Splitting: {soc_splitting * 1000:.1f} meV")
print("="*50)
print("This energy difference separates the A and B optical exciton peaks!")

'''
--- SCALAR RELATIVISTIC BASELINE (No SOC) ---
VBM   (Band 11): -3.3485 eV
VBM-1 (Band 10): -3.3485 eV
CBM   (Band 12): -1.5548 eV

3. Applying Spin-Orbit Coupling perturbation...

--- SPIN-ORBIT COUPLING RESULTS ---
Top Valence Bands (The Split 'Holes'):
  SOC Band 20: -3.4074 eV
  SOC Band 21: -3.4074 eV
  SOC Band 22: -3.2643 eV
  SOC Band 23: -3.2643 eV

==================================================
Valence Band SOC Splitting: 143.1 meV
==================================================
'''
