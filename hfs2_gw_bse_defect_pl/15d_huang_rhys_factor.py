# Run this in your mattersim environment, not in GPAW

import numpy as np
from ase.io import read
from ase.vibrations import Vibrations
from mattersim.forcefield import MatterSimCalculator

# 1. Load relaxed ground-state structure
atoms = read('structures/HfS2_Odoped_relaxed.traj')

# 2. Attach MatterSim configured for CPU execution
calc = MatterSimCalculator(device="cpu")
atoms.calc = calc

# 3. Compute the 81 vibrational modes (162 finite displacements)
# name specifies the directory storing displacement pickle files
vib = Vibrations(atoms, name='data/vib_mattersim_cpu')   # vib is <ase.vibrations.vibrations.Vibrations object at 0x71cd8d91ca90>
vib.run()
vib.summary()
# To see the Hessian Matrix just do vib.H

# 4. Extract frequencies and eigenvectors
energies_meV = vib.get_energies() * 1000.0  # shape: (81,) in meV, These are quantized vibrational energies for each of the 81 normal vibrational modes(phonons at the Gamma-point) of my 27-atom supercell.
Delta_Q = np.load('data/HfS2_Odoped_Delta_Q.npy')  # shape: (27, 3) in amu^(1/2) Å

Q_k = []
S_k = []
pref = 0.11961267  # Unit conversion: 1 / (meV * amu * Å^2)

for k in range(len(energies_meV)):
    e_k = vib.get_mode(k)  # Cartesian displacement eigenvector (27, 3).
    ''' e_k is the eigenvector of each mode which has the del_x, del_y, del_z of all the atoms in the supercell. It is a single snapshot 
    of collective atomic motion. When mode k vibrates, all 27 atoms move simultaneously along their respective (del_x, del_y, del_z) 
    vectors, oscillating back and forth at the exact same frequency omega_k.
    The 3 columns in e_k represent the relative displacement vector(del_x, del_y, del_z) of that specific atom during that vibration. 
    It is normalized in mass-weighted space, meaning it represents the direction and relative amplitude of atomic motion, not an absolute 
    distance in angstroms. 
    The 81 modes do not belong to individual atoms; they belong to the entire coupled crystal, and each mode is an independent vibrational 
    highway through which phonons can be born, propagate, and dissipate energy.'''    
    # Ensure eigenvector is unit normalized
    norm = np.linalg.norm(e_k)  # calculates the magnitude or "length" of a vector or matrix
    if norm > 1e-8:
        e_k = e_k / norm
    
    # Eq. (3): Project mass-weighted displacement onto mode k
    q = np.sum(e_k * Delta_Q)
    ''' q measures how much the 3D displacement vector of each atom during defect relaxation Delta_Q aligns with the 3D displacement vector that 
    atom executes during vibration mode k which is e_k. '''
    Q_k.append(q)
    
    # Eq. (4): Partial HR factor (filter acoustic / translation modes near 0 meV)
    if energies_meV[k].real > 0.5:
        s = pref * energies_meV[k].real * (q**2)
        print(s, k)
    else:
        s = 0.0
    S_k.append(s)

Q_k = np.array(Q_k)   # (81, )
S_k = np.array(S_k)   # (81, )
S_total = np.sum(S_k)

# Verification checks
print(f"\n--- Phonon Projection Results ---")
print(f"Total Huang-Rhys factor S: {S_total:.4f}")
print(f"Sum of Q_k^2:             {np.sum(Q_k**2):.4f} amu Å^2")
print(f"Total |Delta Q|^2:         {np.linalg.norm(Delta_Q)**2:.4f} amu Å^2")

# Save mode-resolved spectrum for spectral density plotting
np.savez(
    'data/HfS2_Odoped_HR_spectrum.npz',
    energies_meV=energies_meV.real,
    Q_k=Q_k,
    S_k=S_k,
    S_total=S_total
)
print("Saved spectrum data to data/HfS2_Odoped_HR_spectrum.npz")

'''Output
Zero-point energy: 1.029 eV

--- Phonon Projection Results ---
Total Huang-Rhys factor S: 2.5011. This is the average number of phonons created during the emission process. This means that upon 
emitting a photon, the system generates an average of 2.5 phonons distributed across the 81 modes.
Sum of Q_k^2:             1.0324 amu Å^2
Total |Delta Q|^2:         1.2407 amu Å^2
'''