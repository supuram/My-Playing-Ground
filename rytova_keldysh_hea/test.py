import numpy as np
from ase.io import read
from gpaw import GPAW, PW, FermiDirac
from gpaw.mom import prepare_mom_calculation
from ase.optimize import BFGS
# 1. Load relaxed ground-state structure (.traj)
atoms = read('structures/HfS2_Odoped_relaxed.traj')
R_GS = atoms.get_positions()
print(dir(atoms))

# 2. Run ground state to converge the reference orbitals
calc = GPAW(
    mode=PW(500),
    xc='PBE',
    nbands=160,
    kpts={'size': (3, 3, 1), 'gamma': True},
    occupations=FermiDirac(0.01),
    spinpol=True,
    symmetry='off',
    txt='data/HfS2_Odoped_GS.txt'
)
atoms.calc = calc
print("Converging base spin-polarized state...")

# 3. Bypass the bug by dynamically calculating the HOMO/LUMO indices
n_electrons = int(calc.get_number_of_electrons())
print(n_electrons)
n_occupied = n_electrons // 2  # Number of fully occupied bands per spin
homo_idx = n_occupied - 1      # Index of HOMO
lumo_idx = n_occupied          # Index of LUMO

print(f"Total electrons: {n_electrons}. HOMO index: {homo_idx}, LUMO index: {lumo_idx}")

# Manually construct the exact MOM reference occupation list (nspins, nbands)
nspins = 2
nbands = 160
numbers = np.zeros((nspins, nbands))

# Fill all ground-state occupations (1.0 for bands below HOMO)
numbers[:, :n_occupied] = 1.0  

# Apply the vertical excitation in the spin-up channel (s=0)
numbers[0, homo_idx] = 0.0  # Create hole at HOMO
numbers[0, lumo_idx] = 1.0  # Populate LUMO

# MOM requires a native Python list of lists, not a numpy array
numbers = numbers.tolist()

# 4. Prepare MOM using the verified helper function from mom.py
print("Preparing MOM calculation...")
prepare_mom_calculation(calc, atoms, numbers)

# 5. Relax the Excited-State Geometry
print("Starting MOM Excited State Geometry Relaxation...")
opt = BFGS(atoms, trajectory='structures/HfS2_Odoped_ES_relaxed.traj', logfile='data/HfS2_Odoped_ES_BFGS.log')
opt.run(fmax=0.02)

# 6. Extract and save the displacement vector
R_ES = atoms.get_positions()
Delta_R = R_ES - R_GS
np.save('data/HfS2_Odoped_Delta_R.npy', Delta_R)

print(f"Excited state relaxation complete. Max displacement: {np.max(np.abs(Delta_R)):.4f} Å")
print("Saved displacement tensor to data/HfS2_Odoped_Delta_R.npy")