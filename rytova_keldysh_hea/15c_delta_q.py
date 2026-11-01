'''Check out this paper 
Accelerating point defect photo-emission calculations with machine learning interatomic potentials, 2025
Sharma, Kartikeya; Loew, Antoine; Wang, Haiyuan; Nilsson, Fredrik A.; Jain, Manjari; Marques, Miguel A.L.; Thygesen, Kristian S.'''

import numpy as np
from ase.io import read

# 1. Load relaxed structure to get atomic masses
atoms = read('structures/HfS2_Odoped_relaxed.traj')
masses = atoms.get_masses()  # Array of shape (N_atoms,) in amu

# 2. Load the displacement vector Delta R (shape: N_atoms, 3)
Delta_R = np.load('data/HfS2_Odoped_Delta_R.npy')

# 3. Calculate mass-weighted displacement vector Delta Q (Eq. 2)
# Multiply along atom axis: sqrt(M_alpha) * Delta_R_{alpha, i}
Delta_Q = np.sqrt(masses)[:, np.newaxis] * Delta_R    # Delta_Q has shape of (27, 3)

# 4. Total mass-weighted displacement Delta Q
Delta_Q_tot = np.linalg.norm(Delta_Q)

print(f"Total mass-weighted displacement (Delta Q): {Delta_Q_tot:.4f} amu^(1/2) Å")
np.save('data/HfS2_Odoped_Delta_Q.npy', Delta_Q)

'''Output = Total mass-weighted displacement (Delta Q): 1.1139 amu^(1/2) Å'''