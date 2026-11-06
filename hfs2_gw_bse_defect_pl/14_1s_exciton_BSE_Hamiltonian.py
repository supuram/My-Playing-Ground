import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW
from gpaw.response.bse import BSE
from gpaw.mpi import world

print("1. Loading ground state...")
calc = GPAW('structures/HfS2_relaxed.gpw')
atoms = calc.get_atoms()

v_bands = [8, 9, 10, 11]
c_bands = [12, 13, 14, 15]

print("2. Initializing the BSE with 2D Truncation...")
bse = BSE('structures/HfS2_relaxed.gpw',
          truncation='2D',           
          integrate_gamma='reciprocal',  
          valence_bands=v_bands,
          conduction_bands=c_bands,
          ecut=50,                   
          eshift=1.856)

print("3. Extracting and Diagonalizing the BSE Hamiltonian...")
H_bse_obj = bse.get_bse_matrix()
np.savez_compressed('data/14_1s_exciton_BSE_Hamiltonian.npz', BSE_Hamiltonian_object=H_bse_obj)
print("SUCCESS: Full BSE matrix safely checkpointed to disk.")