import os
import numpy as np
from ase.io import read
from gpaw import GPAW
from ase.vibrations import Vibrations

# 1. Load the relaxed Gamma-point ground state
atoms = read('structures/HfS2_Odoped_relaxed.traj')
calc = GPAW('data/HfS2_Odoped_GS_FDPWETDM.gpw', txt='data/hybrid_gpaw_vib.txt')
atoms.calc = calc

# 2. Identify the Defect Core (O atom and neighbors within 3.0 Å)
o_idx = atoms.get_chemical_symbols().index('O')
distances = atoms.get_distances(o_idx, range(len(atoms)), mic=True)
core_indices = [i for i, d in enumerate(distances) if d <= 3.0]

print(f"Defect core atoms to be evaluated by GPAW: {core_indices}")
for idx in core_indices:
    print(f"  Atom {idx}: {atoms[idx].symbol} (Dist: {distances[idx]:.3f} Å)")

# 3. Create the hybrid cache directory
os.makedirs('data/vib_hybrid', exist_ok=True)

# 4. Run DFT Phonons ONLY for the core atoms
# By passing indices, ASE only displaces these specific atoms.
vib_gpaw = Vibrations(atoms, name='data/vib_hybrid', indices=core_indices)
print("Starting GPAW force evaluations for the defect core...")
vib_gpaw.run()
print("GPAW core evaluations complete. Cache saved in data/vib_hybrid/")