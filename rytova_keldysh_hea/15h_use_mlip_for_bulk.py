# Run this in your mattersim environment, not in GPAW

'''import os
import numpy as np
from ase.io import read
from mattersim.forcefield import MatterSimCalculator
from ase.vibrations import Vibrations

# 1. Load the exact same structure
atoms = read('structures/HfS2_Odoped_relaxed.traj')
atoms.calc = MatterSimCalculator(device="cpu")

# 2. Identify the bulk atoms (everything NOT in the core)
o_idx = atoms.get_chemical_symbols().index('O')
distances = atoms.get_distances(o_idx, range(len(atoms)), mic=True)
core_indices = [i for i, d in enumerate(distances) if d <= 3.0]
bulk_indices = [i for i in range(len(atoms)) if i not in core_indices]

print(f"Bulk atoms to be evaluated by MatterSim: {len(bulk_indices)} atoms")

# 3. Run MatterSim Phonons ONLY for the bulk atoms
# It will save the .pckl files into the EXACT SAME folder as GPAW
vib_ms = Vibrations(atoms, name='data/vib_hybrid', indices=bulk_indices)
print("Starting MatterSim force evaluations for the bulk lattice...")
vib_ms.run()
print("MatterSim bulk evaluations complete. Hybrid Hessian is now fully populated!")'''