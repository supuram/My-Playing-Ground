'''To ensure the relaxation is physically centered around the oxygen dopant, I will check the per-atom displacement magnitudes '''

import numpy as np
from ase.io import read

atoms = read('structures/HfS2_Odoped_ES_relaxed.traj')
Delta_R = np.load('data/HfS2_Odoped_Delta_R.npy')

dr_norms = np.linalg.norm(Delta_R, axis=1)   # (27, )
symbols = atoms.get_chemical_symbols()   # total 27 symbols will be printed
o_idx = symbols.index('O')    # o_idx = 14

# Calculate the actual distance of every atom from the Oxygen atom (accounting for periodic boundaries)
distances = atoms.get_distances(o_idx, range(len(atoms)), mic=True)     # (27, )

# Combine into a structured view: [Index, Symbol, Distance from O, Displacement norm]
data = []
for i in range(len(atoms)):
    data.append((i, symbols[i], distances[i], dr_norms[i]))

# Sort atoms by their distance from Oxygen
data.sort(key=lambda x: x[2])

print(f"{'Atom':<8} {'Symbol':<8} {'Distance from O (Å)':<22} {'Displacement ΔR (Å)':<20}")
print("-" * 60)
for item in data[:8]:   # Near the defect
    print(f"{item[0]:<8} {item[1]:<8} {item[2]:<22.3f} {item[3]:<20.4f}")

print("...")
for item in data[-4:]:  # Far away at the cell boundary
    print(f"{item[0]:<8} {item[1]:<8} {item[2]:<22.3f} {item[3]:<20.4f}")
    
''' Output
Atom     Symbol   Distance from O (Å)    Displacement ΔR (Å) 
------------------------------------------------------------
14       O        0.000                  0.0384              
12       Hf       2.101                  0.0152              
24       Hf       2.104                  0.0156              
15       Hf       2.118                  0.0254              
16       S        3.082                  0.0248              
4        S        3.083                  0.0255              
13       S        3.083                  0.0233              
23       S        3.660                  0.0049              
...
22       S        5.924                  0.0205              
19       S        5.930                  0.0193              
20       S        6.339                  0.0490              
8        S        6.343                  0.0385   
'''