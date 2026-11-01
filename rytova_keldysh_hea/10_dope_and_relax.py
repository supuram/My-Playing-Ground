import numpy as np
from ase.io import read
from gpaw import GPAW, PW, FermiDirac
from ase.optimize import BFGS

# 1. Load the pristine relaxed structure
pristine = read('structures/HfS2_relaxed.traj')

# 2. Create a 3x3 in-plane supercell (Z remains 1 layer thick)
# The pristine cell has 3 atoms. The supercell will have 3 * 3 * 3 = 27 atoms.
supercell = pristine * (3, 3, 1)

# 3. Defect Engineering: Find a Sulfur atom and replace it with Oxygen
# We extract all indices of Sulfur atoms in the new supercell
s_indices = [atom.index for atom in supercell if atom.symbol == 'S']  # [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26]

# Pick a Sulfur atom near the center of the array to avoid edge-effect artifacts
target_idx = s_indices[len(s_indices) // 2]   # target_idx = 14
supercell[target_idx].symbol = 'O'

print(f"Created 3x3 supercell. Replaced atom {target_idx} (S) with O.")   # You can check supercell with symbols, pbc, cell
print(f"New composition: {supercell.symbols}")

# 4. Set up the DFT Calculator
# Because the real-space cell is now 3x larger, the reciprocal space (Brillouin zone) 
# is 3x smaller. We can safely reduce the k-point grid to (3, 3, 1) to save computation time.
calc = GPAW(
    mode=PW(500),  # Reduced slightly to 500 eV for supercell speed
    xc='PBE',
    kpts={'size': (3, 3, 1), 'gamma': True},
    occupations=FermiDirac(0.01),
    txt='data/HfS2_Odoped_relax.txt'
)
supercell.calc = calc

# 5. Run the Geometry Relaxation
print("Starting supercell relaxation... (This will take significant time)")
opt = BFGS(supercell, trajectory='structures/HfS2_Odoped_relaxed.traj')
opt.run(fmax=0.02)  # Relax until forces are below 0.02 eV/Å

supercell.write('structures/HfS2_Odoped_relaxed.cif')
print("Doped relaxation complete! Saved to structures/HfS2_Odoped_relaxed.cif")

'''
Starting supercell relaxation... (This will take significant time)
      Step     Time          Energy          fmax
BFGS:    0 13:17:14     -184.694074        4.414044
BFGS:    1 13:51:59     -185.138642        4.386422
BFGS:    2 14:37:17     -186.271192        3.267402
BFGS:    3 15:24:29     -186.811131        1.097626
BFGS:    4 16:00:17     -186.870297        0.791316
BFGS:    5 16:37:50     -186.942609        0.521811
BFGS:    6 17:12:05     -186.969591        0.196941
BFGS:    7 17:40:09     -186.976469        0.218050
BFGS:    8 18:03:33     -186.992883        0.181206
BFGS:    9 18:26:03     -187.000745        0.136655
BFGS:   10 18:42:22     -187.005467        0.108885
BFGS:   11 18:58:14     -187.008705        0.100765
BFGS:   12 19:17:46     -187.013896        0.143140
BFGS:   13 19:37:41     -187.020655        0.166498
BFGS:   14 20:04:47     -187.027084        0.206853
BFGS:   15 20:24:39     -187.032799        0.197790
BFGS:   16 20:43:40     -187.036883        0.143439
BFGS:   17 21:03:07     -187.039583        0.074539
BFGS:   18 21:22:03     -187.040789        0.045944
BFGS:   19 21:38:33     -187.041256        0.038002
BFGS:   20 21:54:36     -187.041536        0.032755
BFGS:   21 22:08:48     -187.041766        0.024049
BFGS:   22 22:20:39     -187.041901        0.016650
'''