from ase.build import mx2
from ase.optimize import BFGS
from gpaw import GPAW, PW, FermiDirac

# 1. Setup 1T-HfS2 Monolayer (P-3m1, 3 atoms)
# mx2 default generates 2H; for 1T use octahedral coordination or structural parameters
atoms = mx2(formula='HfS2', kind='1T', a=3.63, thickness=2.87, vacuum=10.0)

# 2. Relax geometry with GPAW (PBE functional, plane-wave mode)
# In a 3-atom cell, PW(600) uses < 1.5 GB RAM
calc = GPAW(
    mode=PW(600),
    xc='PBE',
    kpts={'density': 6.0, 'gamma': True},
    occupations=FermiDirac(0.01),
    txt='structures/HfS2_relax.txt'
)
atoms.calc = calc

opt = BFGS(atoms, trajectory='structures/HfS2_relaxed.traj')
opt.run(fmax=0.01)

atoms.write('structures/HfS2_relaxed.cif')
print(f"Relaxed lattice constant a: {atoms.cell.cellpar()[0]:.3f} Å")

''' Output of the above code
  Step     Time          Energy          fmax
BFGS:    0 17:05:26      -20.434208        0.283279
BFGS:    1 17:08:23      -20.436300        0.238775
BFGS:    2 17:12:20      -20.440707        0.029211
BFGS:    3 17:17:00      -20.440780        0.001982
Relaxed lattice constant a: 3.630 Å
'''
