import numpy as np
from ase.build import mx2
from ase.optimize import BFGS
from gpaw import GPAW, FermiDirac

# 1. Setup 1T-HfS2 Monolayer (P-3m1, 3 atoms)
atoms = mx2(formula='HfS2', kind='1T', a=3.63, thickness=2.87, vacuum=10.0)

# 2. Relax geometry with GPAW in LCAO mode
calc = GPAW(
    mode='lcao',
    basis='dzp',
    xc='PBE',
    h=0.2,
    kpts={'density': 6.0, 'gamma': True},
    occupations=FermiDirac(0.01),
    txt='HfS2_lcao_relax.txt'
)
atoms.calc = calc

opt = BFGS(atoms, trajectory='HfS2_lcao_relaxed.traj')
opt.run(fmax=0.01)

atoms.write('HfS2_lcao_relaxed.cif')
print(f"Relaxed lattice constant a: {atoms.cell.cellpar()[0]:.3f} Å")