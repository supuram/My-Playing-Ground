import numpy as np
from ase.build import mx2
from gpaw import GPAW, FermiDirac

atoms = mx2(formula='HfS2', kind='1T', a=3.63, thickness=2.87, vacuum=10.0)

calc = GPAW(
    mode='lcao',
    basis='dzp',
    xc='PBE',
    h=0.2,
    kpts=(1, 1, 1),  # Forces kpt: 1, requiring domain: 4
    occupations=FermiDirac(0.01),
    txt='HfS2_gamma_domain_test.txt'
)
atoms.calc = calc
atoms.get_potential_energy()
forces = atoms.get_forces()
print(f"Gamma-point energy: {atoms.get_potential_energy():.4f} eV")