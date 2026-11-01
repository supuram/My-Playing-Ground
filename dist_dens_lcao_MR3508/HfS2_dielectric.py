import numpy as np
from ase.build import mx2
from gpaw import GPAW, FermiDirac
from gpaw.response.bse import BSE

# 1. Monolayer setup with 3D PBC
atoms = mx2(formula='HfS2', kind='1T', a=3.63, thickness=2.87, vacuum=10.0)
atoms.pbc = True

# 2. Ground state LCAO calculation saving wavefunctions
calc = GPAW(
    mode='lcao',
    basis='dzp',
    xc='PBE',
    h=0.25,
    kpts={'density': 3.0, 'gamma': True},
    occupations=FermiDirac(0.01),
    txt='HfS2_gs_bse.txt'
)
atoms.calc = calc
atoms.get_potential_energy()
calc.write('HfS2_lcao_bse.gpw', mode='all')

# 3. BSE calculation on top of LCAO ground state
bse = BSE(
    calc='HfS2_lcao_bse.gpw',
    valence_bands=[10, 11],       # Top valence bands
    conduction_bands=[12, 13],    # Bottom conduction bands
    ecut=50,
    eshift=0.0,
    txt='HfS2_bse_calc.txt'
)

# Diagonalize BSE Hamiltonian
bse.get_polarizability(filename='bse_pol.dat')
print("BSE calculation completed successfully!")