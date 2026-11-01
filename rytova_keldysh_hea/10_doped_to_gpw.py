from ase.io import read
from gpaw import GPAW, PW, FermiDirac

# 1. Load the relaxed structure from step 1
atoms = read('structures/HfS2_Odoped_relaxed.traj')

# 2. Run single-point SCF with extra empty bands for response calculations
calc = GPAW(
    mode=PW(500),
    xc='PBE',
    nbands=250,
    kpts={'density': 6.0, 'gamma': True},
    occupations=FermiDirac(0.01),
    txt='structures/HfS2_Odoped_scf.txt'
)
atoms.calc = calc
atoms.get_potential_energy()

# 3. Save wavefunctions (mode='all' is required for response/dielectric calculations)
calc.write('structures/HfS2_Odoped_relaxed.gpw', mode='all')
print("Successfully generated structures/HfS2_Odoped_relaxed.gpw with wavefunctions!")