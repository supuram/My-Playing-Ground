from ase.io import read
from ase.parallel import paropen
from gpaw import GPAW
import itertools
from ase.io import read

atoms = read('gs_Bi2Se3.gpw')
trim_kpts = list(itertools.product([0.0, 0.5], repeat=3))

bandpath = atoms.cell.bandpath('GZFGL', npoints=200)
with paropen('bandpath.json', 'w') as fd:
    bandpath.write(fd)

calc = GPAW('gs_Bi2Se3.gpw').fixed_density(
    kpts=bandpath,
    symmetry='off',
    txt='Bi2Se3_bands.txt')
calc.write('Bi2Se3_bands.gpw')
