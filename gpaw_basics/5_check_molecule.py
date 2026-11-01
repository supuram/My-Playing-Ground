from ase.build import molecule
from gpaw import GPAW
a = 8.0
name = 'H2O'
atoms = molecule(name)
print(atoms)
print(atoms.set_cell((a, a, a)))
print(atoms.center())
calc = GPAW(mode='pw')
atoms.calc = calc

print(dir(atoms))
print(atoms.get_dipole_moment())
print(atoms.get_positions())