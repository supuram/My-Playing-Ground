import matplotlib.pyplot as plt
from gpaw.dos import DOSCalculator
from ase import Atoms
from ase.visualize import view
from gpaw import GPAW, PW

name = 'Al-fcc'
a = 4.05  # fcc lattice parameter
b = a / 2

bulk = Atoms('Al',
             cell=[[0, b, b],
                   [b, 0, b],
                   [b, b, 0]],
             pbc=True)

view(bulk)

k = 4
calc = GPAW(mode=PW(300),       # cutoff
            kpts=(k, k, k),     # k-points
            txt=name + '.txt')  # output file

bulk.calc = calc

energy = bulk.get_potential_energy()
calc.write(name + '.gpw')
print('Energy:', energy, 'eV')

width = 0.1
dos = DOSCalculator.from_calculator(calc)
energies = dos.get_energies()
if dos.nspins == 2:
    plt.plot(energies, dos.raw_dos(energies, spin=0, width=width))
    plt.plot(energies, dos.raw_dos(energies, spin=1, width=width))
    plt.legend(('up', 'down'), loc='upper left')
else:
    plt.plot(energies, dos.raw_dos(energies, width=width))
plt.xlabel(r'$\epsilon - \epsilon_F \ \rm{(eV)}$')
plt.ylabel('Density of States (1/eV)')
plt.show()
