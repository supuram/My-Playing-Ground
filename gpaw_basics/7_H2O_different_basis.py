from ase.build import molecule
from gpaw import GPAW

a = 8.0
ecut = 400

energies = {}
with open(f'7_H2O_different_basis_results-{ecut:3.0f}.txt', 'w') as resultfile:

    for name in ['H2O', 'H', 'O']:
        atoms = molecule(name)
        atoms.set_cell((a, a, a))
        atoms.center()

        if name in ['H', 'O']:
            hund = True
        else:
            hund = False
            
        for basis in ['sz(dzp)', 'szp(dzp)', 'dzp']:

            calc = GPAW(mode='lcao', hund=hund, basis=basis,
                        txt=f'7_H2O_different_basis_gpaw-{name}-{ecut:3.0f}.txt')
            atoms.calc = calc

            energy = atoms.get_potential_energy()
            energies[name] = energy
            print(name, energy, file=resultfile)

    e_atomization = energies['H2O'] - 2 * energies['H'] - energies['O']
    print("Atomization Energy of H2O = ", e_atomization, file=resultfile)
