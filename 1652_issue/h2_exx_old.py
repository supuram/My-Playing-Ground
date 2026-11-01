from ase.atoms import Atoms
from gpaw import GPAW

b = 0.74
atoms = Atoms(['H', 'H'], positions=[(0, 0, 0), (0, 0, b)])
atoms.center(vacuum=6.0)
calc_params = dict(
    mode={'name': 'pw', 'ecut': 600}, eigensolver={'name': 'davidson', 'niter': 3}, symmetry='off',
    convergence={'eigenstates': 1e-8, 'density': 1e-6},
)
calc_exx_old = GPAW(**calc_params, xc='PBE',
                    txt='h2_exxold.txt', legacy_gpaw=True)
atoms.calc = calc_exx_old
f1 = atoms.get_forces()
e1 = atoms.get_potential_energy(force_consistent=True)
positions = atoms.get_positions()

for h in [0.001]:
    atoms.set_positions(positions)
    atoms.positions[1, 2] += h
    e2 = atoms.get_potential_energy(force_consistent=True)

    atoms.set_positions(positions)
    atoms.positions[1, 2] -= h
    e3 = atoms.get_potential_energy(force_consistent=True)

    atoms.set_positions(positions)
    fd = -(e2 - e3) / (2 * h)

    if calc_exx_old.world.rank == 0:
        print(f'h={h:.3e}, '
              f'force={f1[1, 2]:.8e}, '
              f'finite difference={fd:.8e}, '
              f'error={fd - f1[1, 2]:.3e} eV')
