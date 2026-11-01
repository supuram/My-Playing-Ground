from ase.atoms import Atoms
from gpaw import GPAW

b = 0.74
atoms = Atoms(['H', 'H'], positions=[(0, 0, 0), (0, 0, b)])
atoms.center(vacuum=6.0)

# Store parameters to easily pass to fresh calculator instances
calc_params = dict(
    mode={'name': 'pw', 'ecut': 600}, 
    eigensolver={'name': 'davidson', 'niter': 3}, 
    symmetry='off',
    convergence={'eigenstates': 1e-8, 'density': 1e-6},
    xc='EXX',
    legacy_gpaw=False
)

# 1. Base calculation
calc_exx_new = GPAW(**calc_params, txt='h2_exxnew.txt')
atoms.calc = calc_exx_new
f1 = atoms.get_forces()
e1 = atoms.get_potential_energy(force_consistent=True)
positions = atoms.get_positions()

for h in [0.001]:
    # 2. Forward displacement (attach fresh calculator)
    atoms.set_positions(positions)
    atoms.positions[1, 2] += h
    atoms.calc = GPAW(**calc_params, txt='h2_exxnew_plus.txt')
    e2 = atoms.get_potential_energy(force_consistent=True)

    # 3. Backward displacement (attach fresh calculator)
    atoms.set_positions(positions)
    atoms.positions[1, 2] -= h
    atoms.calc = GPAW(**calc_params, txt='h2_exxnew_minus.txt')
    e3 = atoms.get_potential_energy(force_consistent=True)

    # 4. Calculate finite difference
    atoms.set_positions(positions)
    fd = -(e2 - e3) / (2 * h)

    if calc_exx_new.world.rank == 0:
        print(f'h={h:.3e}, '
              f'force={f1[1, 2]:.8e}, '
              f'finite difference={fd:.8e}, '
              f'error={fd - f1[1, 2]:.3e} eV')