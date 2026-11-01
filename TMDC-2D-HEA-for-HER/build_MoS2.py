from ase.build import mx2
from ase.visualize import view
from gpaw import GPAW, PW, FermiDirac
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter

mos2 = mx2(formula='MoS2', kind='1T', a=3.18, thickness=3.19, size=(2, 2, 1), vacuum=10.0)
calc = GPAW(
    mode=PW(800),
    xc='PBE',
    spinpol=True,
    occupations=FermiDirac(width=0.05),
    kpts={'density': 6.0, 'gamma': True},
    txt='MoS2_1T_relax.log'
)
mos2.calc = calc
ucf = FrechetCellFilter(mos2, mask=[True, True, False, False, False, False])

# Relax until max force < 0.01 eV/Å
opt = BFGS(ucf, trajectory='MoS2_1T_relax.traj')
opt.run(fmax=0.01)

mos2.write('MoS2_1T_relaxed.cif')
print(f"Relaxed potential energy: {mos2.get_potential_energy():.4f} eV")