import numpy as np
from ase.build import fcc111
from gpaw import GPAW, PW
from gpaw.extensions.electrochemistry import SJM

atoms = fcc111('Pt', size=(1, 1, 2), vacuum=10.0)

sjm = SJM(
    potentiostat={
        'target_potential': 4.4,     # V vs vacuum
        'excess_electrons': 0.35,    # warm start closer to target
        'tol': 0.05,
        'max_iters': 12,             # allow sufficient iterations
    }
)

calc = GPAW(
    mode=PW(200),
    kpts=(1, 1, 1),
    xc='PBE',
    extensions=[sjm],
    txt='sjm_test.log'
)
atoms.calc = calc
energy = atoms.get_potential_energy()

print(f"Grand potential energy: {energy:.4f} eV")
print(f"Converged excess electrons: {sjm.excess_electrons:.4f}")
print(f"Electrode potential: {sjm.electrode_potential:.4f} V")