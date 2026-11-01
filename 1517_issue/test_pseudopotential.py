from gpaw.upf import UPFSetupData
from gpaw.pseudopotential import PseudoPotential

upf_file = 'Si.upf'

# 1. Parse and build the UPFSetupData object
setup_data = UPFSetupData(upf_file)

# 2. Build the PseudoPotential instance (or call setup_data.build(xcfunc=None, lmax=0, basis=None))
pp = PseudoPotential(setup_data)

from ase import Atoms
from gpaw import GPAW, PW
import numpy as np

atoms = Atoms('Si', positions=[[5.0, 5.0, 5.0]], cell=[10.0, 10.0, 10.0])
atoms.center()

calc = GPAW(
    mode=PW(300),          # Plane-wave cutoff (300 eV is good for a quick test)
    xc='PBE',
    setups={'Si': setup_data},
    txt=None   
)

atoms.calc = calc
print("Starting SCF calculation for Si...")
energy = atoms.get_potential_energy()
print(f"Calculation converged! Potential energy: {energy:.6f} eV")

# 5. Check forces
forces = atoms.get_forces()
print("Forces on Si atom:")
print(forces)

d0 = 2.35  # Angstroms
cell = [10.0, 10.0, 14.0]
def make_dimer(z1_pos):
  dimer = Atoms(
      'Si2',
      positions=[
          [5.0, 5.0, 7.0 - d0 / 2.0],  # Atom 0 fixed
          [5.0, 5.0, z1_pos],  # Atom 1 displaced
      ],
      cell=cell,
  )
  dimer.calc = GPAW(
      mode=PW(300),
      xc='PBE',
      setups={'Si': setup_data},
      occupations={'name': 'fermi-dirac', 'width': 0.05},
      nbands=16,
      convergence={'energy': 1e-7, 'density': 1e-6},
      txt=None,
  )
  return dimer


# Analytical forces at equilibrium reference
z1_ref = 7.0 + d0 / 2.0
dimer_ref = make_dimer(z1_ref)
forces_ref = dimer_ref.get_forces()

print("\nAnalytical Forces on Si2 (eV/Å):")
print(forces_ref)
print(
    f"Sum of forces (Newton's 3rd Law):"
    f" {np.linalg.norm(np.sum(forces_ref, axis=0)):.2e} eV/Å"
)

# Finite Difference along z for Atom 1
delta = 0.005  # displacement in Å
e_plus = make_dimer(z1_ref + delta).get_potential_energy()
e_minus = make_dimer(z1_ref - delta).get_potential_energy()

f_z1_num = -(e_plus - e_minus) / (2.0 * delta)

print(f"\nAtom 1 Analytical F_z : {forces_ref[1, 2]:10.6f} eV/Å")
print(f"Atom 1 Numerical  F_z : {f_z1_num:10.6f} eV/Å")
print(f"Discrepancy           : {abs(forces_ref[1, 2] - f_z1_num):10.6f} eV/Å")