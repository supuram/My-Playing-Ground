import numpy as np
from ase.build import bulk, surface
from gpaw import GPAW, PW
from gpaw.solvation import Solvation, EffectivePotentialCavity, Power12Potential
from gpaw.solvation import NonLinearDielectric, LinearDielectric

nacl_bulk = bulk('NaCl', 'rocksalt', a=5.64)
slab = surface(nacl_bulk, (1, 0, 0), layers=2, vacuum=8.0)

slab.pbc = (True, True, False)

atomic_radii = {
    'Na': 2.27,
    'Cl': 1.75  # Standard Bondi/vdW radius for Chlorine
}

cavity = EffectivePotentialCavity(
    effective_potential=Power12Potential(atomic_radii=atomic_radii, u0=0.18),
    temperature=298.15
)

#dielectric_linear = NonLinearDielectric(solvent='thf') 
dielectric_linear = LinearDielectric(epsinf=78.4) 
# To run linear, simply swap back to gpaw.solvation.LinearDielectric(epsinf=78.4)

solvation = Solvation(cavity=cavity, dielectric=dielectric_linear)

# 4. GPAW Calculator with 2D K-points
calc = GPAW(
    mode=PW(200),
    xc='PBE',
    kpts=(4, 4, 1),      # 2D k-point mesh for the slab
    charge=0,            # The slab is overall neutral
    maxiter=1000,
    symmetry='off',      # Turn off symmetry to avoid continuum grid conflicts
    extensions=[solvation]
)

slab.calc = calc

# 5. Run the Calculation
print("Starting SCF for NaCl(100) in Non-Linear Water...")
E_total_solvent = slab.get_potential_energy()

calc = GPAW(
    mode=PW(200),
    xc='PBE',
    kpts=(4, 4, 1),      # 2D k-point mesh for the slab
    charge=0,            # The slab is overall neutral
    maxiter=1000,
    symmetry='off',      # Turn off symmetry to avoid continuum grid conflicts
)

slab.calc = calc
E_total_vacuum = slab.get_potential_energy()
Solvation_Energy = (E_total_solvent - E_total_vacuum) * 23.0609
print("Solvation Energy=", Solvation_Energy)