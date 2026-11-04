from gpaw import GPAW, PW
from ase.build import molecule
from ase.units import mol, kJ, kcal, Pascal, m
from ase.parallel import parprint
from ase import Atoms
from gpaw.solvation import (
    Solvation,             # the solvation calculator
    EffectivePotentialCavity,  # cavity using an effective potential
    Power12Potential,          # a specific effective potential
    NonLinearDielectric,  # rule to construct permittivity func from the cavity
    GradientSurface,  # rule to calculate the surface area from the cavity
    SurfaceInteraction  # rule to calculate non-electrostatic interactions
)

# all parameters on the user side of the solvation API follow the ASE
# unit conventions (eV, Angstrom, ...)

# non-solvent related DFT parameters
vac = 4.0

# solvent parameters for water from J. Chem. Phys. 141, 174108 (2014)
u0 = 0.180  # eV
epsinf = 78.36  # dimensionless
gamma = 18.4 * 1e-3 * Pascal * m  # convert from dyne / cm to eV / Angstrom**2
T = 298.15  # Kelvin
# atomic_radii = {'Rb': 3.90}
atoms = molecule('CH3CONH2')
atoms.center(vacuum=vac)
# atoms.set_initial_magnetic_moments([0.0])

cavity=EffectivePotentialCavity(effective_potential=Power12Potential(u0=0.18), temperature=T)
dielectric=NonLinearDielectric(solvent='thf')
interactions=[SurfaceInteraction(surface_tension=gamma)]
solvation = Solvation(cavity=cavity, dielectric=dielectric)
# perform calculation with continuum solvent model from
# J. Chem. Phys. 141, 174108 (2014)
calc = GPAW(
    mode=PW(200),
    xc='PBE',
    charge=0,
    maxiter=1000,
    extensions=[solvation])
atoms.calc = calc
E_non_linear = atoms.get_potential_energy()
E_force = atoms.get_forces()
print("E_non_linear = ", E_non_linear)
print("E_force = ", E_force)

calc = GPAW(
    mode=PW(200),
    xc='PBE',
    charge=0,
    maxiter=1000)
atoms.calc = calc
E_non_linear_alone = atoms.get_potential_energy()
E_force_alone = atoms.get_forces()
print("E_non_linear_alone = ", E_non_linear_alone)
print("E_force_alone = ", E_force_alone)

Solvation_Energy = (E_non_linear - E_non_linear_alone) * 23.0609
print("Solvation Energy = ", Solvation_Energy)