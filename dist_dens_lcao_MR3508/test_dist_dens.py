from ase.build import molecule
from gpaw import GPAW

# 1. Build a small test system (swap 'H2' with 'H2O' to test water)
system_name = 'H2O'
atoms = molecule(system_name)
atoms.center(vacuum=4.0)

# 2. Set up the LCAO calculator
# On Jens's branch, this should automatically use dist_dens=True
calc = GPAW(mode='lcao',
            basis='dzp',
            h=0.2,
            txt=f"{system_name}_lcao_test.txt") 

atoms.calc = calc

# 3. Run the SCF cycle
print(f"Starting LCAO calculation for {system_name}...")
energy = atoms.get_potential_energy()

print(f"Success! {system_name} Energy: {energy:.4f} eV")