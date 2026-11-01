import numpy as np
from ase.io import read
from gpaw import GPAW, PW, FermiDirac
from ase.constraints import FixCartesian
from ase.optimize import BFGS

# 1. Load the converged pristine monolayer
monolayer = read('structures/HfS2_relaxed.traj')
cell = monolayer.get_cell()

# 2. Define the sliding grid (Fractional coordinates along the a-axis)
# Sliding from 0.0 to 1.0 covers exactly one full unit cell
shifts = np.linspace(0.0, 1.0, 11)
energies = []
dipoles_z = []

# Approximate van der Waals interlayer gap for HfS2 (in Ångströms)
vdw_gap = 3.10  

print("Starting sliding landscape scan...")

for shift in shifts:
    # 3. Build the Bilayer for this specific shift step
    bilayer = monolayer.copy()
    top_layer = monolayer.copy()
    
    # Extract fractional (crystal) coordinates to easily shift by unit cell percentages
    frac_positions = top_layer.get_scaled_positions()
    '''get_scaled_positions() translates the physical Cartesian coordinates of your atoms (in absolute Ångströms) into fractional 
        coordinates (percentages of your unit cell vectors).'''
    frac_positions[:, 0] += shift  # Slide along the 'a' lattice vector
    top_layer.set_scaled_positions(frac_positions)  # use print(top_layer.get_positions()) to see the top_layer movement
    '''set_scaled_positions() does the exact reverse, translating those fractions back into real physical distances and updating the 
        atoms' locations in space.''' 
    
    # Apply the out-of-plane Cartesian shift to create the van der Waals gap
    top_layer.positions[:, 2] += vdw_gap   # pushes the Z coordinate by vdw gap
    
    # Merge the two layers
    bilayer.extend(top_layer)  # combines the two layers
    
    # Expand the total simulation box Z-height to accommodate the second layer
    new_cell = cell.copy()
    new_cell[2, 2] += vdw_gap
    bilayer.set_cell(new_cell)
    bilayer.center(axis=2) # This command takes all the atoms and physically shifts them up the Z-axis (axis=2) until the entire 6-atom slab is perfectly in the geometric center of the new 25.97 Å box
    constraints = [FixCartesian(atom.index, [1, 1, 0]) for atom in bilayer]
    bilayer.set_constraint(constraints)
    
    '''
    [FixCartesian(atom.index, [1, 1, 0]) for atom in bilayer] creates a mathematical rule for the upcoming BFGS optimization step. It 
    iterates through every single atom in the bilayer and applies a Cartesian mask of [1, 1, 0], which corresponds directly to the 
    [X, Y, Z] axes.
    1 (True): Lock this coordinate. The atom is strictly forbidden from moving in the X and Y directions.
    0 (False): Free this coordinate. The atom is allowed to move up and down along the Z-axis.
    '''
    

    # 4. DFT Calculation for Energy and Polarization
    calc = GPAW(
        mode=PW(500), 
        xc='optPBE-vdW',
        kpts={'density': 6.0, 'gamma': True},
        occupations=FermiDirac(0.01),
        txt=f'data/HfS2_bilayer_shift_11_bilayer{shift:.1f}.txt'
    )
    bilayer.calc = calc
    print(f"Relaxing Z-coordinates for shift {shift:.2f}...")
    opt = BFGS(bilayer, trajectory=f'data/HfS2_shift_11_bilayer_{shift:.2f}.traj', logfile=None)
    opt.run(fmax=0.02)
    
    # Extract the total energy of the configuration
    e = bilayer.get_potential_energy()
    energies.append(e)
    
    # Extract the out-of-plane (Z) dipole moment to measure spontaneous sliding polarization
    # A non-zero Z-dipole proves the breaking of out-of-plane inversion symmetry
    dipole = calc.get_dipole_moment()[2] 
    dipoles_z.append(dipole)
    
    print(f"Shift: {shift:.1f} | Energy: {e:.3f} eV | Z-Dipole: {dipole:.4f} e*Å")

# 5. Save the landscape data
np.savez("data/HfS2_Sliding_Landscape_11_bilayer.npz", shifts=shifts, energies=energies, dipoles_z=dipoles_z)
print("Saved energy and polarization landscape to data/HfS2_Sliding_Landscape_11_bilayer.npz")

'''
The [2] index strictly extracts the out-of-plane Z-component of the dipole moment. The calc.get_dipole_moment() function calculates the 
total electric dipole of your simulation cell based on the ground-state electron density and returns it as a 3-element NumPy array 
representing Cartesian coordinates: [p_x, p_y, p_z].

In the context of your script, tracking this specific Z-component is measuring a phenomenon known as sliding ferroelectricity.
When 1T-HfS₂ layers are perfectly aligned in their native high-symmetry stacking, the system possesses inversion symmetry, meaning the 
out-of-plane dipole is exactly zero. As your script manually slides the top layer horizontally, it breaks that out-of-plane symmetry. 
This shift alters the local hybridization across the van der Waals gap, forcing a spontaneous redistribution of charge between the top 
and bottom sheets and generating a net electric polarization along the Z-axis.

By appending this Z-dipole value at every step, your script is building a polarization landscape map to identify exactly which sliding 
configurations produce the strongest spontaneous out-of-plane electric fields.
'''


''' Output
Shift: 1.0 | Energy: -97.267 eV | Z-Dipole: -0.0001 e*Å
Shift: 1.0 | Energy: -97.267 eV | Z-Dipole: -0.0001 e*Å
Shift: 1.0 | Energy: -97.267 eV | Z-Dipole: -0.0001 e*Å
Shift: 1.0 | Energy: -97.267 eV | Z-Dipole: -0.0001 e*Å
Shift: 1.0 | Energy: -97.267 eV | Z-Dipole: -0.0001 e*Å
Shift: 1.0 | Energy: -97.267 eV | Z-Dipole: -0.0001 e*Å
Shift: 1.0 | Energy: -97.267 eV | Z-Dipole: -0.0001 e*Å
Shift: 1.0 | Energy: -97.267 eV | Z-Dipole: -0.0001 e*Å
'''
