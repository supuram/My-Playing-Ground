import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW

data = np.load('data/14_1s_exciton_BSE_Hamiltonian.npz', allow_pickle=True)
bse_obj = data['BSE_Hamiltonian_object'].item()     # <class 'gpaw.response.bse.BSEMatrix'>
H_bse = bse_obj.H_sS   # (2304, 2304)
eigenvalues, eigenvectors = np.linalg.eigh(H_bse)    # eigenvalues (2304, ), eigenvectors (2304, 2304)

# The 1s exciton is the lowest energy state (Index 0)
exciton_idx = 0
E_1s = eigenvalues[exciton_idx]
A_1s = eigenvectors[:, exciton_idx]    # A_1s (2304, )
print(f"1s Exciton Energy: {E_1s:.3f} eV")

print("4. Mapping the Real-Space Grid...")
# Grab a dummy wavefunction to get the 3D grid dimensions
calc = GPAW('structures/HfS2_relaxed.gpw')
atoms = calc.get_atoms()

dummy_wf = calc.get_pseudo_wave_function(band=0, kpt=0)  # dummy_wf has shape of (21, 21, 135)
grid_shape = dummy_wf.shape

# Initialize the empty 3D array for the exciton probability amplitude
Psi_exciton = np.zeros(grid_shape, dtype=complex)

# Atom 0 is Hf, Atoms 1 and 2 are S
s_pos = atoms.positions[1]   # s_pos = [ 1.815 1.04789074 12.89339239]
cell = atoms.get_cell()
inv_cell = np.linalg.inv(cell)    # (3, 3), This transforms the absolute Angstrom position into a dimensionless fraction

# Fixes one of the Sulphur atom as hole
fractional_pos = np.dot(s_pos, inv_cell)   # fractional_pos = [0.66666667 0.33333333 0.5637688]
hx = int(fractional_pos[0] * grid_shape[0])
hy = int(fractional_pos[1] * grid_shape[1])
hz = int(fractional_pos[2] * grid_shape[2])

print(f"Fixed hole grid indices: X={hx}, Y={hy}, Z={hz}")

print("5. Executing Tensor Summation for the Exciton Wavefunction...")
# Psi(r_e, r_h) = sum_{v,c,k} A_{vck} * psi_c(r_e) * psi_v^*(r_h)
# We loop over the exact basis ordering used by GPAW to build H_bse
kd = calc.wfs.kd
bz_map = kd.bz2ibz_k
v_bands = [8, 9, 10, 11]
c_bands = [12, 13, 14, 15]
nkpts = len(bz_map)   # nkpts = 144
pair_idx = 0

for k in range(nkpts):
    k_ibz = bz_map[k]
    for v in v_bands:
        # Load the valence (hole) orbital
        psi_v = calc.get_pseudo_wave_function(band=v, kpt=k_ibz)
        # Evaluate the hole orbital at the fixed Hafnium grid position
        psi_v_hole = np.conj(psi_v[hx, hy, hz])
        
        for c in c_bands:
            # Load the conduction (electron) orbital
            psi_c = calc.get_pseudo_wave_function(band=c, kpt=k_ibz)
            
            # Multiply by the BSE eigenvector coefficient and add to the total wave
            Psi_exciton += A_1s[pair_idx] * psi_c * psi_v_hole
            pair_idx += 1

print("6. Processing Probability Density and Plotting...")
# Probability density is |Psi|^2
prob_density = np.abs(Psi_exciton)**2

np.savez_compressed('data/14_1s_exciton_wavefunction.npz',
                    energies=eigenvalues, 
                    weights=eigenvectors,
                    wavefunction_exciton=Psi_exciton,
                    prob_density=prob_density)
# Take a 2D slice through the Z-plane of the Hafnium atom
hz_hf = int(0.5 * grid_shape[2])
Z_slice = prob_density[:, :, hz_hf]

plt.figure(figsize=(8, 7))

# 1. Create a fractional grid (0 to 1) mapping the NumPy array
frac_x = np.linspace(0, 1, grid_shape[0])
frac_y = np.linspace(0, 1, grid_shape[1])
frac_X, frac_Y = np.meshgrid(frac_x, frac_y, indexing='ij')

# 2. Transform the fractional grid into true non-orthogonal Cartesian coordinates (Å)
# X_cartesian = frac_X * a_x + frac_Y * b_x
# Y_cartesian = frac_X * a_y + frac_Y * b_y
X = frac_X * cell[0, 0] + frac_Y * cell[1, 0]
Y = frac_X * cell[0, 1] + frac_Y * cell[1, 1]

# Plot the physically mapped probability cloud
contour = plt.contourf(X, Y, Z_slice, levels=50, cmap='inferno')
plt.colorbar(contour, label='Electron Probability Density $|\\Psi|^2$')

# Overlay the lattice
for atom in atoms:
    color = 'cyan' if atom.symbol == 'Hf' else 'yellow'
    size = 100 if atom.symbol == 'Hf' else 50
    plt.scatter(atom.position[0], atom.position[1], c=color, s=size, edgecolors='black', zorder=5)

# Mark the stationary hole (Updated label)
plt.scatter(s_pos[0], s_pos[1], c='white', marker='X', s=150, edgecolors='black', 
            label='Fixed Hole (S)', zorder=6)

plt.xlabel('x (Å)', fontsize=12)
plt.ylabel('y (Å)', fontsize=12)
plt.title('Reconstructed 1s Exciton Wavefunction (1T-HfS$_2$)', fontsize=14)
plt.legend(loc='upper right')
plt.axis('equal')  # Crucial: forces X and Y axes to use the same physical scale
plt.tight_layout()

plt.savefig('structures/14_1s_exciton_wavefunction.png', dpi=300)
print("Saved exciton wavefunction plot.")