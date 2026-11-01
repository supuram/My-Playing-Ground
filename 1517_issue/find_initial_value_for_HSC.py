import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import trapezoid

# 1. Load the raw PySCF output for Lithium
df = pd.read_csv("lithium_ks_radial_data.csv")

r = df["r_bohr"].to_numpy()
V = df["V_S_hartree"].to_numpy()

dr = r[1] - r[0]
assert np.allclose(np.diff(r), dr)

# 2. Set up the radial grid Hamiltonian
ri = r[1:-1]
Vi = V[1:-1]
n = len(ri)

diagonal = 1.0 / dr**2 + Vi
off_diagonal = np.full(n - 1, -0.5 / dr**2)

# 3. Solve for the 2s valence state:
# Index 0 is 1s (core), Index 1 is 2s (valence)
energies, states = eigh_tridiagonal(
    diagonal,
    off_diagonal,
    select="i",
    select_range=(1, 1)  # Select the 2s first excited state!
)

epsilon_2s_grid = energies[0]
u_interior = states[:, 0]

# Normalize on the radial grid
u_interior /= np.sqrt(trapezoid(u_interior**2, ri))

# Restore boundary points u(0) = 0, u(r_max) = 0
u_2s_grid = np.zeros_like(r)
u_2s_grid[1:-1] = u_interior

# Enforce standard positive phase for the outer valence lobe
if u_2s_grid[np.argmax(np.abs(u_2s_grid))] < 0:
    u_2s_grid = -u_2s_grid

print("Grid 2s eigenvalue (Hartree): ", epsilon_2s_grid)
print("PySCF GTO 2s eigenvalue (Ha): ", df["epsilon_2s_hartree"].iloc[0])
print("Radial norm:                  ", trapezoid(u_2s_grid**2, r))

# 4. Save the grid-consistent file for your HSC and NLCC codes
pd.DataFrame({
    "r_bohr": r,
    "V_S_hartree": V,
    "u_radial": u_2s_grid,                       # True grid eigenstate
    "epsilon_hartree": np.full_like(r, epsilon_2s_grid),
    "rho_core_bohr_minus3": df["rho_core_bohr_minus3"],  # Retain core density for NLCC
    "rho_val_bohr_minus3": df["rho_val_bohr_minus3"]
}).to_csv("lithium_radial_consistent.csv", index=False)

print("Saved: lithium_radial_consistent.csv")