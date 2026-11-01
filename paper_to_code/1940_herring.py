import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

a = 5.0
V = 5.0
R0 = 0.0
m = np.arange(-2, 3)
G = m * (2 * np.pi / a)
k = 0.0
q = k + G
alpha = 2.0
A = (1/np.sqrt(a)) * (((2 * np.pi)/alpha)**0.25) * np.exp((-q**2.0)/(4.0 * alpha))
S = np.eye(len(q)) - np.outer(A, A)    # 5x5 matrix

E_core = -5.0
delta_m = m[None, :] - m[:, None]      # 5x5 matrix
delta_G = delta_m * (2 * np.pi / a)    # 5x5 matrix

V0 = 4.0
beta = 1.0
# Fourier transform of the crystal potential directly evaluated at each delta_G
U_matrix = -(V0 / a) * np.sqrt(np.pi / beta) * np.exp(-delta_G**2 / (4.0 * beta))   # 5x5 matrix

T = np.diag(q**2)   # KE as a plane wave is an eigenstate of the kinetic energy operator with eigenvalue q^2
H_core = E_core * np.outer(A, A)
H = T + U_matrix - H_core

# Solves H c = E S c directly
energies, eigenvectors = la.eigh(H, S)

"""Energy Band Calculation"""
k_vals = np.linspace(-np.pi / a, np.pi / a, 100)
all_energies = []
for k in k_vals:
    G = m * (2.0 * np.pi / a)
    q = k + G
    A = (1.0 / np.sqrt(a)) * (((2.0 * np.pi) / alpha) ** 0.25) * np.exp(-q**2 / (4.0 * alpha))

    S = np.eye(len(q)) - np.outer(A, A)
    T = np.diag(q**2)
    H_core = E_core * np.outer(A, A)
    H = T + U_matrix - H_core

    evals, _ = la.eigh(H, S)
    all_energies.append(evals)

# Convert list of arrays into a 2D array of shape (100, 5)
all_energies = np.array(all_energies)

plt.figure(figsize=(7, 5))
for band_index in range(all_energies.shape[1]):
    plt.plot(k_vals, all_energies[:, band_index], label=f'Band {band_index + 1}')

plt.axvline(-np.pi / a, color='k', linestyle='--', alpha=0.5)
plt.axvline(np.pi / a, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Crystal momentum $k$')
plt.ylabel('Energy $E(k)$ (Rydbergs)')
plt.title('1D OPW Band Structure (Herring 1940)')
plt.legend()
plt.tight_layout()
plt.ylim(-2, 15)
plt.show()

"""The true physical valence wavefunction must oscillate near the nucleus to remain orthogonal to the core. You cannot remove that 
wiggle without violating the Pauli exclusion principle.
Who builds the wiggle?
Pure Plane Waves approach: You ask ordinary, smooth plane waves exp(iGr) to build that sharp wiggle from scratch. Because pure plane 
waves have no nodes near the nucleus, the computer has to add thousands of tiny-wavelength waves to force that dip to appear. 
Herring's OPW approach: You do not force the plane waves to build it. Instead, you subtract the core state analytically to create 
chi = |plane wave> - A* |u_core>. Because |u_core> is already localized at the nucleus, subtracting it instantly bakes the violent 
wiggle into the basis function itself.
The violent wiggle is not eliminated; it is transferred directly into the basis function chi so your solver doesn't have to waste 
thousands of pure plane waves trying to construct it. 
"""
