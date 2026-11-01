'''Equations being talked about are from the 1959 Pseudopotential paper'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

a = 8.0
N_shells = 4
x = np.linspace(-a/2, a/2, 2000, endpoint=False)
dx = x[1] - x[0]

kappa = []
G_vals = []
for n in range(N_shells):
    G = n * (2 * np.pi / a)
    G_vals.append(G)
    if n == 0:
        k_n = np.full_like(x, 1.0 / np.sqrt(a))
    else:
        k_n = np.sqrt(2.0 / a) * np.cos(G * x)
    kappa.append(k_n)
kappa = np.array(kappa)   # kappa has shape (4, 2000)

H_eff = np.zeros((N_shells, N_shells))
A = 14.0
Z = 2.9
lam = 3.2
sigma = 0.5
r = np.sqrt(x**2 + sigma**2)
V_c = - (2.0 * Z / r) - (2.0 * (A - Z) / r) * np.exp(-lam * r)

# Parameters for Si 2s core state (Table I & Table II, Case I)
E_c = -11.64

# 1D core orbital: P_2s(|x|), here we calculate core wavefunction
psi_c_raw = (-29.0 * (r**1.0) * np.exp(-13.66 * r) + 81.5 * (r**2.2) * np.exp(-5.08 * r))

# Normalize on the grid: <psi_c | psi_c> = 1
norm = np.sqrt(np.sum(psi_c_raw**2) * dx)
psi_c = psi_c_raw / norm   # psi_c has shape (2000,), psi_c is the core wavefunction

overlap = np.zeros(N_shells)
for n in range(N_shells):
    overlap[n] = np.sum(psi_c * kappa[n]) * dx

A_mat = np.zeros((N_shells, N_shells))
B_mat = np.zeros((N_shells, N_shells))

# Evaluates eqn 2.6
for m in range(N_shells):
    for n in range(N_shells):
        # Kinetic term (diagonal)
        T_mn = (G_vals[n]**2) if m == n else 0.0
        
        # Attractive potential term: <kappa_m | V_c | kappa_n>
        V_c_mn = np.sum(kappa[m] * V_c * kappa[n]) * dx
        
        # A matrix has -E_c * |psi_c><psi_c|
        A_mat[m, n] = T_mn + V_c_mn - E_c * overlap[m] * overlap[n]
        
        # B matrix has delta_mn - |psi_c><psi_c|
        B_mat[m, n] = (1.0 if m == n else 0.0) - overlap[m] * overlap[n]
        
eigenvalues, eigenvectors = eigh(A_mat, B_mat)
print("Valence Energy E (Ry):", eigenvalues[1])
print("Valence Coefficients b_n:", eigenvectors[:, 1])  # means take the second column

# Construct the smooth wavefunction phi(x)
b = eigenvectors[:, 1]
phi = np.zeros_like(x)
for n in range(N_shells):
    phi += b[n] * kappa[n]
phi /= np.max(np.abs(phi))
if phi[len(phi)//2] < 0:
    phi = -phi


# True Wavefunction
a_c = - np.sum(psi_c * phi) * dx   # Eqn 2.2
psi_true = phi + a_c * psi_c    # Eqn 2.1
phi_plot = phi
psi_plot = psi_true / np.max(np.abs(psi_true))

plt.figure(figsize=(9, 4.5))
plt.plot(x, phi_plot, label=r'Pseudowavefunction $\varphi(x)$ (Smooth)', color='blue', lw=2)
plt.plot(x, psi_plot, label=r'True Wavefunction $\psi(x)$ (Orthogonalized, Nodal)', color='green', lw=2)
plt.plot(x, psi_c / np.max(psi_c), label=r'Core Orbital $\psi_c(x)$', color='red', linestyle='--', alpha=0.6)
plt.axhline(0, color='black', lw=0.5, linestyle=':')
plt.title('Phillips–Kleinman: Smooth $\\varphi(x)$ vs. True $\\psi(x)$')
plt.xlabel('x (Bohr)')
plt.ylabel('Normalized Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Check orthogonality: <psi_c | psi_true> must be 0
overlap_check = np.sum(psi_c * psi_true) * dx
print("Orthogonality check <psi_c | psi_true>:", overlap_check)

# V_net Calculation
overlap_val = np.sum(psi_c * phi) * dx
print("overlap_val = ", overlap_val)
# Local Repulsive Potential V_R(x) according to Eq. (2.8)
# Add a small epsilon to phi to avoid division by zero near its outer nodes
eps_div = 1e-8
# Central value of phi at the nucleus (x = 0)
phi_0 = phi[len(phi) // 2]
print("phi_0 = ", phi_0)
V_R = (eigenvalues[1] - E_c) * np.abs((overlap_val / phi_0)) * np.abs(psi_c)

# 3. Net Effective Potential (Pseudopotential): V_eff = V_c + V_R
V_eff = V_c + V_R

# 4. Plot within the atomic core region [-2, 2] Bohr to observe cancellation
mask = (x >= -2.0) & (x <= 2.0)

plt.figure(figsize=(8, 5))
plt.plot(x[mask], V_c[mask], label=r'Attractive Coulomb $V_c(x)$', color='blue', linestyle='--')
plt.plot(x[mask], V_R[mask], label=r'Repulsive Core $V_R(x)$ (Eq. 2.8)', color='orange', linestyle='--')
plt.plot(x[mask], V_eff[mask], label=r'Net Pseudopotential $V_{\mathrm{eff}}(x) = V_c + V_R$', color='black', lw=2.5)

plt.title('Phillips–Kleinman Cancellation Theorem in Real Space')
plt.xlabel('x (Bohr)')
plt.ylabel('Potential (Rydbergs)')
plt.legend()
plt.grid(True)
plt.show()