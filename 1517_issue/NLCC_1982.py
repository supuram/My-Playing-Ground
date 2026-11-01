import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("lithium_radial_consistent.csv")

r = df["r_bohr"].to_numpy()    # r has shape of (5000,)
V = df["V_S_hartree"].to_numpy()
u = df["u_radial"].to_numpy()
epsilon = df["epsilon_hartree"].iloc[0]

l = 0
peak_index = np.argmax(np.abs(u))     # peak_index is not the value of u but the index at which it is maximum, say the index is 143 for eg
r_peak = r[peak_index]

r_c = 0.75 * r_peak

x = r / r_c   # x has shape of (5000,)
f = np.exp(-x**4)

c_l = 0.0
V_trial = (1.0 - f) * V + c_l * f

from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

dr = r[1] - r[0]

def calculate_eigenvalue(c):
    Vt = (1.0 - f) * V + c * f    # Vt has shape of (5000,)
    diagonal = 1.0 / dr**2 + Vt[1:-1]   # arr[1: -1] picks the whole array except the 1st and last element
    off_diagonal = np.full(len(diagonal) - 1, -0.5 / dr**2)

    energies = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="i",
        select_range=(0, 0),
        eigvals_only=True
    )
    return energies[0]

def eigenvalue_error(c):
    return calculate_eigenvalue(c) - epsilon

# Find c_l
c_candidates = np.linspace(-3.0, 1.0, 40)
errors = [eigenvalue_error(c) for c in c_candidates]

bracket = None
for i in range(len(c_candidates) - 1):
    if errors[i] * errors[i + 1] <= 0:
        bracket = (c_candidates[i], c_candidates[i + 1])
        break

if bracket is None:
    for c, err in zip(c_candidates, errors):
        print(f"c = {c:6.2f} Ha -> error = {err:+.6f} Ha")
    raise ValueError("Could not find a sign change. Inspect the errors above.")
c_l = brentq(eigenvalue_error, bracket[0], bracket[1])

V_trial = (1.0 - f) * V + c_l * f

def calculate_eigenstate(c):
    Vt = (1.0 - f) * V + c * f

    diagonal = 1.0 / dr**2 + Vt[1:-1]
    off_diagonal = np.full(len(diagonal) - 1, -0.5 / dr**2)

    energies, states = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="i",
        select_range=(0, 0),
        eigvals_only=False
    )

    u_trial = np.zeros_like(r)
    u_trial[1:-1] = states[:, 0]

    # Normalize the radial wavefunction
    u_trial /= np.sqrt(np.trapezoid(u_trial**2, r))

    return energies[0], u_trial

epsilon_trial, u_trial = calculate_eigenstate(c_l)

r_match = 2 * r_c
i_match = np.argmin(np.abs(r - r_match))

gamma_l = u[i_match] / u_trial[i_match]
u_scaled = gamma_l * u_trial  

outer = (r >= r_c) & (r <= 8.0)
difference = u_scaled[outer] - u[outer]

inside = r <= r_c
N_AE_core = np.trapezoid(u[inside]**2, r[inside])
N_trial_core = np.trapezoid(u_trial[inside]**2, r[inside])

g_l = (r / r_c)**(l + 1) * f

A = np.trapezoid(g_l**2, r)
B = np.trapezoid(u_trial * g_l, r)
C = np.trapezoid(u_trial**2, r)

roots = np.roots([A, 2.0 * B, C - 1.0 / gamma_l**2])   # Solves for delta_l

real_roots = roots[np.abs(roots.imag) < 1e-10].real
if len(real_roots) == 0:
    raise ValueError("No real solution for delta_l.")

delta_l = real_roots[np.argmin(np.abs(real_roots))]

u_pseudo = gamma_l * (u_trial + delta_l * g_l)

N_AE_core = np.trapezoid(u[inside]**2, r[inside])
N_PS_core = np.trapezoid(u_pseudo[inside]**2, r[inside])

d2u = np.gradient(np.gradient(u_pseudo, r), r)
V_pseudo = (epsilon + (0.5 * d2u / u_pseudo) - (l * (l + 1) / (2.0 * r**2)))
valid = (np.isfinite(V_pseudo) & (np.abs(u_pseudo) > 1e-8))

# ====================================================================================================================================
# NONLINEAR CORE CORRECTION (NLCC) IMPLEMENTATION
# ====================================================================================================================================

from scipy.integrate import cumulative_trapezoid, trapezoid

# 1. Valence and core parameters for Lithium (1s^2 2s^1)
Z_v = 1.0   # 1 valence electron
N_v = 1.0

# Extract all-electron core density saved from the PySCF run
rho_core = df["rho_core_bohr_minus3"].to_numpy()

# 2. Pseudo valence density from our smooth, norm-conserving wavefunction
rho_pseudo_val = np.zeros_like(r)
nonzero = r > 1e-12    # nonzero has shape (5000,)
rho_pseudo_val[nonzero] = N_v * (u_pseudo[nonzero] ** 2) / (4.0 * np.pi * r[nonzero] ** 2)

# 3. Construct the partial core density rho_tilde_core (Louie, Froyen, Cohen 1982)
# Choose core radius r_cc where core density is smooth and starting to decay
# For Li, ~ 0.8 to 1.2 Bohr is typical
r_cc = 0.9  # Bohr
i_cc = np.argmin(np.abs(r - r_cc))   # i_cc is a single number
r0 = r[i_cc]

# Calculate derivatives of rho_core at r_cc
d_rho = np.gradient(rho_core, r)
d2_rho = np.gradient(d_rho, r)

v0 = rho_core[i_cc]
v1 = d_rho[i_cc]
v2 = d2_rho[i_cc]

# Fit polynomial: rho_tilde_c(r) = c0 + c2*r^2 + c4*r^4  (only even powers ensures d/dr = 0 at r=0)
# Matches value, 1st deriv, and 2nd deriv at r0:
# [ 1   r0^2    r0^4  ] [c0]   [v0]
# [ 0   2*r0   4*r0^3 ] [c2] = [v1]
# [ 0    2    12*r0^2 ] [c4]   [v2]

M = np.array([
    [1.0, r0**2, r0**4],
    [0.0, 2.0 * r0, 4.0 * r0**3],
    [0.0, 2.0, 12.0 * r0**2]
])
rhs = np.array([v0, v1, v2])
c_poly = np.linalg.solve(M, rhs)

# Build rho_tilde_core
rho_tilde_core = np.copy(rho_core)
core_mask = r < r0
rho_tilde_core[core_mask] = c_poly[0] + c_poly[1] * r[core_mask]**2 + c_poly[2] * r[core_mask]**4

# Ensure no negative density near origin
rho_tilde_core = np.maximum(rho_tilde_core, 0.0)

# 4. Hartree potential from pseudo valence density V_H[rho_tilde_val]
integrand_1 = u_pseudo**2
cum_int_1 = cumulative_trapezoid(integrand_1, r, initial=0.0)

integrand_2 = np.zeros_like(r)
integrand_2[nonzero] = u_pseudo[nonzero] ** 2 / r[nonzero]
total_int_2 = trapezoid(integrand_2, r)
cum_int_2 = total_int_2 - cumulative_trapezoid(integrand_2, r, initial=0.0)

V_H_val = np.zeros_like(r)
V_H_val[nonzero] = (N_v * cum_int_1[nonzero]) / r[nonzero] + N_v * cum_int_2[nonzero]
V_H_val[~nonzero] = N_v * total_int_2

# 5. Exchange-correlation evaluation
# Standard HSC: evaluates XC solely on valence density
# NLCC: evaluates XC on (core + valence) density
from pyscf.dft import libxc

# Standard HSC: V_xc[rho_v]
_, vxc_hsc, _, _ = libxc.eval_xc('LDA,VWN', rho_pseudo_val, spin=0)
V_xc_hsc = vxc_hsc[0]

# NLCC: V_xc[rho_tilde_c + rho_v]
rho_total_nlcc = rho_tilde_core + rho_pseudo_val
_, vxc_nlcc, _, _ = libxc.eval_xc('LDA,VWN', rho_total_nlcc, spin=0)
V_xc_nlcc = vxc_nlcc[0]

# 6. Unscreen to get both bare potentials
# Standard HSC bare-ion potential:
V_ion_standard = np.full_like(r, np.nan)
V_ion_standard[valid] = V_pseudo[valid] - V_H_val[valid] - V_xc_hsc[valid]

# NLCC bare-ion potential:
V_ion_nlcc = np.full_like(r, np.nan)
V_ion_nlcc[valid] = V_pseudo[valid] - V_H_val[valid] - V_xc_nlcc[valid]

# 7. Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Panel 1: Partial core density
ax1.plot(r, rho_core, label=r"True core density $\rho_c$ (1s)")
ax1.plot(r, rho_tilde_core, "--", label=r"NLCC pseudo core density $\tilde{\rho}_c$")
ax1.plot(r, rho_pseudo_val, ":", label=r"Pseudo valence density $\tilde{\rho}_v$")
ax1.axvline(r_cc, color="gray", linestyle=":", label=r"$r_{cc}$ core boundary")
ax1.set_xlim(0, 4)
ax1.set_xlabel("r (Bohr)")
ax1.set_ylabel(r"Density (Bohr$^{-3}$)")
ax1.set_title("Core & Valence Densities with NLCC")
ax1.legend()

# Panel 2: Standard HSC Bare Ion vs. NLCC Bare Ion
ax2.plot(r[valid], V_ion_standard[valid], label=r"Standard HSC $V_{ion}(r)$")
ax2.plot(r[valid], V_ion_nlcc[valid], "--", label=r"NLCC $V_{ion}^{NLCC}(r)$")
ax2.plot(r[valid], -Z_v / r[valid], ":", color="black", label=r"Asymptotic $-Z_v/r$")
ax2.axvline(r_c, color="gray", linestyle=":", label=r"HSC $r_c$")
ax2.set_xlim(0, 5)
ax2.set_ylim(-3, 1)
ax2.set_xlabel("r (Bohr)")
ax2.set_ylabel("Potential (Hartree)")
ax2.set_title(r"Bare Ion Potentials ($V_{ion}$)")
ax2.legend()

plt.tight_layout()
plt.show()