import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load our existing radial data
df = pd.read_csv("lithium_radial_consistent.csv")

# V is the V(s) in eqn 5-15 in A Chemists Guide to DFT by Koch and Holthausen, u is the wave function, epsilon is the energy in KS eqn
r = df["r_bohr"].to_numpy()    # r has shape of (5000,)
V = df["V_S_hartree"].to_numpy()
u = df["u_radial"].to_numpy()
epsilon = df["epsilon_hartree"].iloc[0]

l = 0

# 2. Locate the maximum of the radial wavefunction
peak_index = np.argmax(np.abs(u))     # peak_index is not the value of u but the index at which it is maximum, say the index is 143 for eg
r_peak = r[peak_index]

# HSC suggests a cutoff radius around 0.5–1.0 times
# the radius of the outermost peak of u_l(r).
r_c = 0.75 * r_peak

# 3. Define the HSC cutoff function
x = r / r_c   # x has shape of (5000,)
f = np.exp(-x**4)

# 4. First trial: use c_l = 0 temporarily
c_l = 0.0

''' V_trial modifies the all-electron effective potential inside the chosen core region, while approaching the original potential in 
the outer region. It is not yet the final pseudopotential.
Inside the core, r << r_c, f = 1, V_trial = c_l
Outside the core, r >> r_c, f = 0, V_trial = V '''
V_trial = (1.0 - f) * V + c_l * f

print("Original eigenvalue (Hartree):", epsilon)
print("Radial wavefunction peak (Bohr):", r_peak)
print("Chosen core radius r_c (Bohr):", r_c)
print("Trial constant c_l (Hartree):", c_l)

# ====================================================================================================================================

from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

dr = r[1] - r[0]

def calculate_eigenvalue(c):
    Vt = (1.0 - f) * V + c * f    # Vt has shape of (5000,)
    
    '''Constructs the diagonal and off-diagonal elements of the Hamiltonian matrix for solving the radial Schrödinger equation numerically. 
    We solve only for interior grid points because we impose u = 0 at the two endpoints.
    The equation is , (-0.5 * (d2u/dr2)) + [{(l*(l+1))/(2*r**2)} + V(trial)] * u = epsilon * u
    '''
    diagonal = 1.0 / dr**2 + Vt[1:-1]   # arr[1: -1] picks the whole array except the 1st and last element
    off_diagonal = np.full(len(diagonal) - 1, -0.5 / dr**2)

    # Calculates the epsilon values of the lowest state
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

print(f"Found root bracket: [{bracket[0]:.4f}, {bracket[1]:.4f}]")
c_l = brentq(eigenvalue_error, bracket[0], bracket[1])

V_trial = (1.0 - f) * V + c_l * f  # This potential looks just like the real all-electron potential V(r) in the valence region, but inside the core, it smoothly flattens out to the constant c_l

print("Adjusted c_l (Hartree):", c_l)
print("Target eigenvalue:", epsilon)
print("Trial eigenvalue:", calculate_eigenvalue(c_l))

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

print("Eigenvalue:", epsilon_trial)
print("Radial normalization:", np.trapezoid(u_trial**2, r))

''' What gurantees that u_trial will be smooth inside the core ?
 The smoothness of u_trial (which the paper calls w_1l) is entirely guaranteed by how you mathematically rigged your fake potential, 
V_trial. It comes down to the clever interaction between these two elements in your code:
The Cutoff Function (f): You defined your cutoff function as f(x) = exp(-x^4). Deep inside the core, as r approaches 0, this function 
smoothly evaluates to exactly 1.   
The Erased Singularity: Now look at your potential equation: V_trial = (1.0 - f) * V + c_l * f.
Because f->1 near the nucleus, the (1.0 - f) term becomes zero. This completely multiplies away the true all-electron potential V 
exactly where it is most dangerous—at the infinitely deep Coulomb singularity (-Z/r). In its place, the c_l * f term takes over, 
meaning the potential well smoothly flattens out to the simple, shallow constant value c_l '''

# HSC Eq. (3): scale the intermediate pseudo-wavefunction to match the original wavefunction outside the core.
# HSC amplitude matching at the core radius
r_match = 1.6 * r_c
i_match = np.argmin(np.abs(r - r_match))

gamma_l = u[i_match] / u_trial[i_match]
u_scaled = gamma_l * u_trial  # If u_scaled did not perfectly match the all-electron wavefunction u in the outer region, there would be no point in proceeding to the norm-conservation step.

print("HSC scaling factor gamma_l:", gamma_l)
print("Scaled wavefunction at matching region:")
print("  Original u(r):", u[i_match])
print("  Scaled u(r):  ", u_scaled[i_match])

# Check agreement outside the core
outer = (r >= r_match) & (r <= 6.0)
difference = u_scaled[outer] - u[outer]

print("Maximum absolute difference outside core:", np.max(np.abs(difference)))
print("RMS difference outside core:", np.sqrt(np.mean(difference**2)))

# All-electron norm inside the core
'''You are checking how much u and u_trial fail to match.
At this stage in the code, u_trial is just the initial smooth guess. Because it lacks the sharp, oscillating nodes of the real 
all-electron wavefunction (u), it inherently holds a different amount of integrated charge (norm) inside that core boundary (r<=r_c).
By calculating N_AE_core and N_trial_core, you are quantifying exactly how much charge is "wrong" in your trial state.
This discrepancy is the entire reason the subsequent block of code exists. Property 3 of the HSC paper demands strict "norm conservation,"
meaning the integral of the pseudo charge density must perfectly match the integral of the real charge density inside the core. If this 
norm is not conserved, Gauss's theorem dictates that the electrostatic potential outside the core will be incorrect, completely ruining 
how the atom bonds with other atoms in a large-scale DFT calculation.   
You calculate these two values as a diagnostic to see the error. The immediate next step in your code - solving the quadratic equation for 
delta - is the mathematical mechanism that explicitly forces that error to zero. 

True meaning of Norm -
When Hamann, Schlüter, and Chiang named their method "Norm-Conserving," they just meant "Charge-Conserving". They are stating a simple, 
non-negotiable rule of physics: if the real atom has exactly 2.0 electrons worth of charge locked inside that core radius, your fake 
pseudo-atom must also contain exactly 2.0 electrons worth of charge inside that same boundary. If it doesn't, Gauss's law dictates that 
the electrostatic repulsion felt by the valence electrons in the outer region will be completely wrong, and the atom won't bond correctly'''

inside = r <= r_c
N_AE_core = np.trapezoid(u[inside]**2, r[inside])
N_trial_core = np.trapezoid(u_trial[inside]**2, r[inside])

print("All-electron core norm:", N_AE_core)
print("Trial core norm:", N_trial_core)
print("Core norm difference:", N_trial_core - N_AE_core)

# ====================================================================================================================================

# Eqn 4, 5, 7 from the paper
# 1. HSC correction function: g_l(x) = x^(l+1) f(x)
'''Because your u_trial has the wrong amount of charge inside the core, you need a way to inject or remove charge without ruining the 
smoothness you just created. That's what g_l is for.'''
g_l = (r / r_c)**(l + 1) * f

# 2. Solve the norm-conservation equation:
# gamma_l^2 * integral[(u_trial + delta_l*g_l)^2 dr] = 1

A = np.trapezoid(g_l**2, r)
B = np.trapezoid(u_trial * g_l, r)
C = np.trapezoid(u_trial**2, r)

# A*delta_l^2 + 2*B*delta_l + C - 1/gamma_l^2 = 0
roots = np.roots([A, 2.0 * B, C - 1.0 / gamma_l**2])   # Solves for delta_l

# Select the real root with the smaller absolute magnitude
real_roots = roots[np.abs(roots.imag) < 1e-10].real
if len(real_roots) == 0:
    raise ValueError("No real solution for delta_l.")

delta_l = real_roots[np.argmin(np.abs(real_roots))]

# 3. Construct the corrected pseudo-wavefunction
u_pseudo = gamma_l * (u_trial + delta_l * g_l)

# 4. Check norm conservation inside the core
N_AE_core = np.trapezoid(u[inside]**2, r[inside])
N_PS_core = np.trapezoid(u_pseudo[inside]**2, r[inside])

print("\n--- HSC norm-conservation step ---")
print("Correction coefficient delta_l:", delta_l)
print("All-electron core norm:", N_AE_core)
print("Pseudo core norm:", N_PS_core)
print("Core norm difference:", N_PS_core - N_AE_core)
print("Total pseudo norm:", np.trapezoid(u_pseudo**2, r))

# =====================================================================================================================================

# Test whether u_pseudo is same as u in a regoin r<=R_test
R_test = 1.8 * r_c
test_region = r <= R_test

N_AE_test = np.trapezoid(u[test_region]**2, r[test_region])
N_PS_test = np.trapezoid(u_pseudo[test_region]**2, r[test_region])

print("Test radius R:", R_test)
print("All-electron norm up to R:", N_AE_test)
print("Pseudo norm up to R:      ", N_PS_test)
print("Difference at R:          ", N_PS_test - N_AE_test)

# 5. Plot the reference and corrected pseudo-wavefunctions
plt.figure(figsize=(8, 5))
plt.plot(r, u, label="Reference wavefunction")
plt.plot(r, u_pseudo, "--", label="Corrected pseudo-wavefunction")
plt.axvline(r_c, color="gray", linestyle=":", label="Core radius")
plt.xlim(0, 5)
plt.xlabel("r (Bohr)")
plt.ylabel("u(r)")
plt.legend()
plt.tight_layout()
plt.show()

# ======================================================================================================================================

# Radial Schrödinger equation inversion:
# V_PS(r) = epsilon + 0.5*u''(r)/u(r) - l(l+1)/(2*r^2)

d2u = np.gradient(np.gradient(u_pseudo, r), r)

V_pseudo = (epsilon + (0.5 * d2u / u_pseudo) - (l * (l + 1) / (2.0 * r**2)))

# Exclude points where division by a very small wavefunction is unstable
valid = (np.isfinite(V_pseudo) & (np.abs(u_pseudo) > 1e-8))

plt.figure(figsize=(8, 5))
plt.plot(r[valid], V[valid], label="Reference potential")
plt.plot(r[valid], V_pseudo[valid], label="Inverted pseudo-potential")
plt.axvline(r_c, color="gray", linestyle=":", label="Core radius")
plt.xlim(0, 5)
plt.ylim(-20, 10)
plt.xlabel("r (Bohr)")
plt.ylabel("Potential (Hartree)")
plt.legend()
plt.tight_layout()
plt.show()