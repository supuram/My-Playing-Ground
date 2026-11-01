from gpaw import GPAW
from gpaw.response.df import DielectricFunction
import numpy as np

# Load converged ground-state calculation
calc = GPAW('structures/HfS2_relaxed.gpw')

# Calculate macroscopic dielectric function / polarizability at small q
# For 2D systems, alpha_2D = (L_z / 4*pi) * [eps_M(q) - 1] as q -> 0
L_z = calc.atoms.cell[2, 2]  # Cell height along z (including vacuum)

df = DielectricFunction(
    calc='structures/HfS2_relaxed.gpw',
    frequencies=[0.0],       # Static limit omega = 0
    eta=0.05,
    ecut=50,                 # Local field cutoff (eV)
    hilbert=False,
    truncation='2D'
)

# Compute chi0 / dielectric function along in-plane q-vector [q_x, 0, 0]
q_val = [1/12, 0, 0]             # Small momentum transfer in 1/Å
alpha_nlf, alpha_lf = df.get_polarizability(q_c=q_val)   # without and with local-field effects
alpha_2D = alpha_lf[0].real

# r_0 in atomic units (Bohr)
# 1 Å = 1.88973 Bohr
r0_angstrom = 2 * np.pi * alpha_2D
r0_bohr = r0_angstrom * 1.88973

print(f"Static 2D polarizability alpha_2D: {alpha_2D:.3f} Å")
print(f"Effective screening length r0: {r0_angstrom:.3f} Å ({r0_bohr:.2f} a.u.)")

''' Output with nbands = 36
Static 2D polarizability alpha_2D: 1.861 Å
Effective screening length r0: 11.694 Å (22.10 a.u.)

Output with nbands = 250
Static 2D polarizability alpha_2D: 1.799 Å
Effective screening length r0: 11.304 Å (21.36 a.u.)
'''