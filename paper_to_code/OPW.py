import numpy as np
import matplotlib.pyplot as plt

# 1. Real-space grid for one unit cell centered at the nucleus (r = 0)
a = 5.0
r = np.linspace(-a / 2, a / 2, 1000)
dr = r[1] - r[0]

# 2. Localized core state u(r) (normalized Gaussian)
alpha = 2.0
u_core = ((2.0 * alpha / np.pi) ** 0.25) * np.exp(-alpha * r**2)

# Point 1: Smooth plane wave |k> (take k = 0, which is the flat constant wave)
phi_pw = (1.0 / np.sqrt(a)) * np.cos(0.0 * r)  # or exp(i*k*r)

# Point 2: Measure core character A = <core | plane wave>
# Numerical integration over the grid
A_numerical = np.sum(u_core * phi_pw) * dr

# Point 3: Subtract that exact portion out to get chi(r)
chi = phi_pw - A_numerical * u_core

# Check orthogonality: <core | chi> must be 0
overlap = np.sum(u_core * chi) * dr
print(f"Core projection A: {A_numerical:.4f}")
print(f"Overlap <u_core | chi>: {overlap:.1e}")

# Plotting to see the nodal dip near the nucleus
plt.figure(figsize=(8, 4))
plt.plot(r, phi_pw, '--', label='Plane Wave $|k\\rangle$ (smooth)', color='gray')
plt.plot(r, u_core, ':', label='Core State $|u\\rangle$', color='crimson')
plt.plot(r, chi, label='OPW $\\chi(r) = |k\\rangle - A|u\\rangle$', color='blue', linewidth=2)
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel('r (Bohr)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()