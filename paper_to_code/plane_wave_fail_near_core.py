""" Plane waves fail near nuclei because valence electrons must oscillate to stay orthogonal to core states.To show why pure plane waves 
fail, you can expand a valence wave that has a core-orthogonalization dip near the nucleus into standard Fourier plane waves and 
observe how many components are required to reproduce it.
"""

import numpy as np
import matplotlib.pyplot as plt

''' Real-space grid for one unit cell [-a/2, a/2] '''
a = 5.0
r = np.linspace(-a / 2, a / 2, 2000)
dr = r[1] - r[0]

''' Localized core state u(r) and smooth background envelope '''
alpha = 2.0
u_core = ((2.0 * alpha / np.pi) ** 0.25) * np.exp(-alpha * r**2)
phi_smooth = np.ones_like(r) / np.sqrt(a)

''' True valence state: smooth in interstitial, orthogonal dip at the nucleus '''
A = np.sum(u_core * phi_smooth) * dr
psi_valence = phi_smooth - A * u_core   # orthogonalized plane wave

''' Fourier expansion using pure plane waves: sum_{m} c_m * exp(i * G_m * r) '''
def fourier_reconstruct(psi_target, num_waves):
    m_vals = np.arange(-num_waves // 2, num_waves // 2 + 1)
    reconstruction = np.zeros_like(psi_target, dtype=complex)
    print(reconstruction)
    
    for m in m_vals:
        G = m * (2.0 * np.pi / a)
        pw = np.exp(1j * G * r) / np.sqrt(a)
        # Fourier coefficient c_m = <PW | psi>s
        c_m = np.sum(np.conj(pw) * psi_target) * dr  
        '''c_m is a single scalar number. psi_target is sitting there solely as the target reference. It tells the program - Here is 
        the true physical electron state. Measure the overlap of pure plane wave m with this target so we know what coefficient c_m to 
        give it.'''
        reconstruction += c_m * pw   # actual plane wave being made
        
    return np.real(reconstruction)

# Test convergence with 5, 21, and 101 plane waves
psi_5 = fourier_reconstruct(psi_valence, 5)
psi_21 = fourier_reconstruct(psi_valence, 21)
psi_101 = fourier_reconstruct(psi_valence, 101)

# 4. Plot comparison
plt.figure(figsize=(9, 5))
plt.plot(r, psi_valence, 'k-', linewidth=2.5, label='Target Valence State (Orthogonal to Core)')
plt.plot(r, psi_5, '--', label='Pure Plane Waves (N = 5)')
plt.plot(r, psi_21, '-.', label='Pure Plane Waves (N = 21)')
plt.plot(r, psi_101, ':', label='Pure Plane Waves (N = 101)')

plt.xlim(-1.5, 1.5)
plt.xlabel('Position $r$ near nucleus (Bohr)')
plt.ylabel('Amplitude')
plt.title('Fourier Failure: Slow Convergence of Pure Plane Waves Near Nucleus')
plt.legend()
plt.tight_layout()
plt.show()