import numpy as np
from scipy.special import struve, yn
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from gpaw.response.g0w0 import G0W0

# 1. Load parameters from Step 3 & 4
data = np.load('data/HfS2_params.npz')
r0_bohr = float(data['r0_bohr'])      # ~22.10 a.u.
mu_eff = float(data['mu_eff'])        # ~0.315 m0
direct_gap = float(data['direct_gap'])# ~2.066 eV

def rytova_keldysh(r, r0):
    x = np.maximum(r / r0, 1e-7)
    # Attractive potential in atomic units: - (pi / 2*r0) * [H0(x) - Y0(x)]
    return - (np.pi / (2.0 * r0)) * (struve(0, x) - yn(0, x))

def solve_wannier(mu, r0, r_max=150.0, n_pts=3000, n_states=6):
    dr = r_max / n_pts
    r = np.linspace(dr, r_max, n_pts)
    
    # Finite-difference kinetic operator d^2/dr^2
    main = 2.0 * np.ones(n_pts) / (dr**2)                                   # (3000,)
    off = -1.0 * np.ones(n_pts - 1) / (dr**2)                               # (2999,)
    T = diags([off, main, off], [-1, 0, 1]).toarray() / (2.0 * mu)          # (3000, 3000)
    
    # Effective potential for s-like excitons (m=0)
    V_centrifugal = - 1.0 / (8.0 * mu * r**2)           # (3000,), this line filters out the p, d, f states
    V_rk = rytova_keldysh(r, r0)                        # (3000,)
    V_tot = diags(V_centrifugal + V_rk, 0).toarray()    # (3000, 3000)
    
    H = T + V_tot
    vals, vecs = eigsh(H, k=n_states, which='SA')    # vals has shape (6,) and vecs has shape (3000, 6)
    
    HA_TO_EV = 27.211386
    eb_ev = - vals * HA_TO_EV   # Binding energies are positive: E_b = -E, eb_ev has shape (6,)

    # Radial wavefunctions R(r) = u(r) / sqrt(r)
    rad_wf = np.zeros_like(vecs)
    for i in range(n_states):
        u = vecs[:, i]       # (3000,)
        # Normalize: int |R(r)|^2 r dr = 1 => int |u(r)|^2 dr = 1
        u /= np.sqrt(np.trapezoid(u**2, r))
        rad_wf[:, i] = u / np.sqrt(r)    # (3000, 6), rad_wf holds the wavefunction of 1s, 2s, 3s, 4s, 5s, 6s
        
    return r, eb_ev, rad_wf

r_grid, eb_spectrum, r_wavefunctions = solve_wannier(mu_eff, r0_bohr)

print("Starting G0W0 calculation for Quasiparticle Gap (This may take a few minutes)...")
# Calculates the true quasiparticle band gap by running the G0W0 approximation
gw = G0W0(
    calc='structures/HfS2_relaxed.gpw',
    filename='data/HfS2_gw_results',
    bands=(10, 14),           # HfS2 VBM is band 11, CBM is band 12
    nbands=250,
    kpts=[0],                 # Only calculate at Gamma (k=0) where direct gap is
    ecut=50,
    truncation='2D',          # CRITICAL: Prevent layers interacting across vacuum
    eta=0.1
)
results = gw.calculate()

# Extract quasiparticle energies
vbm_qp = results['qp'][0, 0, 11 - 10]
cbm_qp = results['qp'][0, 0, 12 - 10]
direct_gap_qp = cbm_qp - vbm_qp

# ==========================================================
# PART 3: Final Optical Math & Output
# ==========================================================
print(f"\n{'=' * 45}")
print(f"Final Exciton & Optical Spectrum (1T-HfS2):")
print(f"Screening length r0: {r0_bohr:.2f} a.u. | mu_eff: {mu_eff:.3f} m0")
print("-" * 45)
print(f"PBE Direct Gap:            {direct_gap:.3f} eV")
print(f"G0W0 Direct Gap:          {direct_gap_qp:.3f} eV")
print(f"GW Scissors Shift:        {direct_gap_qp - direct_gap:.3f} eV")
print("-" * 45)

for n, eb in enumerate(eb_spectrum):
    if eb > 0:
        # True optical absorption = GW Gap - Binding Energy
        optical_transition = direct_gap_qp - eb
        print(f"State {n+1}s: Binding Energy Eb = {eb:.3f} eV | Optical Peak = {optical_transition:.3f} eV")
print("=" * 45)

# ==========================================================
# PART 4: Visualizing the Screened Potential & Wavefunctions
# ==========================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

r_ang = r_grid * 0.529177
ax1.plot(r_ang, rytova_keldysh(r_grid, r0_bohr) * 27.211386, 'r-', label='Rytova-Keldysh $W(r)$')
ax1.plot(r_ang, - (1.0 / r_grid) * 27.211386, 'k--', alpha=0.5, label=r'Bare Coulomb $-1/r$')
ax1.set_xlim(0, 30)
ax1.set_ylim(-5, 0)
ax1.set_xlabel('r (Å)')
ax1.set_ylabel('Potential (eV)')
ax1.set_title('Screened 2D Coulomb Potential')
ax1.legend()

for n in range(min(3, len(eb_spectrum))):
    rho = r_grid * (r_wavefunctions[:, n]**2)
    ax2.plot(r_ang, rho, lw=1.5, label=f'{n+1}s state')
ax2.set_xlim(0, 40)
ax2.set_xlabel('r (Å)')
ax2.set_ylabel(r'Radial Probability Density $r|R(r)|^2$')
ax2.set_title('Exciton Envelope Wavefunctions')
ax2.legend()

plt.tight_layout()
plt.savefig('structures/HfS2_excitons.png', dpi=200)
print("Saved exciton analysis to structures/HfS2_excitons.png")

''' Without the GW Approximation this was the output 
State 1s: Binding Energy Eb = 1.297 eV | Optical Peak = 0.769 eV
State 2s: Binding Energy Eb = 0.530 eV | Optical Peak = 1.536 eV
State 3s: Binding Energy Eb = 0.294 eV | Optical Peak = 1.772 eV
State 4s: Binding Energy Eb = 0.172 eV | Optical Peak = 1.894 eV
State 5s: Binding Energy Eb = 0.037 eV | Optical Peak = 2.029 eV

With GW Approximation this is the output
Starting G0W0 calculation for Quasiparticle Gap (This may take a few minutes)...
IBZ(<points: 19, symmetries: 12>)
Occupied bands: 12
Transforming wave functions from IBZ to BZ: 2.513 seconds
Distributing wave functions and iFFT-ing to real space: 17.960 seconds
Calculating eigenvalues:
  k-points: 1
  Bands: 10-13 (inclusive)
  Seconds: 52.380 (wave-function broadcasting: 0.137 seconds)
HSE06-eigenvalue shifts:
  min: -2.155 eV
  ave: 1.465 eV
  max: 5.084 eV

=============================================
Final Exciton & Optical Spectrum (1T-HfS2):
Screening length r0: 22.10 a.u. | mu_eff: 0.315 m0
---------------------------------------------
PBE Direct Gap:            2.066 eV
G0W0 Direct Gap:          3.922 eV
GW Scissors Shift:        1.856 eV
---------------------------------------------
State 1s: Binding Energy Eb = 1.297 eV | Optical Peak = 2.625 eV
State 2s: Binding Energy Eb = 0.530 eV | Optical Peak = 3.392 eV
State 3s: Binding Energy Eb = 0.294 eV | Optical Peak = 3.628 eV
State 4s: Binding Energy Eb = 0.172 eV | Optical Peak = 3.750 eV
State 5s: Binding Energy Eb = 0.037 eV | Optical Peak = 3.885 eV
=============================================
'''