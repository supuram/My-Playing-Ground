import numpy as np

''' 1. Spatial grid: r from 0 to 10 '''
r = np.linspace(0, 10, 1000)
dr = r[1] - r[0]
r_c = 2.0  # Augmentation radius (cutoff)

# Building equation 9
''' 2. Define the true AE partial wave phi(r), AE means All Electron
Inside r_c it wiggles violently: sin(4*pi*r) * exp(-r)
Outside r_c it decays smoothly: exp(-r) '''
phi_ae = np.where(r < r_c, np.sin(4 * np.pi * r) * np.exp(-r), np.exp(-r))

''' 3. Define the smooth PS partial wave phi_tilde(r), PS means Pseudo
Must be identical to phi_ae outside r_c, but smooth inside r_c
Here we just use exp(-r) everywhere '''
phi_ps = np.exp(-r)

''' 4. Construct a projector p_tilde(r) localized inside r_c
Condition (Eq. 7): <p_tilde | phi_ps> = 1
Let's take a simple shape that is non-zero ONLY for r < r_c '''
p_raw = np.where(r < r_c, (r_c - r) ** 2, 0.0)
overlap = np.sum(p_raw * phi_ps) * dr   # See the para just below eqn 7, there <f| is p_raw here
p_tilde = p_raw / overlap  # Normalizing so <p_tilde | phi_ps> = 1.0

''' Verify Eq. (7): <p_tilde | phi_ps> == 1.0 '''
print(f"<p_tilde | phi_ps> = {np.sum(p_tilde * phi_ps) * dr:.6f}")

''' 5. Define an actual valence state: the smooth pseudo wave function psi_ps (\tilde{\Psi})
In a real code, this comes from diagonalizing the Hamiltonian on a plane-wave or real-space grid.
For our 1D toy test, let's say our smooth state has an amplitude of 1.5:'''
psi_ps = 1.5 * np.exp(-r)

''' 6. Calculate projection coefficient c_i (Eq. 6: c_i = <p_tilde_i | psi_ps>)
This is just a single number (a scalar product) '''
c = np.sum(p_tilde * psi_ps) * dr

print(f"Projection coefficient c = {c:.4f}")

''' 7. Reconstruct the true All-Electron wave function psi_ae (Eq. 9)
|\Psi> = |\tilde{\Psi}> + (|\phi> - |\tilde{\phi}>) * c '''
psi_ae = psi_ps + (phi_ae - phi_ps) * c

# ====================================================================================================================================

# Building equation 30

''' 8. Compute the overlap difference on the radial grid (Eq. 30):
Delta_S = <phi_ae | phi_ae> - <phi_ps | phi_ps> '''
norm_phi_ae = np.sum(phi_ae * phi_ae) * dr
norm_phi_ps = np.sum(phi_ps * phi_ps) * dr
delta_S = norm_phi_ae - norm_phi_ps

''' 9. Direct All-Electron norm: <psi_ae | psi_ae> '''
norm_direct = np.sum(psi_ae * psi_ae) * dr

''' 10. PAW Overlap operator norm: <psi_ps | O_tilde | psi_ps> using Eq. 30
<psi_ps | 1 | psi_ps> + <psi_ps | p_tilde> * delta_S * <p_tilde | psi_ps> '''
norm_smooth_part = np.sum(psi_ps * psi_ps) * dr
norm_augmentation_part = c * delta_S * c  # since c = <p_tilde | psi_ps>
norm_paw = norm_smooth_part + norm_augmentation_part

print(f"Direct AE norm <Psi|Psi>        = {norm_direct:.8f}")
print(f"PAW operator   <psi_ps|O|psi_ps> = {norm_paw:.8f}")

# ====================================================================================================================================

# Building eqn 11

''' 11. Define a potential operator V(r) '''
V = -2.0 / (r + 0.1)

''' 12. Evaluate the expectation value DIRECTLY with the All-Electron wave function: 
<Psi | V | Psi> = \int psi_ae(r) * V(r) * psi_ae(r) dr'''
val_direct = np.sum(psi_ae * V * psi_ae) * dr

''' 13. Evaluate using the PAW operator A_tilde (Eq. 11):
Term 1: Smooth part <psi_ps | V | psi_ps> '''
val_smooth = np.sum(psi_ps * V * psi_ps) * dr

''' Term 2: Augmentation bracket ( <phi_ae | V | phi_ae> - <phi_ps | V | phi_ps> ) '''
delta_V = np.sum(phi_ae * V * phi_ae) * dr - np.sum(phi_ps * V * phi_ps) * dr

''' Term 3: Assemble using the projection coefficients c = <p_tilde | psi_ps> '''
val_paw = val_smooth + c * delta_V * c

print(f"Direct AE expectation <Psi | V | Psi>       = {val_direct:.8f}")
print(f"PAW expectation       <psi_ps | V_tilde | psi_ps> = {val_paw:.8f}")

# ====================================================================================================================================

# Eqn 14, 15, 16, 17

''' 14. Direct All-Electron density '''
n_ae = psi_ae ** 2

''' 15. The three PAW density components (Eqs. 15, 16, 17):
(a) Smooth pseudo density over entire space '''
n_tilde = psi_ps ** 2

''' (b) AE on-center density (Eq. 16) '''
n_1 = c * (phi_ae ** 2) * c

''' (c) PS on-center density (Eq. 17) '''
n_tilde_1 = c * (phi_ps ** 2) * c

''' 16. Full PAW density reconstruction (Eq. 14): '''
n_paw = n_tilde + n_1 - n_tilde_1

''' Check the maximum absolute error across all grid points '''
max_error = np.max(np.abs(n_ae - n_paw))
print(f"Max absolute density error across grid: {max_error:.2e}")

# ====================================================================================================================================
