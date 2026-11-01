import numpy as np

# 1. Define the 3D Cartesian Box [-L/2, L/2]^3
L = 10.0          # Box width (e.g. 10 Bohr)
N_grid = 64       # 64 points per dimension -> 64^3 grid points
x_1d = np.linspace(-L/2, L/2, N_grid, endpoint=False)   # shape of x_1d is (64,)
dx = x_1d[1] - x_1d[0]
d3r = dx ** 3     # 3D volume element d^3r = dx * dy * dz

# Augmentation cutoff radius
r_c = 2.0

''' Arbitrary Atomic Position R and Quantum Numbers (l, m)
# Set ANY arbitrary atomic site R in the box: '''
R_atom = np.array([2.0, 1.0, 2.0])  
Z_nuc = 1.0                         # Nuclear charge
l = 0                               # Angular momentum l
m = 0                               # Magnetic quantum number m

r_c_gauss = r_c / 2.0               # Gaussian decay length r_c (Eq. 23)

# Create 3D coordinate meshes (X, Y, Z)
X, Y, Z = np.meshgrid(x_1d, x_1d, x_1d, indexing='ij')   # X, Y, Z - all has shape of (64, 64, 64)

''' Coordinate vectors (r - R) across the 3D grid
# Compute shifted coordinates: (r - R) '''
dX = X - R_atom[0]
dY = Y - R_atom[1]
dZ = Z_grid = Z - R_atom[2]  # Z is already the coordinate grid from meshgrid

# Distance field: |r - R|
r_R = np.sqrt(dX**2 + dY**2 + dZ**2)   # r_R has shape (64, 64, 64)
r_R_safe = np.where(r_R == 0.0, 1e-12, r_R)

# Unit vector direction: (r - R) / |r - R|
unit_x = dX / r_R_safe
unit_y = dY / r_R_safe
unit_z = dZ / r_R_safe

''' Real Spherical Harmonic Y_lm( \hat{r - R} ) '''
def real_spherical_harmonic(l_val, m_val, ux, uy, uz):
    """Computes real spherical harmonic Y_lm from directional unit vectors."""
    if l_val == 0 and m_val == 0:
        return np.full_like(ux, 1.0 / np.sqrt(4.0 * np.pi))
    
    elif l_val == 1 and m_val == 0:     # p_z
        return np.sqrt(3.0 / (4.0 * np.pi)) * uz
    elif l_val == 1 and m_val == 1:     # p_x
        return np.sqrt(3.0 / (4.0 * np.pi)) * ux
    elif l_val == 1 and m_val == -1:    # p_y
        return np.sqrt(3.0 / (4.0 * np.pi)) * uy
    
    # d orbitals: real tesseral harmonics
    elif l_val == 2 and m_val == 0:       # d_z2
        return np.sqrt(5.0 / (16.0 * np.pi)) * (3.0 * uz**2 - 1.0)

    elif l_val == 2 and m_val == 1:       # d_xz
        return np.sqrt(15.0 / (4.0 * np.pi)) * ux * uz

    elif l_val == 2 and m_val == -1:      # d_yz
        return np.sqrt(15.0 / (4.0 * np.pi)) * uy * uz

    elif l_val == 2 and m_val == 2:      # d_x2-y2
        return np.sqrt(15.0 / (16.0 * np.pi)) * (ux**2 - uy**2)

    elif l_val == 2 and m_val == -2:     # d_xy
        return np.sqrt(15.0 / (4.0 * np.pi)) * ux * uy
    else:
        raise NotImplementedError("Higher l channels can be added as needed.")

Y_L = real_spherical_harmonic(l, m, unit_x, unit_y, unit_z)

# 2. Radial shapes
# phi_ae wiggles violently inside r_c, smooth outside
phi_ae_radial = np.where(r_R < r_c, 
                         np.sin(4 * np.pi * r_R) * np.exp(-r_R), 
                         np.exp(-r_R))

# phi_ps is smooth everywhere and matches phi_ae for r >= r_c
phi_ps_radial = np.exp(-r_R)

# Form the full 3D partial waves: phi(r) * Y_L
phi_ae_3d = phi_ae_radial * Y_L
phi_ps_3d = phi_ps_radial * Y_L

# 3. Projector p_tilde in 3D: localized strictly inside r_c
p_raw_radial = np.where(r_R < r_c, (r_c - r_R)**2, 0.0)
p_raw_3d = p_raw_radial * Y_L

# Normalization condition: <p_tilde | phi_ps> = 1 (Eq. 7 in 3D)
overlap_3d = np.sum(p_raw_3d * phi_ps_3d) * d3r
p_tilde_3d = p_raw_3d / overlap_3d

'''print(f"3D <p_tilde | phi_ps> = {np.sum(p_tilde_3d * phi_ps_3d) * d3r:.6f}") '''

# 4. Define a smooth 3D pseudo wave function psi_ps
psi_ps_3d = 1.5 * np.exp(-r_R) * Y_L

# 5. Projection coefficient c = <p_tilde | psi_ps> (Eq. 6)
c = np.sum(p_tilde_3d * psi_ps_3d) * d3r
'''print(f"3D Projection coefficient c = {c:.4f}")'''

# 6. Reconstruct true All-Electron 3D wave function (Eq. 9)
psi_ae_3d = psi_ps_3d + (phi_ae_3d - phi_ps_3d) * c

# ===================================================================================================================================

# --- Equation 30: Overlap in 3D ---
delta_S_3d = np.sum(phi_ae_3d**2) * d3r - np.sum(phi_ps_3d**2) * d3r
norm_direct_3d = np.sum(psi_ae_3d**2) * d3r
norm_paw_3d = np.sum(psi_ps_3d**2) * d3r + c * delta_S_3d * c

'''print(f"3D Direct AE norm <Psi|Psi>        = {norm_direct_3d:.8f}")
print(f"3D PAW operator   <psi_ps|O|psi_ps> = {norm_paw_3d:.8f}")'''

# ===================================================================================================================================

# --- Equation 11: 3D Potential Operator ---
V_3d = -2.0 / (r_R + 0.1)
val_direct_3d = np.sum(psi_ae_3d * V_3d * psi_ae_3d) * d3r
val_smooth_3d = np.sum(psi_ps_3d * V_3d * psi_ps_3d) * d3r
delta_V_3d = (np.sum(phi_ae_3d * V_3d * phi_ae_3d) * d3r - 
              np.sum(phi_ps_3d * V_3d * phi_ps_3d) * d3r)
val_paw_3d = val_smooth_3d + c * delta_V_3d * c

'''print(f"3D Direct AE <Psi|V|Psi>            = {val_direct_3d:.8f}")
print(f"3D PAW operator <psi_ps|V_tilde|psi_ps> = {val_paw_3d:.8f}")'''

# ===================================================================================================================================

# --- Equation 14: 3D Densities ---
n_ae_3d = psi_ae_3d ** 2
n_tilde_3d = psi_ps_3d ** 2
n_1_3d = c * (phi_ae_3d ** 2) * c
n_tilde_1_3d = c * (phi_ps_3d ** 2) * c
n_paw_3d = n_tilde_3d + n_1_3d - n_tilde_1_3d

max_err_3d = np.max(np.abs(n_ae_3d - n_paw_3d))
'''print(f"3D Max density error                = {max_err_3d:.2e}")'''

# ===================================================================================================================================

''' Nuclear Point Charge Density n^Z(r) on the 3D Grid (Eq. 18 & 24)
n^Z(r) is -Z * delta(r - R) '''
n_Z = np.zeros_like(r_R)   # n_Z has shape (64, 64, 64)
# Locate nearest grid point to R_atom
i, j, k = np.unravel_index(np.argmin(r_R), r_R.shape)
n_Z[i, j, k] = -Z_nuc / d3r  # delta function normalized such that \int n^Z d^3r = -Z

''' Multipole Moment Q_RL (Equation 24)
Q_RL = \int d^3r |r - R|^l * [ n^1(r) + n^Z(r) - n_tilde^1(r) ] * Y_L^*(r - R) '''
integrand_Q = (r_R ** l) * (n_1_3d + n_Z - n_tilde_1_3d) * Y_L
Q_RL = np.sum(integrand_Q) * d3r

# print(f"Computed Q_RL for site R={R_atom}, l={l}, m={m}: {Q_RL:.6f}")

''' Generalized Gaussian g_RL(r) (Equation 23)
Unnormalized shape: |r - R|^l * Y_L * exp( -(|r - R| / r_c)^2 ) '''
g_unnorm = (r_R ** l) * Y_L * np.exp(-(r_R / r_c_gauss)**2)

''' Determine normalization constant C_l such that multipole moment of g_RL is 1.0:
\int d^3r [ |r - R|^l * Y_L ] * g_RL(r) = 1.0 '''
g_multipole_integral = np.sum((r_R ** l) * Y_L * g_unnorm) * d3r
C_l = 1.0 / g_multipole_integral

g_RL = C_l * g_unnorm

''' Verification of Eq. (23) normalization: '''
check_norm = np.sum((r_R ** l) * Y_L * g_RL) * d3r
# print(f"Normalization check for g_RL (must be 1.0): {check_norm:.6f}")

''' Compensation Charge Density \hat{n}(r) (Equation 22)
\hat{n}_R(r) = g_RL(r) * Q_RL '''
n_hat = g_RL * Q_RL

''' 8. Exact Check: Vanishing Multipole Moment (Text below Eq. 21)
The combined on-center charge distribution is:
rho_center(r) = [ n^1(r) + n^Z(r) ] - [ n_tilde^1(r) + \hat{n}(r) ] '''
rho_center = (n_1_3d + n_Z) - (n_tilde_1_3d + n_hat)

''' Multipole moment of rho_center:
\int d^3r |r - R|^l * rho_center(r) * Y_L(r - R) '''
residual_multipole = np.sum((r_R ** l) * rho_center * Y_L) * d3r

# print(f"Residual Multipole of [ (n^1 + n^Z) - (n_tilde^1 + n_hat) ] = {residual_multipole:.2e}")

# ===================================================================================================================================

''' Kinetic Energy Decomposition (Eqs. 19-21, 33) '''

''' Reciprocal grid wavevectors G_x, G_y, G_z 
In Fourier space on our periodic grid, the Laplacian acts diagonally , laplacian = -G2 '''
G_1d = 2.0 * np.pi * np.fft.fftfreq(N_grid, d=dx)   # G_1d has shape (64,)
GX, GY, GZ = np.meshgrid(G_1d, G_1d, G_1d, indexing='ij')  # GX has shape (64, 64, 64)
G2 = GX**2 + GY**2 + GZ**2

def kinetic_operator(field_3d):
    """Evaluates (-1/2 \nabla^2) on a 3D field using FFT."""
    field_G = np.fft.fftn(field_3d)
    T_field_G = 0.5 * G2 * field_G
    return np.real(np.fft.ifftn(T_field_G))

# 1. Direct AE Kinetic Energy: <Psi_ae | -1/2 \nabla^2 | Psi_ae>
T_psi_ae = kinetic_operator(psi_ae_3d)
E_kin_direct = np.sum(psi_ae_3d * T_psi_ae) * d3r

# 2. Smooth Pseudo Kinetic Energy (Eq. 19): <psi_ps | -1/2 \nabla^2 | psi_ps>
T_psi_ps = kinetic_operator(psi_ps_3d)
E_kin_smooth = np.sum(psi_ps_3d * T_psi_ps) * d3r

# 3. AE and PS Partial Wave Kinetic Matrix Elements (Eqs. 20-21, 33)
T_phi_ae = kinetic_operator(phi_ae_3d)
T_phi_ps = kinetic_operator(phi_ps_3d)

T_11_ae = np.sum(phi_ae_3d * T_phi_ae) * d3r
T_11_ps = np.sum(phi_ps_3d * T_phi_ps) * d3r

delta_T = T_11_ae - T_11_ps

# 4. PAW Evaluated Kinetic Energy (Eq. 33)
# E_kin_paw = \tilde{T} + \sum_{ij} c_i * (T^1_ij - \tilde{T}^1_ij) * c_j
E_kin_paw = E_kin_smooth + (c * delta_T * c)

print(f"3D Direct AE Kinetic Energy  <Psi|-1/2 del^2|Psi>    = {E_kin_direct:.8f} Ha")
print(f"3D Smooth Pseudo Kinetic     <psi_ps|-1/2 del^2|psi_ps> = {E_kin_smooth:.8f} Ha")
print(f"3D PAW Kinetic Energy        <psi_ps|T_tilde|psi_ps>    = {E_kin_paw:.8f} Ha")
print(f"Kinetic Energy Error |Direct - PAW|                     = {abs(E_kin_direct - E_kin_paw):.2e} Ha")