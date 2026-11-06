import matplotlib.pyplot as plt
from gpaw import GPAW, PW
from scipy.optimize import curve_fit

# Physical constants
HARTREE_TO_EV = 27.2114
BOHR_TO_ANG = 0.529177
m0 = 1.0  # Electron rest mass in atomic units

# 1. Load the pristine relaxed structure
calc = GPAW('structures/HfS2_relaxed.gpw')
atoms = calc.get_atoms()

# 2. Set up a fine k-point path strictly around the Gamma point
# We take 21 points going slightly towards M, and slightly towards K
kpts_path = atoms.cell.bandpath('MGK', npoints=41)

# 3. Ground State PBE Calculation
print("Starting SCF calculation for band structure...")
calc = GPAW(
    mode=PW(500),
    xc='PBE',
    kpts={'density': 6.0, 'gamma': True},
    txt='data/HfS2_eff_mass_scf.txt'
)
atoms.calc = calc
atoms.get_potential_energy()

# 4. Calculate the Band Structure along the path
print("Calculating band energies around Gamma...")
calc = calc.fixed_density(kpts=kpts_path, symmetry='off')

bs = calc.band_structure()
energies = bs.energies[0]  # Spin 0
k_distances = bs.path.get_linear_kpoint_axis()[0]

# Gamma point is exactly in the middle of our 41-point path (index 20), Hf = 12 valence e, 2S = 6 x 2 = 12  valence e, HfS2 = 24 valence e
gamma_idx = 20
vbm_band = 11
cbm_band = 12

# Extract k-points and energies near Gamma (e.g., 5 points on each side)
k_fit = k_distances[gamma_idx-5 : gamma_idx+6]
e_hole = energies[gamma_idx-5 : gamma_idx+6, vbm_band]
e_elec = energies[gamma_idx-5 : gamma_idx+6, cbm_band]

# 5. Fit the Parabola: E(k) = E0 + (\hbar^2 / 2m*) * k^2
# In atomic units: m* = 1 / (d^2E / dk^2)
def parabola(k, E0, a):
    return E0 + a * (k - k_distances[gamma_idx])**2

# Fit holes (Valence Band)
popt_h, _ = curve_fit(parabola, k_fit, e_hole)
a_h = popt_h[1] 
# Convert fitting parameter 'a' to effective mass
m_hole = (HARTREE_TO_EV * BOHR_TO_ANG**2) / (2 * a_h)

# Fit electrons (Conduction Band)
popt_e, _ = curve_fit(parabola, k_fit, e_elec)
a_e = popt_e[1]
m_elec = (HARTREE_TO_EV * BOHR_TO_ANG**2) / (2 * a_e)

print("\n" + "="*40)
print("EFFECTIVE MASS RESULTS (in units of m0)")
print("="*40)
print(f"Hole effective mass (m_h*)     = {m_hole:.3f} m0")
print(f"Electron effective mass (m_e*) = {m_elec:.3f} m0")

# Optional: Plot to verify the fit
plt.figure(figsize=(6, 4))
plt.plot(k_fit, e_elec, 'bo', label='DFT Conduction Band')
plt.plot(k_fit, parabola(k_fit, *popt_e), 'r-', label='Parabolic Fit (Electron)')

plt.plot(k_fit, e_hole, 'go', label='DFT Valence Band')
plt.plot(k_fit, parabola(k_fit, *popt_h), 'r-', label='Parabolic Fit (Hole)')

plt.title("Effective Mass Parabolic Fit around $\Gamma$")
plt.xlabel("k-path distance (1/Å)")
plt.ylabel("Energy (eV)")
plt.legend()
plt.tight_layout()
plt.savefig('structures/HfS2_Effective_Mass_Fit.png')
print("Saved fit plot to 'structures/HfS2_Effective_Mass_Fit.png'")

summary_file = 'data/HfS2_Effective_Mass.txt'
with open(summary_file, 'w') as f:
    f.write("HfS2 Effective Mass Results (Units of m0)\n")
    f.write("-" * 45 + "\n")
    f.write(f"Hole effective mass (m_h*)     : {m_hole:.3f}\n")
    f.write(f"Electron effective mass (m_e*) : {m_elec:.3f}\n")

print(f"Saved numerical data to {summary_file}")

'''
Output

Hole effective mass (m_h*)     = -1.191 m0
Electron effective mass (m_e*) = 2.569 m0

The hole mass is -1.191 m0 because the valence band curves downward. In semiconductor physics, the hole mass is defined by the 
negative of that curvature, so your actual hole effective mass is 1.191 m0.
'''