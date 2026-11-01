import matplotlib.pyplot as plt
from gpaw import GPAW, PW
from scipy.optimize import curve_fit

# Physical constants
HARTREE_TO_EV = 27.2114
BOHR_TO_ANG = 0.529177
m0 = 1.0  # Electron rest mass in atomic units

calc = GPAW('structures/HfS2_Odoped_relaxed.gpw')
atoms = calc.get_atoms()

# 2. Set up a fine k-point path strictly around the Gamma point
# We take 21 points going slightly towards M, and slightly towards K
kpts_path = atoms.cell.bandpath('MGK', npoints=41)

# 4. Calculate the Band Structure along the path
print("Calculating band energies around Gamma...")
calc = calc.fixed_density(kpts=kpts_path, symmetry='off', txt='data/HfS2_eff_mass_Odoped_fixed_density.txt')

bs = calc.band_structure()
energies = bs.energies[0]  # Spin 0
k_distances = bs.path.get_linear_kpoint_axis()[0]

# 3x3 supercell. So 9 Hf x 12 = 108 valence e + 17 S x 6 = 102 valence e + 1 O x 6 = 6 valence e = 216 valence e
gamma_idx = 20
vbm_band = 107
cbm_band = 108

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
plt.savefig('structures/HfS2_Odoped_Effective_Mass_Fit.png')
print("Saved fit plot to 'structures/HfS2_Odoped_Effective_Mass_Fit.png'")

summary_file = 'data/HfS2_Odoped_Effective_Mass_summary.txt'
with open(summary_file, 'w') as f:
    f.write("HfS2 Effective Mass Results (Units of m0)\n")
    f.write("-" * 45 + "\n")
    f.write(f"Hole effective mass (m_h*)     : {m_hole:.3f}\n")
    f.write(f"Electron effective mass (m_e*) : {m_elec:.3f}\n")

print(f"Saved numerical data to {summary_file}")