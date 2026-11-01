import numpy as np
from ase.build import bulk
from gpaw import GPAW, PW
from gpaw.upf import UPFSetupData

# 1. Parse the UPF setup
setup_data = UPFSetupData('Si.upf')

# 2. Build primitive diamond Silicon (a = 5.43 Å)
atoms = bulk('Si', 'diamond', a=5.43)

# Apply a slight shear distortion so off-diagonal stress components are non-zero
cell = atoms.get_cell()
cell[0, 1] += 0.05
cell[1, 2] += 0.05
atoms.set_cell(cell, scale_atoms=True)


def get_calculator():
  return GPAW(
      mode=PW(350),
      xc='PBE',
      kpts=(4, 4, 4),
      setups={'Si': setup_data},
      convergence={'energy': 1e-8, 'density': 1e-7},
      txt=None,
  )


# 3. Analytical Stress (Voigt order: xx, yy, zz, yz, xz, xy)
atoms.calc = get_calculator()
stress_analytical = atoms.get_stress()  # eV / Å^3
vol = atoms.get_volume()

print("Analytical Stress (eV/Å^3) [xx, yy, zz, yz, xz, xy]:")
print(np.round(stress_analytical, 6))

# 4. Numerical Stress via Central Differences
# Strain tensor in Voigt ordering
delta = 1e-3  # dimensionless strain step
voigt_pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
stress_numerical = np.zeros(6)

orig_cell = atoms.get_cell().copy()
orig_positions = atoms.get_positions().copy()

print("\nCalculating numerical stress via finite strains...")
for idx, (i, j) in enumerate(voigt_pairs):
  energies = []
  for s in [-delta, delta]:
    eps = np.zeros((3, 3))
    if i == j:
      eps[i, i] = s
    else:
      eps[i, j] = s / 2.0
      eps[j, i] = s / 2.0

    # Deform unit cell and scale atomic positions affine-style
    deformed_cell = np.dot(orig_cell, (np.eye(3) + eps).T)
    scale_mat = np.dot(deformed_cell, np.linalg.inv(orig_cell))
    deformed_pos = np.dot(orig_positions, scale_mat.T)

    atoms_distorted = atoms.copy()
    atoms_distorted.set_cell(deformed_cell, scale_atoms=False)
    atoms_distorted.set_positions(deformed_pos)
    atoms_distorted.calc = get_calculator()

    e = atoms_distorted.get_potential_energy()
    energies.append(e)

  # sigma = (1 / V) * dE / d(eps)
  dE = (energies[1] - energies[0]) / (2.0 * delta)
  stress_numerical[idx] = dE / vol

print("\n--- Stress Comparison (eV/Å^3) ---")
diff = stress_analytical - stress_numerical
components = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']
for name, sa, sn, d in zip(components, stress_analytical, stress_numerical, diff):
  print(
      f"sigma_{name}: analytical = {sa:10.6f}, numerical = {sn:10.6f}, diff ="
      f" {d:10.6f}"
  )

max_err = np.max(np.abs(diff))
print(f"\nMax absolute difference: {max_err:.6f} eV/Å^3")