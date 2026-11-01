import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from gpaw import GPAW, PW, FermiDirac
from gpaw.response.g0w0 import G0W0
import gc

# ==========================================================
# 1. Define the Strain Grid
# ==========================================================
# -2% (Compressive) to +2% (Tensile)
strain_percents = [-2.0, -1.0, 0.0, 1.0, 2.0]
pbe_gaps = []
gw_gaps = []

# Load the pristine relaxed structure
pristine_atoms = read('structures/HfS2_relaxed.traj')
original_cell = pristine_atoms.get_cell()
original_positions = pristine_atoms.get_positions()

for s in strain_percents:
    print(f"\n{'='*50}")
    print(f"Applying {s}% Biaxial Strain...")
    print(f"{'='*50}")
    
    # ==========================================================
    # 2. Apply Biaxial Strain safely to a 2D Material
    # ==========================================================
    scale = 1.0 + (s / 100.0)
    atoms = pristine_atoms.copy()
    
    new_cell = original_cell.copy()
    new_cell[0] *= scale  # Stretch/Compress X-vector
    new_cell[1] *= scale  # Stretch/Compress Y-vector
    
    # Scale atoms to new cell, but restore the original Z-thickness
    atoms.set_cell(new_cell, scale_atoms=True)
    new_positions = atoms.get_positions()
    new_positions[:, 2] = original_positions[:, 2]  # Snap Z back to pristine
    atoms.set_positions(new_positions)
    
    # ==========================================================
    # 3. Ground State PBE (Need 250 bands for GW)
    # ==========================================================
    calc = GPAW(
        mode=PW(600),
        xc='PBE',
        nbands=250,
        kpts={'density': 6.0, 'gamma': True},
        occupations=FermiDirac(0.01),
        txt=f'data/HfS2_strain_{s}_scf.txt'
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    
    # Extract PBE Gap (VBM=11, CBM=12)
    vbm_pbe = calc.get_eigenvalues(kpt=0)[10]
    cbm_pbe = calc.get_eigenvalues(kpt=0)[11]
    pbe_gap = cbm_pbe - vbm_pbe
    pbe_gaps.append(pbe_gap)
    
    gpw_file = f'data/HfS2_strain_{s}.gpw'
    calc.write(gpw_file, mode='all')
    
    # ==========================================================
    # 4. G0W0 Quasiparticle Correction
    # ==========================================================
    gw = G0W0(
        calc=gpw_file,
        filename=f'data/HfS2_strain_{s}_gw',
        bands=(10, 14),
        nbands=250,
        kpts=[0],
        ecut=50,
        truncation='2D',
        eta=0.1
    )
    results = gw.calculate()
    
    vbm_gw = results['qp'][0, 0, 11 - 10]
    cbm_gw = results['qp'][0, 0, 12 - 10]
    gw_gap = cbm_gw - vbm_gw
    gw_gaps.append(gw_gap)
    
    del calc
    del gw
    del results
    gc.collect()
    
    print(f"\nResult for {s}% Strain: PBE Gap = {pbe_gap:.3f} eV | GW Gap = {gw_gap:.3f} eV")

# ==========================================================
# 5. Plot the Band Gap Engineering Curve
# ==========================================================
plt.figure(figsize=(8, 5))
plt.plot(strain_percents, pbe_gaps, 'k--o', label='PBE Gap', alpha=0.7)
plt.plot(strain_percents, gw_gaps, 'r-o', linewidth=2.5, label='True G0W0 Gap')

plt.xlabel('Biaxial Strain (%)', fontsize=12)
plt.ylabel('Fundamental Band Gap (eV)', fontsize=12)
plt.title('Strain Engineering of 1T-HfS$_2$ Band Gap', fontsize=14)
plt.legend(loc='best')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('structures/HfS2_Strain_Engineering.png', dpi=300)
print("\nStrain calculation complete! Saved plot to 'structures/HfS2_Strain_Engineering.png'")

summary_file = 'data/HfS2_Strain_Summary.txt'
with open(summary_file, 'w') as f:
    f.write("Strain(%) | PBE_Gap(eV) | GW_Gap(eV)\n")
    f.write("-" * 40 + "\n")
    for i in range(len(strain_percents)):
        f.write(f"{strain_percents[i]:9.1f} | {pbe_gaps[i]:11.3f} | {gw_gaps[i]:10.3f}\n")

print(f"Saved numerical data to {summary_file}")