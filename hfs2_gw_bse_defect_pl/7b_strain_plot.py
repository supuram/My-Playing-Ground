import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW
import pickle

strain_percents = [-2.0, -1.0, 0.0, 1.0, 2.0]
pbe_gaps = []
gw_gaps = []

print("Extracting gaps from saved files...")

for s in strain_percents:
    # 1. Extract PBE Gaps directly from the saved .gpw files
    calc = GPAW(f'data/HfS2_strain_{s}.gpw')
    vbm_pbe = calc.get_eigenvalues(kpt=0)[10]
    cbm_pbe = calc.get_eigenvalues(kpt=0)[11]
    pbe_gap = cbm_pbe - vbm_pbe
    pbe_gaps.append(pbe_gap)
    
    # 2. Extract GW Gaps from the saved G0W0 .pckl files
    pckl_file = f'data/HfS2_strain_{s}_gw_results_GW.pckl'
    with open(pckl_file, 'rb') as f:
        results = pickle.load(f)
        # results['qp'] is a numpy array, shape is typically (spin, kpt, band)
        vbm_gw = results['qp'][0, 0, 11 - 10]
        cbm_gw = results['qp'][0, 0, 12 - 10]
        gw_gap = cbm_gw - vbm_gw
        gw_gaps.append(gw_gap)
        
    print(f"Strain {s}%: PBE Gap = {pbe_gap:.3f} eV | GW Gap = {gw_gap:.3f} eV")

# 3. Regenerate the Summary Text File
summary_file = 'data/HfS2_Strain_Summary.txt'
with open(summary_file, 'w') as f:
    f.write("Strain(%) | PBE_Gap(eV) | GW_Gap(eV)\n")
    f.write("-" * 40 + "\n")
    for i in range(len(strain_percents)):
        f.write(f"{strain_percents[i]:9.1f} | {pbe_gaps[i]:11.3f} | {gw_gaps[i]:10.3f}\n")
print(f"\nSaved complete numerical data to {summary_file}")

# 4. Regenerate the Plot
plt.figure(figsize=(8, 5))
plt.plot(strain_percents, pbe_gaps, 'k--o', label='PBE Gap', alpha=0.7)
plt.plot(strain_percents, gw_gaps, 'r-o', linewidth=2.5, label='True G0W0 Gap')

plt.xlabel('Biaxial Strain (%)', fontsize=12)
plt.ylabel('Fundamental Band Gap (eV)', fontsize=12)
plt.title('Strain Engineering of 1T-HfS$_2$ Band Gap', fontsize=14)
plt.legend(loc='best')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('structures/HfS2_Strain_engineering.png', dpi=300)
print("Saved complete plot.")