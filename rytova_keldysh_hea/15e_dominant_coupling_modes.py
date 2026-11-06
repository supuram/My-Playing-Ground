import numpy as np

# Load saved spectrum
data = np.load('data/HfS2_Odoped_HR_spectrum.npz')
energies_meV = data['energies_meV']
Q_k = data['Q_k']
S_k = data['S_k']
S_total = float(data['S_total'])

# 1. Franck-Condon relaxation energy (Stokes shift / 2)
E_relax_meV = np.sum(S_k * energies_meV)
''' E_relax_meV calculates the total lattice relaxation energy (E_rel or Franck–Condon relaxation energy), which is the total vibrational 
energy released into the crystal as heat when the atoms settle back into their equilibrium positions after the optical transition.
S_k = The average number of phonons created in mode k during the electronic transition.
S_k * energies_meV = The average vibrational energy dumped into that specific mode.
This means E_relax_meV is literally the elastic strain energy stored in the 81 springs of the crystal lattice at the instant of optical 
emission. Because the electron fell from the excited orbital in an attosecond, the atoms are momentarily frozen at the excited-state 
geometry (Delta_Q), holding 62.94 meV of mechanical potential energy that must be dissipated into lattice vibrations.'''
print(f"Total Huang-Rhys factor S:      {S_total:.4f}")
print(f"Franck-Condon relaxation energy: {E_relax_meV:.2f} meV ({E_relax_meV/1000.0:.4f} eV)")
print(f"Estimated Stokes Shift (2*E_rel): {2 * E_relax_meV:.2f} meV\n")

# 2. Identify top contributing phonon modes
top_indices = np.argsort(S_k)[::-1][:5]
print("Top 5 phonon modes coupling to the optical transition:")
print(f"{'Mode Index':^12} | {'Energy (meV)':^14} | {'Freq (cm^-1)':^14} | {'S_k':^10} | {'% of Total S':^12}")
print("-" * 72)
for idx in top_indices:
    e_mev = energies_meV[idx]
    cm1 = e_mev * 8.06554
    sk = S_k[idx]
    pct = (sk / S_total) * 100
    print(f"{idx:^12d} | {e_mev:^14.2f} | {cm1:^14.1f} | {sk:^10.4f} | {pct:^11.1f}%")
    
''' Output
Total Huang-Rhys factor S:      2.5011
Franck-Condon relaxation energy: 62.08 meV (0.0621 eV)
Estimated Stokes Shift (2*E_rel): 124.17 meV

Top 5 phonon modes coupling to the optical transition:
 Mode Index  |  Energy (meV)  |  Freq (cm^-1)  |    S_k     | % of Total S
------------------------------------------------------------------------
     19      |     15.83      |     127.7      |   0.7128   |    28.5    %
     55      |     33.50      |     270.2      |   0.5369   |    21.5    %
     23      |     16.29      |     131.4      |   0.4292   |    17.2    %
     59      |     34.53      |     278.5      |   0.2340   |     9.4    %
     78      |     47.79      |     385.5      |   0.0881   |     3.5    %
'''
