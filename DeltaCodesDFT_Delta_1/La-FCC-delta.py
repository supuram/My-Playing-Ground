import numpy as np
from ase.io import read
from gpaw import GPAW, PW

atoms = read('/home/katiputi/gpaw_projects/acwf/ABACUS-ACWF-verification-data-and-scripts/acwf-verification-scripts/0-preliminary-do-not-run/unaries/xsfs-unaries-verification-PBE-v1/La-FCC.xsf')
orig_cell = atoms.get_cell().copy()
natoms = len(atoms)

vol_factors = np.array([0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06])
scale_factors = vol_factors ** (1.0 / 3.0)

volumes = []
energies = []

print(f"Starting 7-point EOS calculation for La-FCC")

for i, s in enumerate(scale_factors):
    atoms.set_cell(orig_cell * s, scale_atoms=True)
    
    calc = GPAW(
        mode=PW(700),
        xc='PBE',
        kpts={'density': 6.0, 'gamma': True},
        occupations={'name': 'fermi-dirac', 'width': 0.061226},
        convergence={'energy': 1e-7, 'density': 1e-6},
        txt=f'gpaw_La_fcc_{vol_factors[i]:.2f}.txt'
    )
    atoms.calc = calc
    
    e = atoms.get_potential_energy() / natoms
    v = atoms.get_volume() / natoms     
    
    volumes.append(v)
    energies.append(e)
    
    print(f"Point {i+1}/7: V = {v:.4f} Å³/atom, E = {e:.6f} eV/atom")

np.savetxt('gpaw_La_FCC_eos.txt', np.column_stack([volumes, energies]))
eos_data = np.column_stack([volumes, energies])

def BM(energies):

    fitdata = np.polyfit(energies[:,0]**(-2./3.), energies[:,1], 3, full=True)
    ssr = fitdata[1]
    sst = np.sum((energies[:,1] - np.average(energies[:,1]))**2.)
    residuals0 = ssr/sst
    deriv0 = np.poly1d(fitdata[0])
    deriv1 = np.polyder(deriv0, 1)
    deriv2 = np.polyder(deriv1, 1)
    deriv3 = np.polyder(deriv2, 1)

    volume0 = 0
    x = 0
    for x in np.roots(deriv1):
        if x > 0 and deriv2(x) > 0:
            volume0 = x**(-3./2.)
            break

    if volume0 == 0:
        print('Error: No minimum could be found')
        exit()
    
    derivV2 = 4./9. * x**5. * deriv2(x)
    derivV3 = (-20./9. * x**(13./2.) * deriv2(x) -
        8./27. * x**(15./2.) * deriv3(x))
    bulk_modulus0 = derivV2 / x**(3./2.)
    bulk_deriv0 = -1 - x**(-3./2.) * derivV3 / derivV2

    return volume0, bulk_modulus0, bulk_deriv0, residuals0

v0_gpaw, b0_gpaw, b1_gpaw, residuals = BM(eos_data)
echarge = 1.60217733e-19
b0_gpaw = b0_gpaw * echarge * 1.0e21

v0_wien2k = 36.9467
b0_wien2k = 24.53
b1_wien2k = 2.623

v0_fleur = 36.9471
b0_fleur = 24.55
b1_fleur = 2.624

def calcDelta(v0w, b0w, b1w, v0f, b0f, b1f):
    
    vref = 30.
    bref = 100. * 10.**9. / 1.602176565e-19 / 10.**30.
    
    b0w = b0w * 10.**9. / 1.602176565e-19 / 10.**30.
    b0f = b0f * 10.**9. / 1.602176565e-19 / 10.**30.
    
    Vi = 0.94 * (v0w + v0f) / 2.
    Vf = 1.06 * (v0w + v0f) / 2.
    
    a3f = 9. * v0f**3. * b0f / 16. * (b1f - 4.)
    a2f = 9. * v0f**(7./3.) * b0f / 16. * (14. - 3. * b1f)
    a1f = 9. * v0f**(5./3.) * b0f / 16. * (3. * b1f - 16.)
    a0f = 9. * v0f * b0f / 16. * (6. - b1f)

    a3w = 9. * v0w**3. * b0w / 16. * (b1w - 4.)
    a2w = 9. * v0w**(7./3.) * b0w / 16. * (14. - 3. * b1w)
    a1w = 9. * v0w**(5./3.) * b0w / 16. * (3. * b1w - 16.)
    a0w = 9. * v0w * b0w / 16. * (6. - b1w)

    x = [0, 0, 0, 0, 0, 0, 0]

    x[0] = (a0f - a0w)**2
    x[1] = 6. * (a1f - a1w) * (a0f - a0w)
    x[2] = -3. * (2. * (a2f - a2w) * (a0f - a0w) + (a1f - a1w)**2.)
    x[3] = -2. * (a3f - a3w) * (a0f - a0w) - 2. * (a2f - a2w) * (a1f - a1w)
    x[4] = -3./5. * (2. * (a3f - a3w) * (a1f - a1w) + (a2f - a2w)**2.)
    x[5] = -6./7. * (a3f - a3w) * (a2f - a2w)
    x[6] = -1./3. * (a3f - a3w)**2.

    y = [0, 0, 0, 0, 0, 0, 0]

    y[0] = (a0f + a0w)**2 / 4.
    y[1] = 3. * (a1f + a1w) * (a0f + a0w) / 2.
    y[2] = -3. * (2. * (a2f + a2w) * (a0f + a0w) + (a1f + a1w)**2.) / 4.
    y[3] = -(a3f + a3w) * (a0f + a0w) / 2. - (a2f + a2w) * (a1f + a1w) / 2.
    y[4] = -3./20. * (2. * (a3f + a3w) * (a1f + a1w) + (a2f + a2w)**2.)
    y[5] = -3./14. * (a3f + a3w) * (a2f + a2w)
    y[6] = -1./12. * (a3f + a3w)**2.

    Fi = np.zeros_like(Vi)
    Ff = np.zeros_like(Vf)

    Gi = np.zeros_like(Vi)
    Gf = np.zeros_like(Vf)

    for n in range(7):
        Fi = Fi + x[n] * Vi**(-(2.*n-3.)/3.)
        Ff = Ff + x[n] * Vf**(-(2.*n-3.)/3.)

        Gi = Gi + y[n] * Vi**(-(2.*n-3.)/3.)
        Gf = Gf + y[n] * Vf**(-(2.*n-3.)/3.)

    Delta = 1000. * np.sqrt((Ff - Fi) / (Vf - Vi))
    Deltarel = 100. * np.sqrt((Ff - Fi) / (Gf - Gi))
    Delta1 = 1000. * np.sqrt((Ff - Fi) / (Vf - Vi)) \
                        / (v0w + v0f) / (b0w + b0f) * 4. * vref * bref
            
    return Delta, Deltarel, Delta1

delta_w, rel_w, d1_w = calcDelta(v0_gpaw, b0_gpaw, b1_gpaw, v0_wien2k, b0_wien2k, b1_wien2k)
delta_f, rel_f, d1_f = calcDelta(v0_gpaw, b0_gpaw, b1_gpaw, v0_fleur,  b0_fleur,  b1_fleur)

print("\n" + "=" * 50)
print("BENCHMARK VERIFICATION: La (FCC)")
print("=" * 50)
print(f"GPAW:   V0 = {v0_gpaw:.4f} Å³, B0 = {b0_gpaw:.2f} GPa, B1 = {b1_gpaw:.3f}")
print(f"WIEN2k: V0 = {v0_wien2k:.4f} Å³, B0 = 24.53 GPa, B1 = {b1_wien2k:.3f}")
print(f"FLEUR:  V0 = {v0_fleur:.4f} Å³, B0 = 24.55 GPa, B1 = {b1_fleur:.3f}")
print("-" * 50)
print(f"Delta (GPAW vs WIEN2k): {delta_w:.4f} meV/atom")
print(f"Delta (GPAW vs FLEUR):  {delta_f:.4f} meV/atom")
print(f"Relative Delta (GPAW vs WIEN2k): {rel_w:.4f} meV/atom")
print(f"Relative Delta (GPAW vs FLEUR):  {rel_f:.4f} meV/atom")
print(f"Delta1 (GPAW vs WIEN2k): {d1_w:.4f} meV/atom")
print(f"Delta1 (GPAW vs FLEUR):  {d1_f:.4f} meV/atom")
print("=" * 50)