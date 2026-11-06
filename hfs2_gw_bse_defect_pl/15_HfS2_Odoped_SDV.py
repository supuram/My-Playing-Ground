import os
import numpy as np
from ase.io import read
from gpaw import GPAW
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.mom import prepare_mom_calculation
from gpaw.directmin.tools import excite
from ase.optimize import BFGS

# ---------------------------------------------------------
# File Paths
# ---------------------------------------------------------
gs_traj_file = 'structures/HfS2_Odoped_relaxed.traj'
gs_gpw_file  = 'data/HfS2_Odoped_GS_FDPWETDM.gpw'
es_traj_file = 'structures/HfS2_Odoped_ES_relaxed.traj'
es_gpw_file  = 'data/HfS2_Odoped_ES_relaxed.gpw'
bfgs_pckl    = 'data/HfS2_Odoped_ES_bfgs.pckl'

# 1. Load relaxed ground-state atomic coordinates
atoms_gs = read(gs_traj_file)
R_GS = atoms_gs.get_positions()

# ---------------------------------------------------------
# 2. Ground-State Checkpoint Loader / Single-Point Run
# ---------------------------------------------------------
if os.path.exists(gs_gpw_file):
    print(f"Loading pre-converged GS from checkpoint: {gs_gpw_file}")
    calc = GPAW(gs_gpw_file, txt='data/HfS2_Odoped_GS_FDPWETDM.txt')
    atoms = calc.get_atoms()
    E_gs = atoms.get_potential_energy()
    # Load cached forces from disk to avoid on-the-fly re-evaluation
    force_file = 'data/HfS2_Odoped_Force_GS.npy'
    if os.path.exists(force_file):
        F_gs = np.load(force_file)
    else:
        F_gs = atoms.get_forces()
        np.save(force_file, F_gs)
else:
    print("Running initial spin-polarized Gamma-point GS calculation...")
    atoms = atoms_gs.copy()
    calc = GPAW(
        mode={'name': 'pw', 'ecut': 500},
        xc='PBE',
        nbands=160,
        kpts=(1, 1, 1),                        # 1 k-point: fast and avoids multi-k MOM errors
        occupations={'name': 'fixed-uniform'}, # Strict integer occupations: avoids LUMO hunt errors
        mixer={'backend': 'no-mixing'},        # Required for ETDM
        eigensolver=FDPWETDM(converge_unocc=True),
        spinpol=True,                          # Open-shell capability
        symmetry='off',
        txt='data/HfS2_Odoped_GS_FDPWETDM.txt'
    )
    atoms.calc = calc
    E_gs = atoms.get_potential_energy()
    F_gs = atoms.get_forces()

    # Save full ground-state wavefunctions for instant reuse
    calc.write(gs_gpw_file, mode='all')
    np.save('data/HfS2_Odoped_Force_GS.npy', F_gs)
    print(f"Ground-state checkpoint saved to {gs_gpw_file}")

# ---------------------------------------------------------
# 3. Setup Excited State (DO-MOM)
# ---------------------------------------------------------
print("Switching ETDM to excited-state tracking...")
calc.set(eigensolver=FDPWETDM(excited_state=True,
                              converge_unocc=False,
                              momevery=10,
                              max_step_inner_loop=0.2,
                              maxiter_inner_loop=20))

f_sn = excite(calc, 0, 0, (0, 0))
prepare_mom_calculation(calc, atoms, f_sn)

# ---------------------------------------------------------
# 4. Resume Relaxation If Trajectory Exists
# ---------------------------------------------------------
if os.path.exists(es_traj_file) and os.path.getsize(es_traj_file) > 0:
    try:
        last_image = read(f'{es_traj_file}@-1')
        atoms.set_positions(last_image.get_positions())
        print(f"Resuming relaxation from existing geometry in {es_traj_file}")
    except Exception as e:
        print(f"Starting relaxation from R_GS: {e}")

# ---------------------------------------------------------
# 5. Geometry Optimization (BFGS)
# ---------------------------------------------------------
print("Starting Excited-State Geometry Optimization...")
opt = BFGS(
    atoms,
    trajectory=es_traj_file,
    restart=bfgs_pckl,
    logfile='data/HfS2_Odoped_ES_BFGS.log'
)
opt.run(fmax=0.02)

# ---------------------------------------------------------
# 6. Extract Outputs & Checkpoint Excited State
# ---------------------------------------------------------
R_ES = atoms.get_positions()
Delta_R = R_ES - R_GS
E_es = atoms.get_potential_energy()
F_es = atoms.get_forces()

# Save displacement vector for MatterSim
np.save('data/HfS2_Odoped_Delta_R.npy', Delta_R)

# Save forces and scalar energies
np.save('data/HfS2_Odoped_Force_ES.npy', F_es)
np.savez(
    'data/HfS2_Odoped_energies.npz',
    E_gs=E_gs,
    E_es=E_es,
    E_ex=E_es - E_gs,
    fmax_final=np.max(np.sqrt(np.sum(F_es**2, axis=1)))
)

# Save full excited-state wavefunctions
calc.write(es_gpw_file, mode='all')

print("Excited-state relaxation complete.")
print(f"Excitation energy: {E_es - E_gs:.4f} eV")
print(f"Max residual force: {np.max(np.sqrt(np.sum(F_es**2, axis=1))):.4f} eV/Å")
print(f"Max atomic displacement: {np.max(np.abs(Delta_R)):.4f} Å")
print(f"Displacement saved to data/HfS2_Odoped_Delta_R.npy")

'''
Excitation energy: 1.3109 eV
Max residual force: 0.0184 eV/Å
Max atomic displacement: 0.0486 Å
'''
