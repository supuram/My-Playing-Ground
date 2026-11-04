import numpy as np
from ase.build import molecule
from ase.units import Bohr, Ha
from gpaw.core import UGDesc, PWDesc
from gpaw.solvation.cavity import EffectivePotentialCavity, Power12Potential
from gpaw.solvation.dielectric import LinearDielectric
from gpaw.solvation.poisson import PWSolvationPoissonSolver

def test_poisson_solve_co2():
    # 1. Setup CO2 in a periodic box
    atoms = molecule('CO2')
    box_size = 10.0  # Å
    atoms.set_cell([box_size, box_size, box_size])
    atoms.center()

    # 2. Create Real-Space (grid) and Plane-Wave (pw) descriptors
    cell_cv = atoms.get_cell() / Bohr
    grid = UGDesc(cell=cell_cv, size=(40, 40, 40))
    pw = PWDesc(ecut=200 / Ha, cell=cell_cv)
    legacy_gd = grid._gd

    # 3. Build the Water Cavity around CO2
    pot = Power12Potential(u0=0.180)
    cavity = EffectivePotentialCavity(effective_potential=pot, temperature=298.15)
    cavity.set_grid_descriptor(legacy_gd)
    cavity.allocate()
    cavity.update_atoms(atoms, log=lambda *args: None)
    cavity.update(atoms, density=None)

    # 4. Build the Linear Dielectric (Water: eps = 78.4)
    eps = LinearDielectric(epsinf=78.4)
    eps.set_grid_descriptor(legacy_gd)
    eps.allocate()
    eps.update(cavity)

    # 5. Initialize the Poisson Solver
    solver = PWSolvationPoissonSolver(
        pw=pw, 
        grid=grid, 
        dielectric=eps, 
        rtol=1e-5,       # Tolerance to stop solving
        maxiter=200      # Max iterations
    )

    # ---------------------------------------------------------
    # PREPARING THE 3 ARGUMENTS FOR THE SOLVER
    # ---------------------------------------------------------

    # A) rhot_g: Create a mock charge density for CO2
    # Carbon gets +1.0, Oxygens get -0.5 to make it neutral
    rhot_r = grid.zeros()
    r_vg = grid.xyz()   # (40, 40, 40, 3)
    pos = atoms.positions / Bohr   # (3, 3)
    sigma = 0.8  # Gaussian width
    charges = [1.0, -0.5, -0.5]  # C, O, O

    for i, charge in enumerate(charges):
        # Subtract the (3,) position directly and sum along the last axis
        d2 = ((r_vg - pos[i])**2).sum(axis=-1)
        rhot_r.data[:] += charge * np.exp(-d2 / (2 * sigma**2))    # rhot_r.data.shape (40, 40, 40)

    # Enforce exact neutrality for the periodic plane-wave solver
    rhot_r.data[:] -= rhot_r.data.mean()
    
    # Transform density from real-space to plane-wave space
    rhot_g = rhot_r.fft(pw=pw)   # rhot_g.data.shape (3202, )

    # B) vHt_g: Empty array for the potential
    vHt_g = pw.zeros()   # vHt_g.data.shape (3202, )
    # C) warm_start = False
    
    # ---------------------------------------------------------
    # EXECUTING THE SOLVER
    # ---------------------------------------------------------
    epot = solver.solve(vHt_g=vHt_g, rhot_g=rhot_g, warm_start=False)

    print(f"\nSolve complete!")
    print(f"Electrostatic Energy: {epot * Ha:.6f} eV")

    # Transform the solved potential back to real space to inspect it
    vHt_r = vHt_g.ifft(grid=grid)
    print(f"Max Potential near nuclei: {vHt_r.data.max():.4f} Ha")
    print(f"Min Potential in solvent:  {vHt_r.data.min():.4f} Ha")

if __name__ == "__main__":
    test_poisson_solve_co2()