import numpy as np
from ase.units import Bohr
from gpaw.core import UGDesc
from gpaw.solvation.dielectric import LinearDielectric

def test_dielectric():
    cell_cv = np.diag([10.0, 10.0, 10.0]) / Bohr
    grid = UGDesc(cell=cell_cv, size=(32, 32, 32))
    
    # Pass the internal legacy descriptor, exactly as SolvationExtension does
    legacy_gd = grid._gd

    class MockCavity:
        def __init__(self, gd):
            self.g_g = gd.zeros()
            self.g_g[:] = 0.5
            self.grad_g_vg = [gd.zeros(), gd.zeros(), gd.zeros()]

    cavity = MockCavity(legacy_gd)

    eps = LinearDielectric(epsinf=78.4)
    eps.set_grid_descriptor(legacy_gd)
    eps.allocate()
    eps.update(cavity)
    # eps_gradeps is a 4-element list containing the dielectric function and its 3D spatial gradient on the grid: [eps, d(eps)/dx, d(eps)/dy, d(eps)/dz]
    
    print("Testing LinearDielectric with grid._gd...")
    print(f"  Min Permittivity: {eps.eps_gradeps[0].min():.2f}")   # eps.eps_gradeps[0] has shape (32, 32, 32)
    print(f"  Max Permittivity: {eps.eps_gradeps[0].max():.2f}")
    print(f"  Derivative (del_eps_del_g_g): {eps.del_eps_del_g_g}")

if __name__ == "__main__":
    test_dielectric()