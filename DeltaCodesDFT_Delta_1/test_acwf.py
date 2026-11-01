from pathlib import Path
from gpaw import GPAW, PW, FermiDirac

from gpaw.utilities import acwf
from gpaw.utilities.acwf import read_bm_wien2k_data, eos_birch_murnaghan, calc_delta

code_data_dir = (
    Path.home()
    / "gpaw_projects"
    / "acwf"
    / "ABACUS-ACWF-verification-data-and-scripts"
    / "acwf-verification-scripts"
    / "acwf_paper_plots"
    / "code-data"
)

wien2k_oxides_path = (
    code_data_dir / "results-oxides-verification-PBE-v1-wien2k-dk_0.06.json"
)

element = "Al"
structure = "FCC"

# Extract WIEN2k reference values
all_wien2k = read_bm_wien2k_data(wien2k_oxides_path)
ref_data = all_wien2k[element][structure]

v0_ref = ref_data["min_volume"]
b0_ref = ref_data["bulk_modulus_ev_ang3"]
b1_ref = ref_data["bulk_deriv"]

# Setup structure and GPAW calculator
atoms = acwf.reference_structure(element, structure)

calc = GPAW(
    mode=PW(800),
    xc="PBE",
    spinpol=True,
    occupations=FermiDirac(0.01),
    kpts={"density": 16.67, "gamma": True},
    convergence={'energy': 1e-8, 'density': 1e-7},
    txt=f"gpaw_{element}_{structure}.txt",
)
atoms.calc = calc

# Run EOS calculation and extract fit parameters
d_dict_gpaw = eos_birch_murnaghan(atoms)
v0_gpaw = d_dict_gpaw["v0"]
b0_gpaw = d_dict_gpaw["b0"]  
b1_gpaw = d_dict_gpaw["b1"]

delta, deltarel, delta1 = calc_delta(
    v0_gpaw, b0_gpaw, b1_gpaw,
    v0_ref, b0_ref, b1_ref
)

echarge = 1.602176565e-19
to_gpa = echarge * 1.0e21

print(f"\n--- Benchmark Results: {element} ({structure}) ---")
print(f"GPAW:   V0 = {v0_gpaw:.4f} Å³/atom, B0 = {b0_gpaw * to_gpa:.2f} GPa, B1 = {b1_gpaw:.3f}")
print(f"WIEN2k: V0 = {v0_ref:.4f} Å³/atom, B0 = {b0_ref * to_gpa:.2f} GPa, B1 = {b1_ref:.3f}")
print(f"Delta:       {delta:.4f} meV/atom")
print(f"Relative:    {deltarel:.2f} %")
print(f"Delta_1:     {delta1:.4f} meV/atom")