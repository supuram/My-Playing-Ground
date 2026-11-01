from xml.etree.ElementTree import parse as xmlparse
import numpy as np
pp = {}
root = xmlparse('Po.nc.pbe.z_16.oncvpsp3.dojo.v0.4.1-std.upf').getroot()
def toarray(element):
    attr = element.attrib
    numbers = [float(n) for n in element.text.split()]
    if attr:
        assert attr['type'] == 'real'
        assert int(attr['size']) == len(numbers)
        icut = attr.get('cutoff_radius_index')
        # XXX There's also something called ultrasoft_cutoff_radius
        if icut is not None:
            numbers = numbers[:int(icut)]

    return np.array(numbers)
header_element = root.find('PP_HEADER')
header = header_element.attrib
for key, val in header.items():
    header[key] = val.strip()
for key in ['is_paw', 'is_coulomb', 'has_so', 'has_wfc', 'has_gipaw', 'paw_as_gipaw', 'core_correction']:
    if key in header:
        header[key] = {'F': False, 'T': True}[header[key]]
info_element = root.find('PP_INFO')
pp['info'] = info_element.text
pp['header'] = header
mesh_element = root.find('PP_MESH')
pp['r'] = toarray(mesh_element.find('PP_R'))
non_local = root.find('PP_NONLOCAL')
for element in non_local:
    print(element)
pp['nlcc'] = toarray(root.find('PP_NLCC'))
print(pp['nlcc'])