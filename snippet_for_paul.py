from typing import List
import gemmi 
import numpy as np 

def get_base_plane(residue: gemmi.Residue):
    
    
    position_list: List[np.array] = []
    
    for atom in residue: 
        if atom.name in base_atoms: 
            position_list.append(np.array(atom.pos.tolist()))
            
    tmp_A = []
    tmp_B = []
    
    for position in position_list[:3]:
        tmp_A.append([position[0], position[1], 1])
        tmp_B.append(position[2])
        
    x = position_list[:3]
    
    BA = x[1] - x[0]
    CA = x[2] - x[0]
    
    cross = np.cross(BA, CA)
    d = x[0][0]*cross[0] + x[0][1]*cross[1] + x[0][2]*cross[2]
    return (*cross, d), position_list


def calc_cos_phi(a, b, c):
    return c / np.sqrt(a*a + b*b + c*c)


def calc_sin_phi(a, b, c):
    return np.sqrt((a*a + b*b) / (a*a + b*b + c*c))


def calc_u1(a, b, c):
   return b / np.sqrt(a*a + b*b + c*c)


def calc_u2(a, b, c):
    return -a / np.sqrt(a*a + b*b + c*c)

# From https://math.stackexchange.com/a/1167779
def get_rotation_matrix(plane: np.matrix) -> gemmi.Mat33:
    a, b, c, d = plane
    
    cos_phi = calc_cos_phi(a, b, c)
    sin_phi = calc_sin_phi(a, b, c)
    u1 = calc_u1(a, b, c)
    u2 = calc_u2(a, b, c)
    out = np.array([
        [cos_phi + u1 * u1 * (1 - cos_phi)  , u1 * u2 * (1 - cos_phi)           , u2 * sin_phi  ],
        [u1 * u2 * (1 - cos_phi)            , cos_phi + u2 * u2 * (1 - cos_phi) , -u1 * sin_phi ],
        [-u2 * sin_phi                      , u1 * sin_phi                      ,      cos_phi  ],
    ])
    
    # return np.matrix(out)
    return gemmi.Mat33(out)

def get_res_midpoint(residue: gemmi.Residue) -> gemmi.Position:
    info = gemmi.find_tabulated_residue(residue.name)
    
    shift = gemmi.Position((shape/2), (shape/2), (shape/2))
    
    if info.kind in (gemmi.ResidueInfoKind.RNA, gemmi.ResidueInfoKind.DNA):
        if residue.name not in base_types:
            return None

        base_midpoint = gemmi.Position(0, 0, 0)
        base_atom_count = 0

        for a in residue:
            if a.name in base_atoms:
                base_midpoint += a.pos
                base_atom_count += 1

        base_midpoint = base_midpoint / base_atom_count     
        return (base_midpoint-shift)

def main(): 
    structure = gemmi.read_structure("data/pdb_files/pdb1hr2.ent")
    mtz = gemmi.read_mtz_file("data/mtz_files/1hr2.mtz")
    grid = mtz.transform_f_phi_to_map("FWT", "PHWT")
    grid.normalize()
    
    grid_spacing = 0.7
    shape = 16

    
    residue = structure[0][0][0]

    plane, position_list = get_base_plane(residue=residue)
    rotation = get_rotation_matrix(plane)
    translation = get_res_midpoint(residue)
    
    # From https://github.com/paulsbond/densitydensenet/blob/b34bcaded855f0729b31a36439387050b86dc87c/train.py#L90C18-L90C18
    scale = gemmi.Mat33([[grid_spacing, 0, 0], [0, grid_spacing, 0], [0, 0, grid_spacing]])
    transform = gemmi.Transform(scale.multiply(rotation), translation)
    
    values = np.zeros((shape, shape, shape), dtype=np.float32)
    grid.interpolate_values(values, transform)
    
    interpolated_values = np.zeros((shape, shape, shape), dtype=np.float32)
    interpolated_transform = gemmi.Transform(scale, translation)
    grid.interpolate_values(interpolated_values, interpolated_transform)
    
    sample_cell = gemmi.UnitCell(
        shape*grid_spacing,
        shape*grid_spacing,
        shape*grid_spacing,
        90,
        90,
        90
    )
    
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = gemmi.FloatGrid(interpolated_values, sample_cell)
    ccp4.grid.unit_cell = sample_cell
    ccp4.grid.spacegroup = gemmi.SpaceGroup("P1")
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map("test/rotation_tests/normal_grid.map")
    
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = gemmi.FloatGrid(values, sample_cell)
    ccp4.grid.unit_cell = sample_cell
    ccp4.grid.spacegroup = gemmi.SpaceGroup("P1")
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map("test/rotation_tests/rotated_grid.map")
    
if __name__ == "__main__":
    
    base_atoms = [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "N1",
        "N2",
        "N3",
        "N4",
        "N5",
        "N6",
        "N7",
        "N8",
        "N9",
        "O2",
        "O4",
        "O6"]
    base_types = ['U', 'C', 'G', 'A', 'DA', 'DT', 'DG', 'DC']

    
    main()
    