import numpy as np 
import gemmi
from typing import List
from generate_dataset import get_bounding_box
from dataclasses import dataclass

@dataclass
class Params: 
    shape: int = 32


def _get_base_plane(residue: gemmi.Residue) -> np.matrix:
        
    position_list: List[np.array] = []
    
    for atom in residue: 
        if atom.name in base_atoms: 
            position_list.append(np.array(atom.pos.tolist()))
            
    tmp_A = []
    tmp_B = []
    
    for position in position_list[:3]:
        tmp_A.append([position[0], position[1], 1])
        tmp_B.append(position[2])
        
    A = np.matrix(tmp_A)
    B = np.matrix(tmp_B).T
    
    fit = (A.T * A).I * A.T * B
    
    print("POS LIST" , position_list[:3])
    x = np.array(fit.T)[0]
    
    samp_coord = position_list[0]
    
    
    d = -(x[0]*samp_coord[0] + x[1]*samp_coord[1] - x[2]*samp_coord[2])
    
    # w = [-30.916, -5.869, 85.493]
    # d = x[0][0]*w[0] + x[0][1]*w[1] + x[0][2]*w[2]
    
    print(x, d)
    return fit 

def get_base_plane(residue: gemmi.Residue):
    
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
#    return b / np.sqrt(a*a + b*b + c*c)
   return b / np.sqrt(a*a + b*b)


def calc_u2(a, b, c):
    # return -a / np.sqrt(a*a + b*b + c*c)
    return -a / np.sqrt(a*a + b*b)


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
    
    shift = gemmi.Position((Params.shape/2), (Params.shape/2), (Params.shape/2))
    
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



def calculate_density():
    structure = gemmi.read_structure("data/pdb_files/pdb1hr2.ent")
    mtz = gemmi.read_mtz_file("data/mtz_files/1hr2.mtz")
    grid = mtz.transform_f_phi_to_map("FWT", "PHWT")
    grid.normalize()
    
    grid_spacing = 0.7
    
    residue = structure[0][0][0]
    plane, position_list = get_base_plane(residue=residue)
    rotation = get_rotation_matrix(plane)
    position = get_res_midpoint(residue)

    transform = gemmi.Transform(rotation, gemmi.Vec3(0,0,0))

    for a in residue: 
        a.pos = gemmi.Position(transform.apply(a.pos))
    
    dencalc = gemmi.DensityCalculatorX()
    dencalc.d_min = 2.8
    dencalc.set_grid_cell_and_spacegroup(structure)
    dencalc.put_model_density_on_grid(structure[0])
    
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = dencalc.grid
    ccp4.grid.unit_cell = dencalc.grid.unit_cell
    ccp4.grid.spacegroup = gemmi.SpaceGroup("P1")
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map("test/rotation_tests/calculated_grid.map")
    
    debug_model = gemmi.Model("A")
    debug_chain = gemmi.Chain("A")
    debug_residue = gemmi.Residue()
    debug_residue.seqid = gemmi.SeqId("123")
    debug_residue.name = "PRO"
    
    atom = gemmi.Atom()
    atom.pos = position
    atom.name = "X"
    debug_residue.add_atom(atom)
    
    debug_chain.add_residue(debug_residue)
    debug_model.add_chain(debug_chain)
    
    debug_structure = gemmi.Structure()
    debug_structure.add_model(debug_model)
    debug_structure.write_pdb("test/rotation_tests/calculated_pos.pdb")
    
    

def main(): 
    structure = gemmi.read_structure("data/pdb_files/pdb1hr2.ent")
    mtz = gemmi.read_mtz_file("data/mtz_files/1hr2.mtz")
    grid = mtz.transform_f_phi_to_map("FWT", "PHWT")
    grid.normalize()
    
    grid_spacing = 0.7
    
    residue = structure[0][0][0]

    plane, position_list = get_base_plane(residue=residue)
    rotation = get_rotation_matrix(plane)
    translation = get_res_midpoint(residue)
    
    scale = gemmi.Mat33([[grid_spacing, 0, 0], [0, grid_spacing, 0], [0, 0, grid_spacing]])
    transform = gemmi.Transform(scale.multiply(rotation), translation)
    
    values = np.zeros((Params.shape, Params.shape, Params.shape), dtype=np.float32)
    grid.interpolate_values(values, transform)
    
    interpolated_values = np.zeros((Params.shape, Params.shape, Params.shape), dtype=np.float32)
    interpolated_transform = gemmi.Transform(scale, translation)
    grid.interpolate_values(interpolated_values, interpolated_transform)
    
    sample_cell = gemmi.UnitCell(
        Params.shape*grid_spacing,
        Params.shape*grid_spacing,
        Params.shape*grid_spacing,
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
    

def get_arrays():     
    
    structure = gemmi.read_structure("data/pdb_files/pdb1hr2.ent")
    mtz = gemmi.read_mtz_file("data/mtz_files/1hr2.mtz")
    grid = mtz.transform_f_phi_to_map("FWT", "PHWT")
    grid.normalize()

    grid_spacing = 0.7

    box = get_bounding_box(grid)
    size = box.get_size()

    num_x = -(-int(size.x / grid_spacing) // 16 * 16)
    num_y = -(-int(size.y / grid_spacing) // 16 * 16)
    num_z = -(-int(size.z / grid_spacing) // 16 * 16)

    array = np.zeros((num_x, num_y, num_z), dtype=np.float32)
    scale = gemmi.Mat33(
        [[grid_spacing, 0, 0], [0, grid_spacing, 0], [0, 0, grid_spacing]]
    )
    transform = gemmi.Transform(scale, box.minimum)
    grid.interpolate_values(array, transform)

    cell: gemmi.UnitCell = gemmi.UnitCell(size.x, size.y, size.z, 90, 90, 90)
    interpolated_grid = gemmi.FloatGrid(array, cell)
    
    debug_model = gemmi.Model("A")
    debug_chain = gemmi.Chain("A")
    debug_residue = gemmi.Residue()
    debug_residue.seqid = gemmi.SeqId("123")
    debug_residue.name = "PRO"
    
    residue = structure[0][0][0]
    # for chain in structure[0]:
    #     for residue in chain: 
    plane, position_list = get_base_plane(residue=residue)
    rot_mat = get_rotation_matrix(plane)

    midpoint = get_res_midpoint(residue)
    position = gemmi.Position(transform.inverse().apply(midpoint))
    inter_nearest_point = interpolated_grid.get_nearest_point(position)

    interpolated_sample = np.array(
        interpolated_grid.get_subarray(
            [
                inter_nearest_point.u - int(Params.shape / 2),
                inter_nearest_point.v - int(Params.shape / 2),
                inter_nearest_point.w - int(Params.shape / 2)],
            shape=[Params.shape, Params.shape, Params.shape]
        )
    )
        
    small_cell = gemmi.UnitCell(Params.shape*grid_spacing,
                            Params.shape*grid_spacing,
                            Params.shape*grid_spacing, 90, 90, 90)
    interpolated_sample_grid = gemmi.FloatGrid(interpolated_sample, small_cell)


    zero_trans = gemmi.Vec3(0,0,0)
    rot_transform = gemmi.Transform(rot_mat, zero_trans)
    rotated_array = np.zeros((Params.shape, Params.shape, Params.shape), dtype=np.float32)
    interpolated_sample_grid.interpolate_values(rotated_array, rot_transform)
    rotated_grid = gemmi.FloatGrid(rotated_array, small_cell)
    
    
    # atom = gemmi.Atom()
    # atom.pos = position
    # atom.name = "X"
    # debug_residue.add_atom(atom)
    

    # rotated_position = gemmi.Position(rot_transform.apply(position))

    # nearest_point = rotated_grid.get_nearest_point(rotated_position)
    
    # rotated_sample = np.array(
    #     rotated_grid.get_subarray(
    #         [
    #             nearest_point.u - int(Params.shape / 2),
    #             nearest_point.v - int(Params.shape / 2),
    #             nearest_point.w - int(Params.shape / 2)],
    #         shape=[Params.shape, Params.shape, Params.shape]
    #     )
    # )
    # rotated_sample_grid = gemmi.FloatGrid(rotated_sample, small_cell)

    
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = rotated_grid
    ccp4.grid.unit_cell = small_cell
    ccp4.grid.spacegroup = gemmi.SpaceGroup("P1")
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map("test/rotation_tests/rotated_grid_small.map")
    
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = interpolated_sample_grid
    ccp4.grid.unit_cell = small_cell
    ccp4.grid.spacegroup = gemmi.SpaceGroup("P1")
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map("test/rotation_tests/interpolated_grid_small.map")
    
            # return array,rotated_array
            
        #     break
        # break
    
    debug_chain.add_residue(debug_residue)
    debug_model.add_chain(debug_chain)
    
    debug_structure = gemmi.Structure()
    debug_structure.add_model(debug_model)
    debug_structure.write_pdb("test/rotation_tests/interp_pos.pdb")
    
    

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

    # get_arrays()
    # main()
    calculate_density()