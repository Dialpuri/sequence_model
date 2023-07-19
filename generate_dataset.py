import numpy as np
from sklearn.model_selection import train_test_split
import os
from typing import List, Tuple
import gemmi
import shutil
from dataclasses import dataclass
import copy
from tqdm import tqdm 


@dataclass 
class Parameters: 
    database_dir: str = "dataset"
    shape: int = 16

def get_bounding_box(grid: gemmi.FloatGrid) -> gemmi.PositionBox:
    extent = gemmi.find_asu_brick(grid.spacegroup).get_extent()
    corners = [
        grid.unit_cell.orthogonalize(fractional)
        for fractional in (
            extent.minimum,
            gemmi.Fractional(extent.maximum[0], extent.minimum[1], extent.minimum[2]),
            gemmi.Fractional(extent.minimum[0], extent.maximum[1], extent.minimum[2]),
            gemmi.Fractional(extent.minimum[0], extent.minimum[1], extent.maximum[2]),
            gemmi.Fractional(extent.maximum[0], extent.maximum[1], extent.minimum[2]),
            gemmi.Fractional(extent.maximum[0], extent.minimum[1], extent.maximum[2]),
            gemmi.Fractional(extent.minimum[0], extent.maximum[1], extent.maximum[2]),
            extent.maximum,
        )
    ]
    min_x = min(corner[0] for corner in corners)
    min_y = min(corner[1] for corner in corners)
    min_z = min(corner[2] for corner in corners)
    max_x = max(corner[0] for corner in corners)
    max_y = max(corner[1] for corner in corners)
    max_z = max(corner[2] for corner in corners)
    box = gemmi.PositionBox()
    box.minimum = gemmi.Position(min_x, min_y, min_z)
    box.maximum = gemmi.Position(max_x, max_y, max_z)
    return box

def get_map_list(directory: str) -> List[Tuple[str, str]]:
    map_list = os.listdir(directory)
    return [
        (os.path.join(directory, map_file), map_file.replace(".map", ""))
        for map_file in map_list
        if ".map" in map_file
    ]
    
def get_pdb_path(pdb_code: str) -> str: 
    x = f"data/pdb_files/pdb{pdb_code}.ent"
    if os.path.isfile(x): 
        return x
    return None

def get_mtz_path(pdb_code: str) -> str: 
    x = f"data/mtz_files/{pdb_code}.mtz"
    if os.path.isfile(x):
        return x
    return None

def get_map_path(pdb_code: str) -> str: 
    x = f"data/map_files/{pdb_code}.map"
    if os.path.isfile(x): 
        return x
    return None

def get_map_from_mtz(pdb_code: str) -> Tuple[gemmi.FloatGrid, gemmi.FloatGrid, gemmi.Transform]: 
        
    try:
        mtz = gemmi.read_mtz_file(x.path)
    except (RuntimeError, ValueError) as e:
        print(f"{x.name} raised {e}")
        
    grid = mtz.transform_f_phi_to_map("FWT", "PHWT")
        
    grid_spacing = 0.7

    grid.normalize()
    
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
    
    print("ARRAY" , array.shape, mtz.transform_f_phi_to_map("FWT", "PHWT").array.shape, grid.array.shape)
    
    return mtz.transform_f_phi_to_map("FWT", "PHWT"), grid, transform

    
def generate_base_type_maps(pdb: str):
    
    map_path = get_map_path(pdb)
    pdb_path = get_pdb_path(pdb)

    if not map_path and not pdb_path: 
        return 
    
    grid_spacing = 0.7
    grid = gemmi.read_ccp4_map(map_path).grid
    grid.normalize()
    
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
    
    structure = gemmi.read_structure(pdb_path)
    
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
    
    base_groups = {
        'U': [1,0], 'C': [1,0], 'G': [0,1], 'A': [0,1], 'DA': [0,1], 'DT': [1,0], 'DG': [0,1], 'DC': [1,0]
    }
    
    for c in structure[0]:
        for r in c: 
            info = gemmi.find_tabulated_residue(r.name)
            if info.kind in (gemmi.ResidueInfoKind.RNA, gemmi.ResidueInfoKind.DNA):
                if r.name not in base_types: 
                    continue
                
                base_midpoint = gemmi.Position(0,0,0)
                base_atom_count = 0 
                                
                for a in r: 
                    if a.name in base_atoms: 
                        base_midpoint += a.pos
                        base_atom_count += 1
                        
                base_midpoint = base_midpoint/base_atom_count
                shift_pos = gemmi.Position(Parameters.shape/2, Parameters.shape/2, Parameters.shape/2)
                base_startposition = base_midpoint - shift_pos
                base_startpoint = grid.get_nearest_point(base_startposition)

                index_pos = gemmi.Vec3(base_startpoint.u, base_startpoint.v, base_startpoint.w)
                print(base_startpoint, transform.mat, transform.vec)
                position = gemmi.Position(transform.apply(index_pos))
                
                point = grid.get_nearest_point(position)

                arr = np.array(
                        grid.get_subarray(
                            [point.u, point.v, point.w], shape=[Parameters.shape,Parameters.shape, Parameters.shape]
                            )
                    )
                
    
    
def split_data(): 
    directory = [(x.path, x.name) for x in os.scandir("../sugar_prediction_model/dataset")]
    
    train, test = train_test_split(directory, test_size=0.2)
    
    with open("data/train.csv", "w") as train_file: 
        train_file.write("path,name\n") 
        for x in train: 
            train_file.write(f"{x[0]},{x[1]}\n")
            
    with open("data/test.csv", "w") as test_file: 
        test_file.write("path,name\n") 
        for x in test: 
            test_file.write(f"{x[0]},{x[1]}\n")

if __name__ == "__main__":
    # split_data()
    import urllib.request

    for x in tqdm(os.scandir("data/mtz_files"), total=len(os.listdir("data/mtz_files"))): 
        path = x.path
        name = x.name[:-4]
        
        try:
            mtz = gemmi.read_mtz_file(x.path)
        except (RuntimeError, ValueError) as e:
            print(f"{x.name} raised {e}")
            
        grid = mtz.transform_f_phi_to_map("FWT", "PHWT")
        
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = grid
        ccp4.grid.unit_cell = grid.unit_cell
        ccp4.grid.spacegroup = gemmi.SpaceGroup("P1")
        ccp4.update_ccp4_header()
        density_path = os.path.join("data/map_files", f"{name}.map")
        ccp4.write_ccp4_map(density_path)        
        
        # generate_base_type_maps(name)
        # break
        # pdb_path = os.path.join(x.path, f"pdb{x.name}.ent")
        # shutil.copy(pdb_path, "data/pdb_files")
        
        