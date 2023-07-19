import gemmi
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, List
from tqdm import tqdm
from multiprocessing import Pool
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
import random


@dataclass
class Names:
    sugar_file: str = "_interpolated_sugar"
    phosphate_file: str = "_interpolated_phosphate"
    base_file: str = "_interpolated_base"
    no_sugar_file: str = "_interpolated_no_sugar"
    density_file: str = "_interpolated_density"


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


def _initialise_neighbour_search(structure: gemmi.Structure):
    pur = gemmi.NeighborSearch(structure[0], structure.cell, 1.5)
    pyr = gemmi.NeighborSearch(structure[0], structure.cell, 1.5)

    base_types = ['U', 'C', 'G', 'A', 'DA', 'DT', 'DG', 'DC']
    base_groups = {
        'U': "pur", 'C': "pur", 'G': "pyr", 'A': "pyr", 'DA': "pyr", 'DT': "pur", 'DG': "pyr", 'DC': "pur"
    } 
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


    for n_ch, chain in enumerate(structure[0]):
        for n_res, res in enumerate(chain):
            
            if res.name in base_types:
                base_type = base_groups[res.name]
                
                if base_type == "pur":
                    for n_atom, atom in enumerate(res):
                        if atom.name in base_atoms:
                            pur.add_atom(atom, n_ch, n_res, n_atom)
                
                if base_type == "pyr":
                    for n_atom, atom in enumerate(res):
                        if atom.name in base_atoms:
                            pyr.add_atom(atom, n_ch, n_res, n_atom)
                        

    return (
        pur,pyr
    )

def _initialise_center_neighbour_search(structure: gemmi.Structure):
    
    base_types = ['U', 'C', 'G', 'A', 'DA', 'DT', 'DG', 'DC']
    base_groups = {
        'U': "pur", 'C': "pur", 'G': "pyr", 'A': "pyr", 'DA': "pyr", 'DT': "pur", 'DG': "pyr", 'DC': "pur"
    } 
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


    # base_midpoint_structure = gemmi.Structure()
    pur_midpoint_model = gemmi.Model("A")
    pur_midpoint_chain = gemmi.Chain("A")
    pur_midpoint_residue = gemmi.Residue()
    pur_midpoint_residue.seqid = gemmi.SeqId("2")
    
    pyr_midpoint_model = gemmi.Model("A")
    pyr_midpoint_chain = gemmi.Chain("A")
    pyr_midpoint_residue = gemmi.Residue()
    pyr_midpoint_residue.seqid = gemmi.SeqId("2")
    
    
    for n_ch, chain in enumerate(structure[0]):
        for n_res, res in enumerate(chain):
            if res.name not in base_types: 
                continue
            
            base_type = base_groups[res.name]
                
            base_midpoint = gemmi.Position(0,0,0)
            count = 0
                
            for n_atom, atom in enumerate(res):
                if atom.name in base_atoms:
                    base_midpoint += atom.pos
                    count += 1
            
            base_midpoint = base_midpoint/count
            
            atom = res[0]
            atom.pos = base_midpoint
            
            if base_type == "pur":
                pur_midpoint_residue.add_atom(atom)
            elif base_type == "pyr":
                pyr_midpoint_residue.add_atom(atom)
            else: 
                print(base_type, "not found")
            
    pyr_midpoint_chain.add_residue(pyr_midpoint_residue)
    pyr_midpoint_model.add_chain(pyr_midpoint_chain)

    pur_midpoint_chain.add_residue(pur_midpoint_residue)
    pur_midpoint_model.add_chain(pur_midpoint_chain)
    
    pur = gemmi.NeighborSearch(pur_midpoint_model, structure.cell, 1).populate()
    pyr = gemmi.NeighborSearch(pyr_midpoint_model, structure.cell, 1).populate()

    # for n_ch, chain in enumerate(structure[0]):
    #     for n_res, res in enumerate(chain):
            
    #         if res.name in base_types:
    #             base_type = base_groups[res.name]
                
    #             if base_type == "pur":
    #                 for n_atom, atom in enumerate(res):
    #                     if atom.name in base_atoms:
    #                         pur.add_atom(atom, n_ch, n_res, n_atom)
                
    #             if base_type == "pyr":
    #                 for n_atom, atom in enumerate(res):
    #                     if atom.name in base_atoms:
    #                         pyr.add_atom(atom, n_ch, n_res, n_atom)
                        

    return pur,pyr
    



def generate_c_alpha_positions(map_path: str, pdb_code: str, sample_size: int ): 

    # Need to find positions to add to the help file which will include position of high density but no sugars

    grid_spacing = 0.7

    input_grid = gemmi.read_ccp4_map(map_path).grid
    input_grid.normalize()
    try:
        structure = data.import_pdb(pdb_code)
    except FileNotFoundError:
        print("[FAILED]:", map_path, pdb_code)
        return

    box = get_bounding_box(input_grid)
    size = box.get_size()
    num_x = -(-int(size.x / grid_spacing) // 16 * 16)
    num_y = -(-int(size.y / grid_spacing) // 16 * 16)
    num_z = -(-int(size.z / grid_spacing) // 16 * 16)
    array = np.zeros((num_x, num_y, num_z), dtype=np.float32)
    scale = gemmi.Mat33(
        [[grid_spacing, 0, 0], [0, grid_spacing, 0], [0, 0, grid_spacing]]
    )
    transform = gemmi.Transform(scale, box.minimum)
    input_grid.interpolate_values(array, transform)

    cell = gemmi.UnitCell(size.x, size.y, size.z, 90, 90, 90)

    c_alpha_search = gemmi.NeighborSearch(structure[0], structure.cell, 3)

    c_alpha_atoms = ["CA", "CB"]

    grid_sample_size = 32

    for n_ch, chain in enumerate(structure[0]):
            for n_res, res in enumerate(chain):
                for n_atom, atom in enumerate(res):
                    if atom.name in c_alpha_atoms:
                        c_alpha_search.add_atom(atom, n_ch, n_res, n_atom)

    potential_positions = []

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                index_pos = gemmi.Vec3(i, j, k)
                position = gemmi.Position(transform.apply(index_pos))

                any_protein_backbone = c_alpha_search.find_atoms(position, "\0", radius=3)

                if len(any_protein_backbone) > 0: 
                    translatable_position = (i-grid_sample_size/2, j-grid_sample_size/2, k-grid_sample_size/2)
                    potential_positions.append(translatable_position)

    if len(potential_positions) != 0: 
        return random.sample(potential_positions, sample_size)
    return []

def generate_class_files(map_path: str, pdb_code: str, base_dir: str):
    grid_spacing = 0.7

    input_grid = gemmi.read_ccp4_map(map_path).grid
    input_grid.normalize()
    try:
        pdb_path = f"data/pdb_files/pdb{pdb_code}.ent"
        structure = gemmi.read_structure(pdb_path)
    except FileNotFoundError:
        print("[FAILED]:", map_path, pdb_code)
        return

    box = get_bounding_box(input_grid)
    size = box.get_size()
    num_x = -(-int(size.x / grid_spacing) // 16 * 16)
    num_y = -(-int(size.y / grid_spacing) // 16 * 16)
    num_z = -(-int(size.z / grid_spacing) // 16 * 16)
    array = np.zeros((num_x, num_y, num_z), dtype=np.float32)
    scale = gemmi.Mat33(
        [[grid_spacing, 0, 0], [0, grid_spacing, 0], [0, 0, grid_spacing]]
    )
    transform = gemmi.Transform(scale, box.minimum)
    input_grid.interpolate_values(array, transform)

    cell = gemmi.UnitCell(size.x, size.y, size.z, 90, 90, 90)

    interpolated_grid = gemmi.FloatGrid(array, cell)
    
    (
        pur_search, pyr_search
    ) = _initialise_neighbour_search(structure)

    pur_map = np.zeros(array.shape, dtype=np.float32)
    pyr_map = np.zeros(array.shape, dtype=np.float32)
    
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                # indices are i,j,k
                # need to turn those indicies into xyz from the transformed
                index_pos = gemmi.Vec3(i, j, k)
                position = gemmi.Position(transform.apply(index_pos))

                any_pur = pur_search.find_atoms(position, "\0", radius=1.5)
                any_pyr = pyr_search.find_atoms(position, "\0", radius=1.5)

                pur_mask = 1.0 if len(any_pur) > 1 else 0.0
                pyr_mask = 1.0 if len(any_pyr) > 1 else 0.0

                pur_map[i][j][k] = pur_mask
                pyr_map[i][j][k] = pyr_mask

    output_dir = os.path.join(base_dir, pdb_code)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pur_grid = gemmi.FloatGrid(pur_map)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = pur_grid
    ccp4.grid.unit_cell.set(array.shape[0], array.shape[1], array.shape[2], 90, 90, 90)
    ccp4.grid.spacegroup = gemmi.SpaceGroup("P1")
    ccp4.update_ccp4_header()
    pur_path = os.path.join(output_dir, f"{pdb_code}_pur.map")
    ccp4.write_ccp4_map(pur_path)

    pyr_grid = gemmi.FloatGrid(pyr_map)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = pyr_grid
    ccp4.grid.unit_cell.set(array.shape[0], array.shape[1], array.shape[2], 90, 90, 90)
    ccp4.grid.spacegroup = gemmi.SpaceGroup("P1")
    ccp4.update_ccp4_header()
    pyr_path = os.path.join(output_dir, f"{pdb_code}_pyr.map")
    ccp4.write_ccp4_map(pyr_path)
    
    return input_grid, interpolated_grid, transform

   
def get_map_list(directory: str) -> List[Tuple[str, str]]:
    # Returns list of tuples containing map_path and pdb_code
    map_list = os.listdir(directory)
    return [
        (os.path.join(directory, map_file), map_file.replace(".map", ""))
        for map_file in map_list
        if ".map" in map_file
    ]


def get_data_dirs(base_dir: str) -> List[Tuple[str, str]]:
    data_dirs = os.listdir(base_dir)
    return [
        (os.path.join(base_dir, directory), directory)
        for directory in data_dirs
        if os.path.isfile(
            os.path.join(base_dir, directory, f"{directory}_pur.map")
        )
    ]


def get_base_positions(raw_grid, interpolated_grid, pdb_code, transform): 
    pdb_path = f"data/pdb_files/pdb{pdb_code}.ent"
    
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
        'U': "pur", 'C': "pur", 'G': "pyr", 'A': "pyr", 'DA': "pyr", 'DT': "pur", 'DG': "pyr", 'DC': "pur"
    } 
    
    pur_ns, pyr_ns = _initialise_center_neighbour_search(structure)
    
    pur_list = []
    pyr_list = []

    
    for i in range(interpolated_grid.array.shape[0]):
       for j in range(interpolated_grid.array.shape[1]):
           for k in range(interpolated_grid.array.shape[2]):
                index_pos = gemmi.Vec3(i, j, k)
                position = gemmi.Position(transform.apply(index_pos))

                any_pur = pur_ns.find_atoms(position, "\0", radius=0.3)
                any_pyr = pyr_ns.find_atoms(position, "\0", radius=0.3)
                
                position_list = [int(i), int(j), int(k)]
                
                if any_pur:
                    if position_list not in pur_list:
                        pur_list.append(position_list)

                if any_pyr:
                    if position_list not in pyr_list:
                        pyr_list.append(position_list)
                    
    return pur_list, pyr_list

def map_worker(data: Tuple[str, str]):
    output_dir = "./dataset"
    map_file, pdb_code = data
    
    if os.path.isdir(f"./dataset/{pdb_code}"):
        return
    
    raw_grid, interpolated_grid, transform = generate_class_files(map_file, pdb_code, output_dir)
    pur_list, pyr_list = get_base_positions(raw_grid, interpolated_grid, pdb_code, transform)
    
    base_dir = os.path.join(output_dir, pdb_code)
    help_file_path = os.path.join(base_dir, "pur_list.csv")

    with open(help_file_path, "w") as help_file:
        help_file.write("X,Y,Z\n")

        for translation in pur_list:
            help_file.write(f"{translation[0]},{translation[1]},{translation[2]}\n")

    help_file_path = os.path.join(base_dir, "pyr_list.csv")

    with open(help_file_path, "w") as help_file:
        help_file.write("X,Y,Z\n")

        for translation in pyr_list:
            help_file.write(f"{translation[0]},{translation[1]},{translation[2]}\n")


def generate_map_files():
    map_list = get_map_list("./data/map_files")


    # for x in map_list: 
    #     if x[1] == "1ais":
    #         print("WORKING")
    #         map_worker(x)

    with Pool() as pool:
        r = list(tqdm(pool.imap(map_worker, map_list), total=len(map_list)))


def generate_position_boxes(
    base_dir: str, pdb_code: str, threshold: float
) -> List[List[int]]:
    pur_map = os.path.join(base_dir, f"{pdb_code}_pur.map")
    pyr_map = os.path.join(base_dir, f"{pdb_code}_pyr.map")

    pur_grid = gemmi.read_ccp4_map(pur_map).grid
    pyr_grid = gemmi.read_ccp4_map(pyr_map).grid

    assert pur_grid.unit_cell.a == pyr_grid.unit_cell.a
    assert pur_grid.unit_cell.b == pyr_grid.unit_cell.b
    assert pur_grid.unit_cell.c == pyr_grid.unit_cell.c

    a = pur_grid.unit_cell.a
    b = pur_grid.unit_cell.b
    c = pur_grid.unit_cell.c

    overlap = 4

    box_dimensions = [8, 8, 8]
    total_points = box_dimensions[0] ** 3

    na = (a // overlap) + 1
    nb = (b // overlap) + 1
    nc = (c // overlap) + 1

    translation_list = []

    for x in range(int(na)):
        for y in range(int(nb)):
            for z in range(int(nc)):
                translation_list.append([x * overlap, y * overlap, z * overlap])

    pur_translations = []
    pyr_translations = []
    
    for translation in translation_list:
        pur_sub_array = np.array(
            pur_grid.get_subarray(start=translation, shape=box_dimensions)
        )

        sum = np.sum(pur_sub_array)
        if (sum / total_points) > 0.5*threshold:
            pur_translations.append(translation)
            
        pyr_sub_array = np.array(
            pyr_grid.get_subarray(start=translation, shape=box_dimensions)
        )

        sum = np.sum(pyr_sub_array)
        if (sum / total_points) > threshold:
            pyr_translations.append(translation)

    print(pur_translations, pyr_translations)

    return pur_translations, pyr_translations

    

def help_file_worker(data_tuple: Tuple[str, str]):
    base_dir, pdb_code = data_tuple

    pur_list, pyr_list = generate_position_boxes(base_dir, pdb_code, 0.4)
    # pur_list, pyr_list = generate_base_pos_list(base_dir, pdb_code)

    help_file_path = os.path.join(base_dir, "pur_list.csv")

    with open(help_file_path, "w") as help_file:
        help_file.write("X,Y,Z\n")

        for translation in pur_list:
            help_file.write(f"{translation[0]},{translation[1]},{translation[2]}\n")

    help_file_path = os.path.join(base_dir, "pyr_list.csv")

    with open(help_file_path, "w") as help_file:
        help_file.write("X,Y,Z\n")

        for translation in pyr_list:
            help_file.write(f"{translation[0]},{translation[1]},{translation[2]}\n")

def generate_help_files():
    # Must be run after map files have been generated

    data_directories = get_data_dirs("./dataset")

    with Pool() as pool:
        r = list(
            tqdm(
                pool.imap(help_file_worker, data_directories),
                total=len(data_directories),
            )
        )


def combine_help_files():
    base_dir = "./dataset"

    pur_main_df = pd.DataFrame(columns=["PDB", "X", "Y", "Z", "classification"])
    pyr_main_df = pd.DataFrame(columns=["PDB", "X", "Y", "Z", "classification"])

    for dir in tqdm(os.scandir(base_dir), total=len(os.listdir(base_dir))):
        pur_context_path = os.path.join(dir.path, "pur_list.csv")
        
        if not os.path.isfile(pur_context_path):
            continue

        pur_df = pd.read_csv(pur_context_path)

        pur_df = pur_df.assign(PDB=dir.name)
        pur_df = pur_df.assign(classification="pur")

        pur_main_df = pd.concat([pur_main_df, pur_df])
        
        pyr_context_path = os.path.join(dir.path, "pyr_list.csv")

        if not os.path.isfile(pyr_context_path):
            continue

        pyr_df = pd.read_csv(pyr_context_path)

        pyr_df = pyr_df.assign(PDB=dir.name)
        pyr_df = pyr_df.assign(classification="pyr")

        pyr_main_df = pd.concat([pyr_main_df, pyr_df])


    pur_main_df.to_csv("./data/entire_pur_list.csv", index=False)
    pyr_main_df.to_csv("./data/entire_pyr_list.csv", index=False)


def generate_test_train_split():

    pur_df = pd.read_csv("data/entire_pur_list.csv")
    pyr_df = pd.read_csv("data/entire_pyr_list.csv")
    
    df = pd.concat([pur_df, pyr_df])

    df = df.sample(frac=1)

    # # df = pd.read_csv("./data/dataset_help_calpha_2.csv")

    train, test = train_test_split(df, test_size=0.2)

    train.to_csv("./data/train_8.csv", index=False)
    test.to_csv("./data/test_8.csv", index=False)


def seeder(data: Tuple[str, str]): 
    output_dir = "./dataset"
    map_file, pdb_code = data

    pdb_folder = os.path.join(output_dir, pdb_code)

    validated_translation_file = os.path.join(pdb_folder, "validated_translations.csv")

    df = pd.read_csv(validated_translation_file)

    if len(df) < 10: 
        sample_size = 4
    else:
        sample_size = len(df) // 5

    samples = generate_c_alpha_positions(map_path=map_file, pdb_code=pdb_code, sample_size=sample_size)

    output_df = pd.concat([df, pd.DataFrame(samples, columns=["X","Y","Z"])])
    output_path = os.path.join(pdb_folder, "validated_translations_calpha_2.csv")
    output_df.to_csv(output_path, index=False)

def seed_c_alpha_positions(): 
    map_list = get_map_list("./data/DNA_test_structures/maps_16")

    with Pool() as pool:
        r = list(tqdm(pool.imap(seeder, map_list), total=len(map_list)))

def analyse(): 
    df = pd.read_csv("data/train_8.csv")
    gby = df.groupby("classification")
    for x in gby:
        print(x)

def check_completion():
    
    count = 0
    non_count = 0
    
    for dir in os.scandir("dataset"):
        file_path = os.path.join(dir.path, "pur_list.csv")
        if os.path.isfile(file_path): 
            count += 1
        else: 
            non_count += 1
    
    print(100*(count/(count+non_count)))

def main():
# 
    # check_completion()
    # generate_class_files()
    
    # analyse()

    # generate_map_files()
    # THEN
    combine_help_files()
    # # THEN
    generate_test_train_split()

    # # THEN
    # generate_help_files()

    # seed_c_alpha_positions()
    # combine_help_files()
    # generate_test_train_split()
    # generate_c_alpha_positions("data/DNA_test_structures/maps_16/1azp.map", "1azp", "./data/DNA_test_structures")
    # generate_class_files("./data/DNA_test_structures/external_test_maps/1hr2.map", "1hr2", "./data/DNA_test_structures")
    # generate_test_train_split()
    # generate_help_files()
    # combine_help_files()


if __name__ == "__main__":
    names = Names()
    main()
