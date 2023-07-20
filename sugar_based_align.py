"""
Sugar Based Alignment - Jordan Dialpuri Began 19/07/23
"""

import json
import math
import os
from typing import List, Tuple
import random
import gemmi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import constants


def load_structure(pdb_code: str) -> gemmi.Structure:
    """Load structure from data directory through pdb_code"""

    path = f"data/pdb{pdb_code}.ent"
    return gemmi.read_structure(path)


def load_density(pdb_code: str) -> gemmi.FloatGrid:
    """Load mtz file, transform observations into map and normalise from data directory"""

    path = f"data/{pdb_code}.mtz"
    mtz = gemmi.read_mtz_file(path)
    grid = mtz.transform_f_phi_to_map("FWT", "PHWT")
    grid.normalize()
    return grid


def load_data(pdb_code: str) -> Tuple[gemmi.Structure, gemmi.FloatGrid]:
    """Load data structure and density"""
    return load_structure(pdb_code), load_density(pdb_code)


def align_residue(src_residue: gemmi.Residue, tgt_residue: gemmi.Residue, add_space: bool = True) -> gemmi.Transform:
    """Align src residue onto tgt residue"""

    # align_atoms = constants.alignment_atoms(src_residue.name)
    align_atoms = list(set(constants.sugar_atoms() +
                           constants.alignment_atoms(src_residue.name)))

    tgt_pos = []
    src_pos = []

    for atom in tgt_residue:
        if atom.name in align_atoms:
            src_atom = src_residue.find_atom(atom.name, "\0", atom.element)
            if src_atom:
                tgt_pos.append(atom.pos)
                src_pos.append(src_atom.pos)

    if not len(align_atoms) == len(tgt_pos) == len(src_pos):
        return None

    transform = gemmi.superpose_positions(tgt_pos, src_pos).transform

    if add_space:
        return gemmi.Transform(transform.mat, transform.vec - gemmi.Vec3(*constants.base_shift()) + gemmi.Vec3(8, 8, 8))
    return transform

    # return gemmi.superpose_positions(
    #     [a.pos for a in tgt_residue if a.name in align_atoms],
    #     [a.pos for a in src_residue if a.name in align_atoms]).transform


def align_residue_from_positions(src_residue: gemmi.Residue,
                                 positions: List[List[float]]) -> gemmi.Transform:
    """Align src residue onto target positions"""

    align_atoms = constants.alignment_atoms(src_residue.name)

    return gemmi.superpose_positions(
        [gemmi.Position(*pos) for pos in positions],
        [a.pos for a in src_residue if a.name in align_atoms]).transform


def apply_transform(residue: gemmi.Residue, transform: gemmi.Transform) -> gemmi.Residue:
    """Apply a transform to a residue"""
    for atom in residue:
        atom.pos = gemmi.Position(transform.apply(atom.pos))
    return residue


def calculate_sugar_middlepoint(residue: gemmi.Residue,
                                only_alignment_atoms: bool = True) -> gemmi.Position:
    """Calculate middlepoint of sugar atoms in residue"""

    if only_alignment_atoms:
        sugar_atoms = constants.alignment_atoms(residue.name)
    else:
        sugar_atoms = constants.sugar_atoms()

    position_sum = gemmi.Position(0, 0, 0)
    position_count = 0

    for atom in residue:
        if atom.name in sugar_atoms:
            position_sum += atom.pos
            position_count += 1

    if position_count:
        return position_sum / position_count

    raise RuntimeError("Residue contains no sugar atoms")


def calculate_base_middlepoint(residue: gemmi.Residue) -> gemmi.Position:
    """Calculate middlepoint of base atoms in residue"""
    base_atoms = constants.base_atoms()

    position_sum = gemmi.Position(0, 0, 0)
    position_count = 0

    for atom in residue:
        if atom.name in base_atoms:
            position_sum += atom.pos
            position_count += 1

    if position_count:
        return position_sum / position_count
    raise RuntimeError("Residue contains no base atoms")


def superimpose_residues() -> Tuple[gemmi.Structure, gemmi.Residue]:
    """Superimpose residues from 1hr2 and return Tuple of
    (structure of superimposed structures, reference residue)"""
    structure, _ = load_data("1hr2")

    src_residue = structure[0][0][0]
    sugar_midpoint = calculate_sugar_middlepoint(
        src_residue, only_alignment_atoms=False)

    box_offset = gemmi.Position(
        constants.shape() / 2, constants.shape() / 2, constants.shape() / 2)

    align_to_origin = gemmi.Transform(gemmi.Mat33(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), -sugar_midpoint + box_offset)

    src_residue = apply_transform(src_residue, align_to_origin)

    base_types = ['U', 'C', 'G', 'A', 'DA', 'DT', 'DG', 'DC']

    out_structure = gemmi.Structure()
    out_model = gemmi.Model("A")
    out_chain = gemmi.Chain("A")

    for chain in structure[0]:
        for residue in chain:
            if residue.name not in base_types:
                continue

            transform = align_residue(residue, src_residue, False)
            tranformed_residue = apply_transform(residue, transform)
            out_chain.add_residue(tranformed_residue)

    out_model.add_chain(out_chain)
    out_structure.add_model(out_model)

    # print(src_residue[0])

    return out_structure, src_residue


def calculate_average_midpoint(structure: gemmi.Structure,
                               residue: gemmi.Residue, output_file: str) -> None:
    """Calculate average midpoint from already superimposed structure and writes it to file"""

    average_midpoint_sum = gemmi.Position(0, 0, 0)
    average_midpoint_count = 0

    for chain in structure[0]:
        for res in chain:
            base_midpoint = calculate_base_middlepoint(res)
            average_midpoint_sum += base_midpoint
            average_midpoint_count += 1

    average_midpoint = average_midpoint_sum / average_midpoint_count

    data = {
        "average_base_point": [round(x, 2) for x in average_midpoint.tolist()],
        # "reference_residue": [{"name": x.name, "positions": [round(p, 2) for p in x.pos.tolist()]}
        #                       for x in residue if x.name in constants.alignment_atoms(residue.name)]
    }

    with open(output_file, 'w', encoding='UTF-8') as out_file:
        json.dump(data, out_file)


def _combine_transforms(tr1: gemmi.Transform, tr2: gemmi.Transform) -> gemmi.Transform:
    """Combine transformations - Paul Bond"""
    mat = tr2.mat.multiply(tr1.mat)
    vec = tr2.mat.multiply(tr1.vec) + tr2.vec
    return gemmi.Transform(mat, vec)


def base_in_density(threshold: float, grid: gemmi.FloatGrid, residue: gemmi.Residue) -> bool:
    """Check if base in density"""
    density_sum = 0
    density_count = 0

    base_atoms = constants.base_atoms()

    for atom in residue:
        if atom.name in base_atoms:
            density_sum += grid.interpolate_value(atom.pos)
            density_count += 1

    if density_count > 0:
        score = density_sum / density_count
        # print(score, threshold, score > threshold)
        return score > threshold

    return False
    # raise RuntimeError("Density count <= 0", density_count)


def calculate_interpolated_boxes(reference_residue: gemmi.Residue) -> np.ndarray:
    """Calculate interpolated box for residue"""
    structure, grid = load_data("1hr2")

    shape = constants.shape()
    base_types = constants.base_types()

    for chain in structure[0]:
        for n, residue in enumerate(chain[2:10]):
            info = gemmi.find_tabulated_residue(residue.name)
            if residue.name not in base_types:
                continue

            if info.kind not in (gemmi.ResidueInfoKind.RNA, gemmi.ResidueInfoKind.DNA):
                continue

            if not base_in_density(2, grid, residue):
                continue

            transform = align_residue(residue, reference_residue, True)
            box = np.zeros((shape, shape, shape), dtype=np.float32)
            grid.interpolate_values(box, transform.inverse())


def calculate_train_test_set():
    pdb_dir = "data/pdb_files"

    file_list = os.listdir(pdb_dir)
    test_train_split = 0.2

    random.shuffle(file_list)

    split_index = math.ceil(0.2 * len(file_list))

    train = [x[3:-4] for x in file_list[split_index:]]
    test = [x[3:-4] for x in file_list[:split_index]]

    train_df = pd.DataFrame(train, columns=["PDB"])
    test_df = pd.DataFrame(test, columns=["PDB"])

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    # print(len(file_list), len(train), len(test), "=", len(train)+len(test))


if __name__ == "__main__":
    calculate_train_test_set()

    # s, r = superimpose_residues()
    # calculate_average_midpoint(s, r, "data/average_data.json")
    #
    # calculate_interpolated_box(r)
