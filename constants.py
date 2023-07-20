import json
from typing import List, Dict

import gemmi


def base_shift() -> List[float]:
    """Get base shift from file"""
    with open("data/average_data.json", "r", encoding="UTF-8") as json_file:
        data = json.load(json_file)
        return data["average_base_point"]


def shape() -> int:
    """Get shape of box"""
    return 16


def sugar_atoms() -> List[str]:
    """Return sugar atom list"""
    return ["C1'", "C2'", "C3'", "O3'", "C4'", "C5'"]


def base_atoms() -> List[str]:
    """Return base atom list"""
    return [
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


def alignment_atoms(residue_name: str) -> List[str]:
    """Calculate alignment atoms for specific residue type"""
    align_atoms = ["C1'", "O4'", "C2'"]

    n9_base_types = ['G', 'A', 'DA', 'DG']
    n1_base_types = ["C", "U", "DC", "DU", "T"]

    if residue_name in n9_base_types:
        align_atoms.append("N9")
    elif residue_name in n1_base_types:
        align_atoms.append("N1")

    return align_atoms


def base_types() -> List[str]:
    return ['U', 'C', 'G', 'A', 'DA', 'DT', 'DG', 'DC']


def base_groups() -> Dict[str, List[int]]:
    return {
        'U': [1, 0], 'C': [1, 0], 'G': [0, 1], 'A': [0, 1], 'DA': [0, 1], 'DT': [1, 0], 'DG': [0, 1], 'DC': [1, 0]
    }


def base_groups_full() -> Dict[str, List[int]]:
    return {
        'U': [1, 0, 0, 0, 0],
        'C': [0, 1, 0, 0, 0],
        'DC': [0, 1, 0, 0, 0],
        'G': [0, 0, 1, 0, 0],
        'DG': [0, 0, 1, 0, 0],
        'A': [0, 0, 0, 1, 0],
        'DA': [0, 0, 0, 1, 0],
        'DT': [0, 0, 0, 0, 1],
    }
