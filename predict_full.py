import os
from dataclasses import dataclass

import gemmi
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt
from tqdm.keras import TqdmCallback
from model import t_model, cnn
import sugar_based_align as sba
import constants


@dataclass
class Params:
    dataset_base_dir: str = "dataset"
    datasets = {"train": "data/train.csv", "test": "data/test.csv"}
    shape: int = 16
    pdb_path: str = "data/pdb_files"
    map_path: str = "data/map_files"
    mtz_path: str = "data/mtz_files"


def generator(pdb_code: str):

    structure = gemmi.read_structure(os.path.join(Params.pdb_path, f"pdb{pdb_code}.ent"))
    mtz = gemmi.read_mtz_file(os.path.join(Params.mtz_path, f"{pdb_code}_phases.mtz"))
    grid = mtz.transform_f_phi_to_map("FWT", "PHWT")


    # grid = gemmi.read_ccp4_map(os.path.join(Params.map_path, f"{pdb_code}.map")).grid
    grid.normalize()

    base_groups = constants.base_groups()

    for chain in structure[0]:
        for residue in chain:

            shape = Params.shape
            base_types = constants.base_types()

            info = gemmi.find_tabulated_residue(residue.name)
            if residue.name not in base_types:
                continue

            if info.kind not in (gemmi.ResidueInfoKind.RNA, gemmi.ResidueInfoKind.DNA):
                continue

            if not sba.base_in_density(2, grid, residue):
                continue

            transform = sba.align_residue(residue, r, True)

            if not transform:
                continue

            box = np.zeros((shape, shape, shape), dtype=np.float32)
            grid.interpolate_values(box, transform.inverse())

            yield box.reshape((shape, shape, shape, 1)), residue.name

def predict():
    model = tf.keras.models.load_model(
        "models/categorical_model1.best.hdf5",
        compile=False,
    )

    # base_groups = {
    #     'U': "pur", 'C': "pur", 'G': "pyr", 'A': "pyr", 'DA': "pyr", 'DT': "pur", 'DG': "pyr", 'DC': "pur"
    # }

    # predict_lookup = {"pyr": [1], "pur": [0]}
    predict_lookup = {}
    for k, v in constants.base_groups_full().items():
        predict_lookup[k] = [np.argmax(v, axis=-1)]

    print(predict_lookup)

    pos = 0
    neg = 0
    total = 0

    logs = []

    for x in generator("1d4r"):
        prediction = model.predict(x[0].reshape(1,16,16,16,1), verbose=2)
        arg_max = np.argmax(prediction, axis=-1)

        total += 1
        if arg_max == predict_lookup[x[1]]:
            pos += 1
        else:
            print(x[1], predict_lookup[x[1]])
            for k,v in predict_lookup.items():
                if v == arg_max:
                    logs.append(f"Real - {x[1]}, Predicted - {k}, {prediction=}")

            neg += 1

    print(f"{pos=},{neg=},{total=} = {100*pos/total}")
    for log in logs:
        print(log, "\n")

if __name__ == "__main__":
    # _, r = sba.superimpose_residues()
    #
    # out_structure = gemmi.Structure()
    # out_model = gemmi.Model("A")
    # out_chain = gemmi.Chain("A")
    # out_chain.add_residue(r)
    # out_model.add_chain(out_chain)
    # out_structure.add_model(out_model)
    # out_structure.write_pdb("data/reference_residue.pdb")

    r = gemmi.read_structure("data/reference_residue.pdb")[0][0][0]

    predict()


