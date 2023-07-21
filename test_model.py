import json
import os
from dataclasses import dataclass

import gemmi
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import gridspec, pyplot as plt
from tqdm.keras import TqdmCallback
from model import t_model, cnn
import sugar_based_align as sba
import constants


@dataclass
class Params:
    test_list_dir: str = "data/test_set"
    shape: int = 16


def generator(pdb_code: str):
    structure = gemmi.read_structure(
        os.path.join(Params.test_list_dir, f"{pdb_code}.pdb")
    )
    mtz = gemmi.read_mtz_file(os.path.join(Params.test_list_dir, f"{pdb_code}.mtz"))
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

            # if not sba.base_in_density(2, grid, residue):
            #     continue

            transform = sba.align_residue(residue, r, True)

            if not transform:
                continue

            box = np.zeros((shape, shape, shape), dtype=np.float32)
            grid.interpolate_values(box, transform.inverse())

            yield box.reshape((shape, shape, shape, 1)), residue.name


def get_test_list():
    return [
        x.name.removesuffix(".pdb")
        for x in os.scandir(Params.test_list_dir)
        if ".pdb" in x.name
    ]


def test_model():
    model = tf.keras.models.load_model(
        "models/categorical_model2.best.hdf5",
        compile=False,
    )

    # base_groups = {
    #     'U': "pur", 'C': "pur", 'G': "pyr", 'A': "pyr", 'DA': "pyr", 'DT': "pur", 'DG': "pyr", 'DC': "pur"
    # }

    # predict_lookup = {"pyr": [1], "pur": [0]}

    predict_lookup = {}
    for k, v in constants.base_groups_full().items():
        predict_lookup[k] = [np.argmax(v, axis=-1)]

    test_list = get_test_list()

    data = {}

    for pdb in test_list:
        pos = 0
        neg = 0
        total = 0

        logs = []

        for x in generator(pdb):
            prediction = model.predict(x[0].reshape(1, 16, 16, 16, 1), verbose=2)
            arg_max = np.argmax(prediction, axis=-1)

            total += 1
            if arg_max == predict_lookup[x[1]]:
                pos += 1
            else:
                # print(x[1], predict_lookup[x[1]])
                for k, v in predict_lookup.items():
                    if v == arg_max:
                        logs.append(f"Real - {x[1]}, Predicted - {k}, {prediction=}")

                neg += 1

        print(f"{pdb=} : {pos=},{neg=},{total=} = {100*pos/total}")

        data[pdb] = {"pos": pos, "neg": neg, "completeness": 100 * pos / total}
        # for log in logs:
        #     print(log, "\n")

        with open(
            "tests/test_model/model_result.json", "w", encoding="UTF-8"
        ) as output_file:
            json.dump(data, output_file, indent=4)

def visualise_results():
    data = None
    with open("tests/test_model/model_result.json", "r", encoding="UTF-8") as output_file:
        data = json.load(output_file)
        
    res = None
    with open("tests/test_model/resolutions.json", "r", encoding="UTF-8") as res_file:
        res = json.load(res_file)
    
    # gs = gridspec.GridSpec(3, 2)
    # fig = plt.figure(figsize=(24, 12))

    # ax0 = fig.add_subplot(gs[0, :])
    # ax1 = fig.add_subplot(gs[1, :])
    # ax2 = fig.add_subplot(gs[2, 0])
    plt.style.use('seaborn-v0_8-whitegrid')
    font = {'size'   : 18}

    plt.rc('font', **font)
    # plt.rc('axes', titlesize=40)
    # plt.rc('axes', labelsize=40)
    fig, ax2 = plt.subplots(1, figsize=(14,12))
    to_pop = []
    for x in data.keys(): 
        path = f"/home/jordan/dev/nautilus/tests/test_modelcraft/base/completeness_files/{x}.json"
        if not os.path.exists(path):
            to_pop.append(x)
            continue
            
        with open(path, "r") as file_:
            c = json.load(file_)
            data[x]["nautilus_base_seq"] = 100* c["nucleic_sequenced"]/c["nucleic_total"]
            # data[x]["nucleic_total"] = c["nucleic_total"]

    for x in to_pop:
        data.pop(x)
    
    for k in data.keys():
        data[k]["pdb"] = k
        
    data = sorted(data.values(), reverse=True, key=lambda x: x["completeness"])
    names = [x["pdb"] for x in data]
    
    completeness = [x["completeness"] for x in data]
    # ax0.bar(names, completeness)
    
    resolutions = [res[x["pdb"]] for x in data]
    # ax1.scatter(resolutions, completeness)
    
    nautilus_base = [x["nautilus_base_seq"] for x in data]
    sc = ax2.scatter(nautilus_base, completeness, c=resolutions, cmap=plt.cm.coolwarm)
    
    ax2.set_xlim((0,100))
    ax2.set_ylim((0,100))
    lims = [
    0,100# max of both axes
    ]

    # now plot both limits against eachother
    cbar = fig.colorbar(sc, ax = ax2)
    cbar.set_label('Resolution / $\AA$')

    ax2.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    ax2.set_xlabel("Nautilus Sequence Completeness / %")
    ax2.set_ylabel("ML Sequence Completeness / % ")
    plt.margins(x=10, y=10)
    


    # ax0.set_xticklabels(names, rotation=90)
    plt.tight_layout()
    plt.savefig("tests/test_model/nautilus_vs_new.png", dpi=400)

if __name__ == "__main__":

    r = gemmi.read_structure("data/reference_residue.pdb")[0][0][0]

    # test_model()
    visualise_results()