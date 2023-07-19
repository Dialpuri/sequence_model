import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import gemmi
from dataclasses import dataclass
from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm.keras import TqdmCallback
from model import t_model, cnn
from generate_dataset import get_bounding_box, get_map_path, get_pdb_path, get_mtz_path


@dataclass
class Params:
    dataset_base_dir: str = "dataset"
    datasets = {"train": "data/train_8.csv", "test": "data/test_8.csv"}
    shape: int = 16


def sample_generator(dataset: str = "train"):
    df = pd.read_csv(Params.datasets[dataset])

    for path in df.itertuples():
        pdb_path = Path(path.path) / f"pdb{path.name}.ent"
        density_map_path = Path(path.path) / f"{path.name}_interpolated_density.map"
        base_map_path = Path(path.path) / f"{path.name}_interpolated_density.map"

        if not pdb_path.is_file():
            continue

        s = gemmi.read_structure(str(pdb_path))
        grid = gemmi.read_ccp4_map(str(map_path)).grid
        grid.normalize()

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
            'U': [1, 0], 'C': [1, 0], 'G': [0, 1], 'A': [0, 1], 'DA': [0, 1], 'DT': [1, 0], 'DG': [0, 1], 'DC': [1, 0]
        }

        for c in s[0]:
            for r in c:
                info = gemmi.find_tabulated_residue(r.name)
                if info.kind in (gemmi.ResidueInfoKind.RNA, gemmi.ResidueInfoKind.DNA):
                    if r.name not in base_types:
                        continue

                    base_midpoint = gemmi.Position(0, 0, 0)
                    base_atom_count = 0

                    n1_pos = gemmi.Position(0, 0, 0)

                    for a in r:
                        if a.name == "N1":
                            n1_pos = a.pos
                        # if a.name in base_atoms: 
                        # base_midpoint += a.pos
                        # base_atom_count += 1

                    # base_midpoint = base_midpoint/base_atom_count

                    # print(base_midpoint)

                    # noise = gemmi.Position(random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1))
                    # base_midpoint += noise
                    # shift_pos = gemmi.Position(Params.shape/2, Params.shape/2, Params.shape/2)
                    # base_startposition = base_midpoint - shift_pos
                    # base_startpoint = grid.get_nearest_point(base_startposition)

                    n1_point = grid.get_nearest_point(n1_pos)

                    arr = np.array(
                        grid.get_subarray(
                            [n1_point.u - int(Params.shape / 2), n1_point.v - int(Params.shape / 2),
                             n1_point.w - int(Params.shape / 2)], shape=[Params.shape, Params.shape, Params.shape]
                        )
                    )

                    # output_list = [1 if name == r.name else 0 for name in base_types]
                    # if sum(output_list) == 0: 
                    #     print(r.name)
                    #     continue

                    arr = arr.reshape((Params.shape, Params.shape, Params.shape, 1))
                    assert not np.any(np.isnan(arr))

                    # print(arr, base_groups[r.name])

                    yield arr, base_groups[r.name], n1_pos
                    # generate interp of subarray 

                    # exit()


def get_data(pdb):
    map_path = get_map_path(pdb)
    pdb_path = get_pdb_path(pdb)
    mtz_path = get_mtz_path(pdb)

    if not map_path and not pdb_path:
        print(map_path, pdb_path)
        return

    grid_spacing = 0.7
    # grid = gemmi.read_ccp4_map(map_path).grid

    mtz = gemmi.read_mtz_file(mtz_path)
    grid = mtz.transform_f_phi_to_map("FWT", "PHWT")
    raw_grid = grid

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

    # print(array.shape)

    grid.interpolate_values(array, transform)

    cell: gemmi.UnitCell = gemmi.UnitCell(size.x, size.y, size.z, 90, 90, 90)
    interpolated_grid = gemmi.FloatGrid(array, cell)

    # ccp4 = gemmi.Ccp4Map()
    # ccp4.grid = grid
    # ccp4.grid.unit_cell.set(size.x, size.y, size.z, 90, 90, 90)
    # ccp4.grid.spacegroup = gemmi.SpaceGroup("P1")
    # ccp4.update_ccp4_header()
    # density_path = "test/interpolation_tests/interpolated_1ais.map"
    # ccp4.write_ccp4_map(density_path)

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
        'U': [1, 0], 'C': [1, 0], 'G': [0, 1], 'A': [0, 1], 'DA': [0, 1], 'DT': [1, 0], 'DG': [0, 1], 'DC': [1, 0]
    }

    # out_s = gemmi.Structure()
    # out_m = gemmi.Model("A")
    # out_c = gemmi.Chain("A")
    # out_r = gemmi.Residue()
    # out_r.name = "X"
    # out_r.seqid = gemmi.SeqId("1")

    for c in structure[0]:
        for r in c:
            info = gemmi.find_tabulated_residue(r.name)
            if info.kind in (gemmi.ResidueInfoKind.RNA, gemmi.ResidueInfoKind.DNA):
                if r.name not in base_types:
                    continue

                base_midpoint = gemmi.Position(0, 0, 0)
                base_atom_count = 0

                repr_a = None

                for a in r:
                    if a.name in base_atoms:
                        base_midpoint += a.pos
                        base_atom_count += 1
                        repr_a = a

                base_midpoint = base_midpoint / base_atom_count
                # print("Midpoint -", base_midpoint)
                # Real space
                base_startpoint = raw_grid.get_nearest_point(base_midpoint)
                index_pos = gemmi.Vec3(base_startpoint.u, base_startpoint.v, base_startpoint.w)
                shift_pos = gemmi.Vec3(Params.shape / 2, Params.shape / 2, Params.shape / 2)
                shifted_pos = index_pos - shift_pos

                arr = np.zeros((Params.shape, Params.shape, Params.shape))

                for i in range(Params.shape):
                    for j in range(Params.shape):
                        for k in range(Params.shape):
                            ijk = gemmi.Vec3(i, j, k)

                            # if any(i > array.shape[0], j > array.shape[1], k > array.shape[2]):
                            #     outside_unitcell 

                            probe_pos = shifted_pos + ijk

                            arr[i, j, k] = interpolated_grid.get_value(int(probe_pos.x), int(probe_pos.y),
                                                                       int(probe_pos.z))

                # Interp space
                # position = gemmi.Position(transform.apply(index_pos))
                # # point = grid.get_nearest_point(position)

                # # print(base_startpoint, "->", point)

                # arr = np.array(
                #         interpolated_grid.get_subarray(
                #             [int(shifted_pos.x), int(shifted_pos.y), int(shifted_pos.z)],
                #             shape=[Params.shape,Params.shape, Params.shape]
                #         )
                #     )

                arr = arr.reshape((Params.shape, Params.shape, Params.shape, 1))

                # repr_a.pos = position

                # out_r.add_atom(repr_a)
                # out_c.add_residue(out_r)
                # out_m.add_chain(out_c)
                # out_s.add_model(out_m)
                # out_s.write_pdb("test/interpolation_tests/base_midpoint.pdb")

                return arr, base_groups[r.name], index_pos


def gen(dataset: str = "train"):
    # yield get_data("1ais")
    df = pd.read_csv(Params.datasets[dataset])
    for path in df.itertuples():
        x = get_data(path.name)
        if x:
            yield x


def generator(dataset: str = "train"):
    df = pd.read_csv(Params.datasets[dataset])
    df: pd.DataFrame = df.astype({'X': 'int', 'Y': 'int', 'Z': 'int'})
    df_np: np.ndarray = df.to_numpy()

    while True:
        for candidate in df_np:
            assert len(candidate) == 5

            pdb_code: str = candidate[0]
            X: int = candidate[1]
            Y: int = candidate[2]
            Z: int = candidate[3]
            classifications: str = candidate[4]

            density_path: str = os.path.join(
                Params.dataset_base_dir, pdb_code, f"{pdb_code}_{classifications}.map"
            )

            density_map: gemmi.FloatGrid = gemmi.read_ccp4_map(density_path).grid

            density_array: np.ndarray = np.array(
                density_map.get_subarray(
                    start=[int(X - (Params.shape / 2)),
                           int(Y - (Params.shape / 2)),
                           int(Z - (Params.shape / 2))],
                    shape=[Params.shape, Params.shape, Params.shape]
                ),
                dtype=np.float32
            )

            class_hot = [0, 0]

            if classifications == "pur":
                class_hot = [1, 0]
            elif classifications == "pyr":
                class_hot = [0, 1]
            else:
                print("Unrecognised classification, ", classifications)

            yield density_array.reshape((Params.shape, Params.shape, Params.shape, 1)), class_hot


def train():
    num_threads: int = 16
    # os.environ["OMP_NUM_THREADS"] = str(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(int(num_threads / 2))
    tf.config.threading.set_intra_op_parallelism_threads(int(num_threads / 2))

    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    _train_gen = generator("train")
    _test_gen = generator("test")

    input = tf.TensorSpec(shape=(Params.shape, Params.shape, Params.shape, 1), dtype=tf.float32)
    output = tf.TensorSpec(shape=(2), dtype=tf.int64)

    train_dataset = tf.data.Dataset.from_generator(lambda: _train_gen,
                                                   output_signature=(input, output))

    test_dataset = tf.data.Dataset.from_generator(lambda: _test_gen,
                                                  output_signature=(input, output))

    model = cnn()
    print(model.summary())

    loss = tf.losses.binary_crossentropy

    optimiser = tf.keras.optimizers.Adam(learning_rate=1e-8)

    model.compile(
        optimizer=optimiser,
        loss=loss,
        metrics=["accuracy", "categorical_accuracy"],
    )

    reduce_lr_on_plat = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.8,
        patience=5,
        verbose=1,
        mode="auto",
        cooldown=5,
        min_lr=1e-7,
    )
    epochs: int = 100
    batch_size: int = 8
    steps_per_epoch: int = 100_00
    validation_steps: int = 1000
    name: str = "test2"

    weight_path: str = f"models/{name}.best.hdf5"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        weight_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=False,
    )

    train_dataset = train_dataset.repeat(epochs).batch(batch_size=batch_size).cache()

    test_dataset = test_dataset.repeat(epochs).batch(batch_size=batch_size)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"./logs/{name}", histogram_freq=1,
        # profile_batch=(1, 10)
    )

    callbacks_list = [
        checkpoint,
        reduce_lr_on_plat,
        TqdmCallback(verbose=2),
        tensorboard_callback,
    ]

    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=0,
        use_multiprocessing=True,
    )

    model.save(f"models/{name}")


if __name__ == "__main__":
    x = next(generator())
    print(np.unique(x[0], return_counts=True))

    array = x[0].reshape((Params.shape, Params.shape, Params.shape))

    pur_grid = gemmi.FloatGrid(array)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = pur_grid
    ccp4.grid.unit_cell.set(array.shape[0]*0.7, array.shape[1]*0.7, array.shape[2]*0.7, 90, 90, 90)
    ccp4.grid.spacegroup = gemmi.SpaceGroup("P1")
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map("test/generator_tests/map.map")
    # train()
    # ...

    # average_data = np.zeros((8,8,8))
    # count = 0

    # for x in gen():
    #     data, classification, pos = x
    #     data = data.reshape(data.shape[:-1])

    #     # if count % 2 == 0: 
    #     #     average_data += data
    #     #     count += 1

    #     # average_data /= count

    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')

    #     X,Y,Z = np.meshgrid(pos.x+np.arange(0,Params.shape)*0.7,
    #                         pos.y+np.arange(0,Params.shape)*0.7,
    #                         pos.z+np.arange(0,Params.shape)*0.7)

    #     ax.scatter(X,Y,Z, s=10*data, c=data)

    #     ax.scatter(pos.x, pos.y, pos.z, s=100, c='blue')

    #     plt.savefig("test.png")
    #     exit()  
    # # plt.show()
