import os
from dataclasses import dataclass

import gemmi
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt
from tqdm.keras import TqdmCallback
from model import t_model, cnn, cnn_full
import sugar_based_align as sba
import constants


@dataclass
class Params:
    dataset_base_dir: str = "dataset"
    datasets = {"train": "data/train.csv", "test": "data/test.csv"}
    shape: int = 16
    pdb_path: str = "data/pdb_files"
    map_path: str = "data/map_files"

def generator_full(dataset: str):
    df = pd.read_csv(Params.datasets[dataset])

    base_groups = constants.base_groups_full()

    for pdb_code in df["PDB"]:
        structure = gemmi.read_structure(os.path.join(Params.pdb_path, f"pdb{pdb_code}.ent"))
        grid = gemmi.read_ccp4_map(os.path.join(Params.map_path, f"{pdb_code}.map")).grid
        grid.normalize()

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

                yield box.reshape((shape, shape, shape, 1)), base_groups[residue.name]


def train():
    num_threads: int = 16
    # os.environ["OMP_NUM_THREADS"] = str(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(int(num_threads / 2))
    tf.config.threading.set_intra_op_parallelism_threads(int(num_threads / 2))

    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    _train_gen = generator_full("train")
    _test_gen = generator_full("test")

    input = tf.TensorSpec(shape=(Params.shape, Params.shape, Params.shape, 1), dtype=tf.float32)
    output = tf.TensorSpec(shape=5, dtype=tf.int64)

    train_dataset = tf.data.Dataset.from_generator(lambda: _train_gen,
                                                   output_signature=(input, output))

    test_dataset = tf.data.Dataset.from_generator(lambda: _test_gen,
                                                  output_signature=(input, output))

    model = cnn_full()
    print(model.summary())

    loss = tf.losses.categorical_crossentropy

    optimiser = tf.keras.optimizers.Adam(learning_rate=1e-3)

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
    steps_per_epoch: int = 1000
    validation_steps: int = 1000
    name: str = "categorical_model1"

    weight_path: str = f"models/{name}.best.hdf5"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        weight_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=False,
    )

    train_dataset = train_dataset.repeat(epochs*10).batch(batch_size=batch_size)

    test_dataset = test_dataset.repeat(epochs*10).batch(batch_size=batch_size)

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
    _, r = sba.superimpose_residues()

    train()

    # count = 0
    # average_box = np.zeros((16, 16, 16))
    # for d in generator("test"):
    #     # print(d[0].shape)
    #
    #     # break
    #
    #     average_box += d[0].squeeze()
    #     count += 1
    #
    # average_box /= count
    # #
    # # print(average_box, count)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # #
    # X, Y, Z = np.meshgrid(np.arange(0, 16),
    #                       np.arange(0, 16),
    #                       np.arange(0, 16))
    #
    # ax.scatter(X, Y, Z, s=10 * average_box, c=average_box)
    # # ax2.scatter(X, Y, Z, s=10 * average_box, c=average_box)
    # plt.show()
