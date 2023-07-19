import gemmi 
from generate_dataset import *
import tensorflow as tf
import tensorflow_addons as tfa
import os


@dataclass
class Params:
    dataset_base_dir: str = "dataset"
    datasets = { "train": "data/train_8.csv", "test": "data/test_8.csv"}
    shape: int = 8

def predict_data(pdb):    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
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

    grid.interpolate_values(array, transform)
    
    cell: gemmi.UnitCell = gemmi.UnitCell(size.x, size.y, size.z, 90, 90, 90)
    interpolated_grid = gemmi.FloatGrid(array, cell)
    
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
    
    model = tf.keras.models.load_model(
                "models/test1.best.hdf5",
                custom_objects={
                    "sigmoid_focal_crossentropy": tfa.losses.sigmoid_focal_crossentropy
                },
                compile=False,
            )

    for c in structure[0]:
        for r in c: 
            info = gemmi.find_tabulated_residue(r.name)
            if info.kind in (gemmi.ResidueInfoKind.RNA, gemmi.ResidueInfoKind.DNA):
                if r.name not in base_types: 
                    continue
                
                base_midpoint = gemmi.Position(0,0,0)
                base_atom_count = 0 
                
                repr_a = None
                                
                for a in r: 
                    if a.name in base_atoms: 
                        base_midpoint += a.pos
                        base_atom_count += 1
                        repr_a = a
                        
                base_midpoint = base_midpoint/base_atom_count
                base_startpoint = raw_grid.get_nearest_point(base_midpoint)
                index_pos = gemmi.Vec3(base_startpoint.u, base_startpoint.v, base_startpoint.w)

                # Interp space
                position = gemmi.Position(transform.apply(index_pos))
                # point = grid.get_nearest_point(position)
                
                # # print(base_startpoint, "->", point)
                
                arr = np.array(
                        interpolated_grid.get_subarray(
                            [int(position.x), int(position.y), int(position.z)],
                            shape=[Params.shape,Params.shape, Params.shape]
                        ), dtype=np.float32
                    )
                
                arr = arr.reshape((1, Params.shape, Params.shape, Params.shape, 1))
                
                # print(model.summary())
                
                assert arr.shape == (1,8,8,8,1)
                
                prediction = model.predict(arr, verbose=2)
                arg_max = np.argmax(prediction, axis=-1)


                print(r.name, prediction, base_groups[r.name])
                
                # if arg_max == [1,0]:
                #     classifications = "pur"
                   
                # elif arg_max == [0,1]:
                #     classifications == "pyr"
                

if __name__ == "__main__":
    predict_data("1ais")