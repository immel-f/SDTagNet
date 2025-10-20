from utils import *
import argparse

def convert_maptrv2_streammapnet(pkl_dir, train_split, val_split, og_pkl_name="nuscenes_map_infos_temporal"):
    train_pkl_name = f"{og_pkl_name}_train.pkl"
    val_pkl_name = f"{og_pkl_name}_val.pkl"
    test_pkl_name = f"{og_pkl_name}_test.pkl"

    train_pkl_path = os.path.join(pkl_dir, train_pkl_name)
    val_pkl_path = os.path.join(pkl_dir, val_pkl_name)
    test_pkl_path = os.path.join(pkl_dir, test_pkl_name)

    print("Loading pickles...")
    with open(train_pkl_path, "rb") as f:
        train_pkl = pickle.load(f)
    with open(val_pkl_path, "rb") as f:
        val_pkl = pickle.load(f)
    with open(test_pkl_path, "rb") as f:
        test_pkl = pickle.load(f)

    token_to_set = {}
    for log_id in train_split:
        token_to_set[log_id] = 'train'
    for log_id in val_split:
        token_to_set[log_id] = 'val'

    scene_to_original_pickles = {"train": train_pkl, "val": val_pkl, "test": test_pkl}

    new_pkl = {"train": {"samples": []},
               "val": {"samples": []},
               "test": {"samples": []}}

    print("Creating new pickles...")
    for split, pkls in scene_to_original_pickles.items():
        for sample in pkls["samples"]:

            if sample["log_id"] not in token_to_set:
                continue

            new_split = token_to_set[sample["log_id"]]
            new_pkl[new_split]["samples"].append(sample)

    return new_pkl["train"], new_pkl["val"], test_pkl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, help="Method to use to generate the splits", default="maptrv2",
                        choices=["maptr", "maptrv2"])
    parser.add_argument("--pkl_dir", type=str, help="Path to the folder of the original pickle files")
    parser.add_argument("--split_name", type=str, help="Name of the split", default="argo_geo")
    parser.add_argument("--out_dir", help="Path to the folder of the output pickle files", default=None)
    parser.add_argument("--og_pkl_name",type=str, help="Name of the original pickle files", default="argoverse_map_infos_temporal")
    args = parser.parse_args()

    method = args.method
    pkl_dir = args.pkl_dir
    split_name = args.split_name
    out_dir = args.out_dir
    og_pkl_name = args.og_pkl_name

    if out_dir is None:
        out_dir = pkl_dir
    else:
        from pathlib import Path
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Generating {split_name} split for {method}")
    new_train_pkl_name = f"{out_dir}/{split_name}_train.pkl"
    new_val_pkl_name = f"{out_dir}/{split_name}_val.pkl"
    new_test_pkl_name = f"{out_dir}/{split_name}_test.pkl"

    # Test if the original pickle files exist
    train_pkl_name = f"{og_pkl_name}_train.pkl"
    train_pkl_path = os.path.join(pkl_dir, train_pkl_name)
    print(f"Will attempt to load pickles with basename: ", og_pkl_name)
    if not os.path.exists(train_pkl_path):
        print(f"Did not find {train_pkl_path}. Use --og_pkl_name to specify the base name of your original pkl files")
        raise ValueError(f"File not found: {train_pkl_path}")
    
    # Load split name
    split_file_train = "near_extrapolation_splits/argoverse2/av2_train_split_streammapnet.txt"
    split_file_val = "near_extrapolation_splits/argoverse2/av2_val_split_streammapnet.txt"
    print(f"Loading split: {split_name} from {split_file_train}")
    with open(split_file_train) as f:
        train_split = [s.strip() for s in f.readlines()]
    print(f"Loading split: {split_name} from {split_file_val}")
    with open(split_file_val) as f:
        val_split = [s.strip() for s in f.readlines()]

    if method == "maptrv2":
        train, val, test = convert_maptrv2_streammapnet(pkl_dir, train_split, val_split, og_pkl_name)
    else:
        raise ValueError("Invalid method: ", method)
    
    print("Saving new pickles...")
    with open(new_train_pkl_name, "wb") as f:
        pickle.dump(train, f)
    with open(new_val_pkl_name, "wb") as f:
        pickle.dump(val, f)
    with open(new_test_pkl_name, "wb") as f:
        pickle.dump(test, f)

    print("Done!")


