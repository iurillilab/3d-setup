import deeplabcut
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def run_dlc_training(dlc_path):
    # Path to your DeepLabCut project's config.yaml
    dlc_path = Path(dlc_path)
    
    dlc_config_file = Path(dlc_path) / "config.yaml" if dlc_path.is_dir() else dlc_path

    all_labels_files = list(dlc_config_file.parent.glob("labeled-data/**/CollectedData*.pkl"))

    # convert all files to h5 files:
    for labels_file in tqdm(all_labels_files, "Converting labels files"):
        df = pd.read_pickle(labels_file)
        df.to_hdf(labels_file.with_suffix(".h5"), key="data")
    # assert False


    assert dlc_config_file, f"File not found: {dlc_config_file}"

    p = deeplabcut.create_training_dataset(str(dlc_config_file))
    print("create_training_dataset: ", p)


    train_pose_config, _, _ = deeplabcut.return_train_network_path(str(dlc_config_file))
    print("train_pose_config", train_pose_config)
    augs = {
        "gaussian_noise": True,
        "elastic_transform": True,
        "rotation": 180,
        "covering": True,
        "motion_blur": True,
    }

    deeplabcut.auxiliaryfunctions.edit_config(
        train_pose_config,
        augs,
    )

    # Start training the DeepLabCut network
    deeplabcut.train_network(str(dlc_config_file), shuffle=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DLC training")
    parser.add_argument(
        "--dlc-path", type=str, required=True, help="Path to DLC project"
    )
    args = parser.parse_args()
    run_dlc_training(args.dlc_path)
