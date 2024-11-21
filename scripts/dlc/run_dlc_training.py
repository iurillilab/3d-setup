import deeplabcut
import argparse


def run_dlc_training(dlc_path):
    # Path to your DeepLabCut project's config.yaml
    path_config_file = str(dlc_path / "config.yaml")

    p = deeplabcut.create_training_dataset(path_config_file)

    train_pose_config, _, _ = deeplabcut.return_train_network_path(path_config_file)
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
    deeplabcut.train_network(path_config_file, shuffle=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DLC training")
    parser.add_argument(
        "--dlc_path", type=str, required=True, help="Path to DLC project"
    )
    args = parser.parse_args()
    run_dlc_training(args.dlc_path)
