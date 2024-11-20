#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert SLEAP project to DLC project
 
Usage:
$ python convertSLP2DLC.py --slp_file /path/to/<project>.pkg.slp --dlc_dir /path/to/dlc/dir
 
Arguments:
--slp_file    Path to the SLEAP project file (.pkg.slp)
--dlc_dir     Path to the output DLC project directory
--run_dlc     Run DLC training (1) or not (0)
 
Once done, run deeplabcut.convertcsv2h5('config.yaml', userfeedback=False) 
on the config file to obtain the .h5 files needed by DeepLabCut.
"""

import argparse
from pathlib import Path
import time
import h5py
import json
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
from PIL import Image
import yaml
import cv2
import io


def convert_slep_to_dlc(
    slp_file, dlc_dir, project_name="converted_project", scorer="me"
):
    """
    Convert a SLEAP project to a DeepLabCut project structure.

    Args:
        slp_file: Path to the SLEAP .pkg.slp file.
        dlc_dir: Output directory for the DLC project.
        project_name: Name of the converted project.
        scorer: Scorer name for DLC.
    """
    # Convert paths to Path objects
    slp_file = Path(slp_file)
    dlc_dir = Path(dlc_dir)

    # Prepare DLC config dictionary
    config = {
        "Task": project_name,
        "scorer": scorer,
        "date": time.strftime("%Y-%m-%d"),
        "project_path": str(dlc_dir),
        "video_sets": {},
        "bodyparts": [],
        "skeleton": [],
    }

    # Create necessary directories
    dlc_dir.mkdir(parents=True, exist_ok=True)
    labeled_data_dir = dlc_dir / "labeled-data"
    videos_dir = dlc_dir / "videos"
    labeled_data_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)

    # Open SLEAP file and process
    with h5py.File(slp_file, "r") as hdf_file:
        video_groups = [
            f"video{i}"
            for i in range(
                len(
                    [
                        k
                        for k in hdf_file.keys()
                        if k.startswith("video") and k[5:].isdigit()
                    ]
                )
            )
        ]
        metadata = json.loads(hdf_file["metadata"].attrs["json"])

        # Extract keypoints and skeleton
        keypoints = [node["name"] for node in metadata["nodes"]]
        links = metadata["skeletons"][0]["links"]
        skeleton = [[keypoints[l["source"]], keypoints[l["target"]]] for l in links]
        config["bodyparts"] = keypoints
        config["skeleton"] = skeleton

        # Pre-index instances by frame_id
        instances_dataset = hdf_file["instances"]
        frame_ids = instances_dataset["frame_id"][:]
        point_id_starts = instances_dataset["point_id_start"][:]
        point_id_ends = instances_dataset["point_id_end"][:]

        frame_to_instances = {}
        for idx, frame_id in enumerate(frame_ids):
            frame_to_instances.setdefault(frame_id, []).append(idx)

        # Read all points once
        points_dataset = hdf_file["points"][:]

        # Process each video group
        for video_group in video_groups:
            print(f"Processing video group: {video_group}")
            video_data = hdf_file[f"{video_group}/video"][:]
            frame_numbers = hdf_file[f"{video_group}/frame_numbers"][:]
            video_filename = Path(
                json.loads(hdf_file[f"{video_group}/source_video"].attrs["json"])[
                    "backend"
                ]["filename"]
            ).name
            video_base_name = video_filename.split(".")[0]
            output_dir = labeled_data_dir / video_base_name

            if len(frame_numbers) > 200:
                print(
                    f"Skipping video '{video_group}' with {len(frame_numbers)} frames."
                )
                continue

            print(f"Processing video: {video_filename} ({len(frame_numbers)} frames)")
            if len(frame_numbers) == 0:
                continue
            output_dir.mkdir(exist_ok=True)

            # Save video frames as images
            for i, (img_bytes, frame_number) in tqdm(
                enumerate(zip(video_data, frame_numbers)),
                total=len(frame_numbers),
                desc="Saving frames",
            ):
                img = Image.open(io.BytesIO(np.array(img_bytes, dtype=np.uint8)))
                img = np.array(img)
                frame_name = f"img{str(frame_number).zfill(8)}.png"
                cv2.imwrite(str(output_dir / frame_name), img)

            # Create DLC CSV
            frames_dataset = hdf_file["frames"][:]
            frame_refs = {
                frame["frame_id"]: frame["frame_idx"]
                for frame in frames_dataset
                if frame["video"] == int(video_group.replace("video", ""))
            }

            data = []

            for frame_id, frame_idx in tqdm(
                frame_refs.items(), desc="Processing frames"
            ):
                instance_indices = frame_to_instances.get(frame_id, [])
                # print("frame_id", frame_id, "frame_idx", frame_idx)
                # assert that all frame_idxs correspond to saved frames
                saved_frames_files = [f.name for f in output_dir.glob("*.png")]
                saved_frames_numbers = [
                    int(f.split("img")[1].split(".")[0]) for f in saved_frames_files
                ]
                # assert
                if not instance_indices or frame_idx not in saved_frames_numbers:
                    print(
                        f"Skipping frame {frame_idx} because it is not in saved_frames_numbers {saved_frames_numbers}"
                    )
                    continue

                for idx in instance_indices:
                    start = point_id_starts[idx]
                    end = point_id_ends[idx]
                    kp_data = points_dataset[start:end]

                    row = [frame_idx]
                    # Vectorized handling of keypoints
                    x = np.where(np.isnan(kp_data["x"]), None, kp_data["x"])
                    y = np.where(np.isnan(kp_data["y"]), None, kp_data["y"])
                    row.extend(x.tolist())
                    row.extend(y.tolist())
                    data.append(row)

            if not data:
                print(f"No labeled data for video: {video_filename}")
                shutil.rmtree(output_dir)
                continue

            columns = ["frame"] + [f"{bp}_{c}" for bp in keypoints for c in ("x", "y")]
            # print(data)
            labels_df = pd.DataFrame(data, columns=columns)
            scorer_header = [scorer] * (len(columns) - 1)
            bodyparts_header = [bp for bp in keypoints for _ in ("x", "y")]
            coords_header = ["x", "y"] * len(keypoints)

            # Create MultiIndex for columns
            multi_columns = pd.MultiIndex.from_arrays(
                [scorer_header, bodyparts_header, coords_header],
                names=["scorer", "bodyparts", "coords"],
            )

            # Prepare data by dropping header rows
            data_df = labels_df  # final_df.iloc[3:].copy()
            # Generate relative frame paths
            data_df["maindir"] = "labeled-data"
            data_df["subdir"] = video_base_name
            data_df["frame"] = data_df["frame"].apply(
                lambda x: f"img{int(x):08}.png"
            )
            # set as multiindex the sequence of maindir, subdir, frame, without keeping the name of the columns
            data_df = data_df.set_index(["maindir", "subdir", "frame"], drop=True)
            # set the multiindex names to None
            data_df.index.names = [None, None, None]

            # Assign MultiIndex columns
            data_df.columns = multi_columns

            # Create the multilevel_df DataFrame
            multilevel_df = data_df.astype(float)
            multilevel_df.to_hdf(output_dir / f"CollectedData_{scorer}.h5", key="data")
            # Flatten the MultiIndex index and drop the names of the levels
            # multilevel_df = multilevel_df.reset_index(names=None)

            # Write to CSV while keeping the column MultiIndex
            multilevel_df.to_csv(output_dir / f"CollectedData_{scorer}.csv", index=True, header=True)

            
            # multilevel_df.to_csv(output_dir / f"CollectedData_{scorer}.csv", index=True, header=True, index_label=False)

        # Update config with additional required parameters
        config.update(
            {
                "start": 0,
                "stop": 1,
                "numframes2pick": 20,
                "TrainingFraction": [0.95],
                "iteration": 3,
                "default_net_type": "resnet_50",
                "default_augmenter": "default",
                "snapshotindex": -1,
                "batch_size": 8,
            }
        )

        # Save DLC config.yaml
        with open(dlc_dir / "config.yaml", "w") as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

    print(f"Conversion complete! DLC project saved to: {dlc_dir}")


# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SLEAP project to DLC project."
    )
    parser.add_argument(
        "--slp_file",
        type=str,
        required=True,
        help="Path to the SLEAP project file (.pkg.slp)",
    )
    parser.add_argument(
        "--dlc_dir",
        type=str,
        required=True,
        help="Path to the output DLC project directory",
    )
    parser.add_argument(
        "--run_dlc",
        action="store_true",
        help="Run DLC training after conversion",
        default=False,
    )

    args = parser.parse_args()

    slp_path = Path(args.slp_file)
    dlc_path = Path(args.dlc_dir)

    if not slp_path.exists():
        raise FileNotFoundError(f"Cannot find SLEAP file: {slp_path}")
    if not dlc_path.exists():
        dlc_path.mkdir(parents=True)

    convert_slep_to_dlc(slp_path, dlc_path)

    # assert False
    # %%
    if args.run_dlc:
        import deeplabcut

        # Path to your DeepLabCut project's config.yaml
        path_config_file = str(dlc_path / "config.yaml")

        p = deeplabcut.create_training_dataset(path_config_file)
        # %%
        # %%
        train_pose_config, _, _ = deeplabcut.return_train_network_path(path_config_file)
        print("train_pose_config", train_pose_config)
        augs = {
            "gaussian_noise": True,
            "elastic_transform": True,
            "rotation": 180,
            "covering": True,
            "motion_blur": True,
        }
        # %%
        deeplabcut.auxiliaryfunctions.edit_config(
            train_pose_config,
            augs,
        )

        # Start training the DeepLabCut network
        deeplabcut.train_network(path_config_file, shuffle=1)

    # Optionally, evaluate the trained network
    # deeplabcut.evaluate_network(config_path)
