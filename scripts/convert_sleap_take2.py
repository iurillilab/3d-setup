#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Script to convert SLEAP project to DLC project
 
Usage:
$ python convertSLP2DLC.py --slp_file /path/to/<project>.pkg.slp --dlc_dir /path/to/dlc/dir
 
Arguments:
--slp_file    Path to the SLEAP project file (.pkg.slp)
--dlc_dir     Path to the output DLC project directory
 
Once done, run deeplabcut.convertcsv2h5('config.yaml', userfeedback=False) 
on the config file to obtain the .h5 files needed by DeepLabCut.
"""
from pathlib import Path
 
import os
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
import flammkuchen as fl
from pprint import pprint

dest_dir = Path("/Users/vigji/Desktop/converted_from_sleap")
slp_source_file = Path("/Users/vigji/Downloads/labels.v001.pkg.slp")
# project_name = "side-views-from-sleap"


labeled_data_dir = dest_dir / "labeled-data"
videos_dir = dest_dir / "videos"

for dir_to_create in [dest_dir, labeled_data_dir, videos_dir]:
    dir_to_create.mkdir(parents=True, exist_ok=True)

sleap_file_fl = fl.load(slp_source_file)
sleap_file_fl
# %%
with h5py.File(slp_source_file, "r") as hdf_file:
    print([k for k in hdf_file.keys() if k.startswith("video")])

# SLEAP file has structure:
#  - frames
#  - instances
#  - metadata
#  - points
#  - pred_points
#  - suggestions_json
#  - tracks_json
#  - video_0: video group, with entries 'frame_numbers', 'source_video', 'video'
#  - video_0/frame_numbers: array of frame numbers
#  - video_0/source_video: source video metadata
#  - video_0/video: video data
#  - video_1
#  - video_<n>/frame_numbers
# %%
sleap_file_fl["pred_points"]
# %%
# Open SLEAP file and process
with h5py.File(slp_source_file, "r") as hdf_file:
    video_group_keys = [val for key, val in hdf_file.items() if key.startswith("video")]
    print(video_group_keys[0]["video"][0].shape)
    metadata = json.loads(hdf_file["metadata"].attrs["json"])

    # Extract keypoints and skeleton
    keypoints = [node["name"] for node in metadata["nodes"]]
    links = metadata["skeletons"][0]["links"]
    skeleton = [[keypoints[l["source"]], keypoints[l["target"]]] for l in links]
    # config["bodyparts"] = keypoints
    # config["skeleton"] = skeleton
# %%

# %%
 
def convert_slep_to_dlc(slp_file, dlc_dir, project_name="converted_project", scorer="me"):
    """
    Convert a SLEAP project to a DeepLabCut project structure.
 
    Args:
        slp_file: Path to the SLEAP .pkg.slp file.
        dlc_dir: Output directory for the DLC project.
        project_name: Name of the converted project.
        scorer: Scorer name for DLC.
    """
 
    # Open SLEAP file and process
    with h5py.File(slp_file, "r") as hdf_file:
        video_groups = [key for key in hdf_file.keys() if key.startswith("video")]
        metadata = json.loads(hdf_file["metadata"].attrs["json"])
 
        # Extract keypoints and skeleton
        keypoints = [node["name"] for node in metadata["nodes"]]
        links = metadata["skeletons"][0]["links"]
        skeleton = [[keypoints[l["source"]], keypoints[l["target"]]] for l in links]
        config["bodyparts"] = keypoints
        config["skeleton"] = skeleton
 
        # Process each video group
        for video_group in video_groups:
            video_data = hdf_file[f"{video_group}/video"][:]
            frame_numbers = hdf_file[f"{video_group}/frame_numbers"][:]
            video_filename = os.path.basename(
                json.loads(hdf_file[f"{video_group}/source_video"].attrs["json"])["backend"]["filename"]
            )
            video_base_name = os.path.splitext(video_filename)[0]
            output_dir = os.path.join(labeled_data_dir, video_base_name)
 
            # Skip videos with too many frames
            if len(frame_numbers) > 200:
                print(f"Skipping video '{video_group}' with {len(frame_numbers)} frames.")
                continue
 
            os.makedirs(output_dir, exist_ok=True)
            print(f"Processing video: {video_filename} ({len(frame_numbers)} frames)")
 
            # Save video frames as images
            for i, (img_bytes, frame_number) in tqdm(enumerate(zip(video_data, frame_numbers)), total=len(frame_numbers)):
                img = Image.open(io.BytesIO(np.array(img_bytes, dtype=np.uint8)))
                img = np.array(img)
                frame_name = f"img{str(frame_number).zfill(8)}.png"
                cv2.imwrite(os.path.join(output_dir, frame_name), img)
 
            # Create DLC CSV
            points_dataset = hdf_file["points"]
            instances_dataset = hdf_file["instances"]
            frame_refs = {
                frame["frame_id"]: frame["frame_idx"]
                for frame in hdf_file["frames"]
                if frame["video"] == int(video_group.replace("video", ""))
            }
 
            data = []
            for frame_id, frame_idx in frame_refs.items():
                instances = [
                    inst
                    for inst in instances_dataset
                    if inst["frame_id"] == frame_id
                ]
                keypoints_data = [
                    points_dataset[inst["point_id_start"]:inst["point_id_end"]]
                    for inst in instances
                ]
                for kp_data in keypoints_data:
                    row = [frame_idx]
                    for kp in kp_data:
                        x, y = kp["x"], kp["y"]
                        row.extend([x if not np.isnan(x) else None, y if not np.isnan(y) else None])
                    data.append(row)
 
            if not data:
                print(f"No labeled data for video: {video_filename}")
                shutil.rmtree(output_dir)
                continue
 
            columns = ["frame"] + [f"{bp}_{c}" for bp in keypoints for c in ("x", "y")]
            labels_df = pd.DataFrame(data, columns=columns)
            scorer_row = ["scorer"] + [scorer] * (len(columns) - 1)
            bodyparts_row = ["bodyparts"] + [bp for bp in keypoints for _ in ("x", "y")]
            coords_row = ["coords"] + ["x", "y"] * len(keypoints)
            header_df = pd.DataFrame([scorer_row, bodyparts_row, coords_row], columns=columns)
            final_df = pd.concat([header_df, labels_df], ignore_index=True)
            final_df.to_csv(os.path.join(output_dir, f"CollectedData_{scorer}.csv"), index=False, header=None)
 
        # Save DLC config.yaml
        with open(os.path.join(dlc_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

            
 
    print(f"Conversion complete! DLC project saved to: {dlc_dir}")
 

 config = {
    "Task": project_name,
    "scorer": scorer,
    "date": time.strftime("%Y-%m-%d"),
    "project_path": dlc_dir,
    "video_sets": {},
    "bodyparts": [],
    "skeleton": [],
}
 
# Main entry point
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Convert SLEAP project to DLC project.")
    parser.add_argument("--slp_file", type=str, required=True, help="Path to the SLEAP project file (.pkg.slp)")
    parser.add_argument("--dlc_dir", type=str, required=True, help="Path to the output DLC project directory")
    args = parser.parse_args()
 
    if not os.path.exists(args.slp_file):
        raise FileNotFoundError(f"Cannot find SLEAP file: {args.slp_file}")
    if not os.path.exists(args.dlc_dir):
        os.makedirs(args.dlc_dir)
 
    convert_slep_to_dlc(args.slp_file, args.dlc_dir)



'''
previous code:

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by: Adam Gosztolai
 
Script to convert SLEAP project to DLC project
 
Usage:
$ python convertSLP2DLC.py --slp_file /path/to/<project>.pkg.slp --dlc_dir /path/to/dlc/dir
 
Arguments:
--slp_file    Path to the SLEAP project file (.pkg.slp)
--dlc_dir      Path to the output DLC project directory
 
Once done, run deeplabcut.convertcsv2h5('config.yaml', userfeedback=False) on the
config file to obtain the .h5 files that Deeplabcut needs.
