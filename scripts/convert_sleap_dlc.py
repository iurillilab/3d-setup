#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


def convert_slep_to_dlc(slp_file, dlc_dir, project_name="converted_project", scorer="me"):
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
        video_groups = [f"video{i}" for i in range(len([k for k in hdf_file.keys() if k.startswith("video") and k[5:].isdigit()]))]
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
                json.loads(hdf_file[f"{video_group}/source_video"].attrs["json"])["backend"]["filename"]
            ).name
            video_base_name = video_filename.split(".")[0]
            output_dir = labeled_data_dir / video_base_name

            if len(frame_numbers) > 200:
                print(f"Skipping video '{video_group}' with {len(frame_numbers)} frames.")
                continue

            print(f"Processing video: {video_filename} ({len(frame_numbers)} frames)")
            if len(frame_numbers) == 0:
                continue    
            output_dir.mkdir(exist_ok=True)


            # Save video frames as images
            for i, (img_bytes, frame_number) in tqdm(
                enumerate(zip(video_data, frame_numbers)),
                total=len(frame_numbers),
                desc="Saving frames"
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

            for frame_id, frame_idx in tqdm(frame_refs.items(), desc="Processing frames"):
                instance_indices = frame_to_instances.get(frame_id, [])
                if not instance_indices:
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
            labels_df = pd.DataFrame(data, columns=columns)
            scorer_row = ["scorer"] + [scorer] * (len(columns) - 1)
            bodyparts_row = ["bodyparts"] + [bp for bp in keypoints for _ in ("x", "y")]
            coords_row = ["coords"] + ["x", "y"] * len(keypoints)
            header_df = pd.DataFrame([scorer_row, bodyparts_row, coords_row], columns=columns)
            final_df = pd.concat([header_df, labels_df], ignore_index=True)
            final_df.to_csv(output_dir / f"CollectedData_{scorer}.csv", index=False, header=None)

        # Update config with additional required parameters
        config.update({
            "start": 0,
            "stop": 1,
            "numframes2pick": 20,
            "TrainingFraction": [0.95],
            "iteration": 3,
            "default_net_type": "resnet_50",
            "default_augmenter": "default", 
            "snapshotindex": -1,
            "batch_size": 8
        })

        # Save DLC config.yaml
        with open(dlc_dir / "config.yaml", "w") as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

    print(f"Conversion complete! DLC project saved to: {dlc_dir}")

# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SLEAP project to DLC project.")
    parser.add_argument("--slp_file", type=str, required=True, help="Path to the SLEAP project file (.pkg.slp)")
    parser.add_argument("--dlc_dir", type=str, required=True, help="Path to the output DLC project directory")
    args = parser.parse_args()

    slp_path = Path(args.slp_file)
    dlc_path = Path(args.dlc_dir)

    if not slp_path.exists():
        raise FileNotFoundError(f"Cannot find SLEAP file: {slp_path}")
    if not dlc_path.exists():
        dlc_path.mkdir(parents=True)

    convert_slep_to_dlc(slp_path, dlc_path)

    import deeplabcut

    # Path to your DeepLabCut project's config.yaml
    path_config_file = str(dlc_path / "config.yaml")

    deeplabcut.create_training_dataset(path_config_file)
    train_pose_config, _ = deeplabcut.return_train_network_path(config_path)
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

    # Optionally, evaluate the trained network
    # deeplabcut.evaluate_network(config_path)


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
 
"""
 
 
import argparse
import io
import json
import yaml
import os
import time
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
 
import PIL.Image as Image
import cv2
 
 
def extract_frames_from_pkg_slp(file_path, base_output_dir, project_name='converted_project', scorer='me'):
 
    config = {'Task': project_name,
              'scorer': scorer,
              'date': time.strftime("%Y-%m-%d"),
              'identity': None,
              'project_path': base_output_dir,
              'engine': 'pytorch',
              'video_sets': {},
              'start': 0,
              'stop': 1,
              'numframes2pick': 20,
              'skeleton_color': 'black',
              'pcutoff': 0.6,
              'dotsize': 12,
              'alphavalue': 0.6,
              'colormap': 'rainbow',
              'TrainingFraction': [0.95],
              'iteration': 0,
              'default_net_type': 'resnet_50',
              'default_augmenter': 'default',
              'snapshotindex': -1,
              'detector_snapshotindex': -1,
              'batch_size': 8,
              'detector_batch_size': 1,
              'cropping': False,
              'x1': 0,
              'x2': 640,
              'y1': 277,
              'y2': 624,
              'corner2move2': [50, 50],
              'move2corner': True,
              'SuperAnimalConversionTables': None
              }
   
   
    #create directories
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        os.makedirs(os.path.join(base_output_dir,"labeled-data"))
        os.makedirs(os.path.join(base_output_dir,"videos"))
       
    #parse .slp file
    with h5py.File(file_path, 'r') as hdf_file:
        # Identify video names
        video_names = {}
        for video_group_name in hdf_file.keys():
            if video_group_name.startswith('video'):
                source_video_path = f'{video_group_name}/source_video'
                if source_video_path in hdf_file:
                    source_video_json = hdf_file[source_video_path].attrs['json']
                    source_video_dict = json.loads(source_video_json)
                    video_filename = source_video_dict['backend']['filename']
                    video_names[video_group_name] = video_filename
                   
        # Extract and save images for each video
        for video_group, video_filename in list(video_names.items()):
            print("processing ", video_filename)
            data_frames = []
            scorer_row, bodyparts_row, coords_row = None, None, None
            output_dir = os.path.join(
                base_output_dir, "labeled-data", os.path.basename(video_filename).split('.')[0]
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
 
            # extract labeled frames and save them in a separate directory for each video
            if video_group in hdf_file and 'video' in hdf_file[video_group]:
                video_data = hdf_file[f'{video_group}/video'][:]
                frame_numbers = hdf_file[f'{video_group}/frame_numbers'][:]
                frame_names = []
 
                if len(frame_numbers) > 200:
                    print('Too many frames, skipping video {}, {}'.format(video_group, frame_numbers))
                    print(f"No labeled data found for video {video_filename}")
                    # Empty folder content and unlink folder
                    for file in os.listdir(output_dir):
                        os.unlink(os.path.join(output_dir, file))
                    os.rmdir(output_dir)
                    continue
 
                for i, (img_bytes, frame_number) in tqdm(list(enumerate(zip(video_data, frame_numbers)))):
                    if len(frame_numbers) > 200:
                        print('Too many frames, skipping video {}, {}'.format(video_group, frame_numbers))
                        print(f"No labeled data found for video {video_filename}")
                        # Empty folder content and unlink folder
                        for file in os.listdir(output_dir):
                            os.unlink(os.path.join(output_dir, file))
                            os.rmdir(output_dir)
                        continue
                    img = Image.open(io.BytesIO(np.array(img_bytes, dtype=np.uint8)))
                    img = np.array(img)
                    if i==0:
                        video_path = os.path.join(base_output_dir,'videos',video_names[video_group].split('/')[-1])
                        config['video_sets'][video_path] = {'crop': f'0, {img.shape[1]}, 0, {img.shape[0]}'}
                    frame_name = f"img{str(frame_number).zfill(8)}.png"
                    cv2.imwrite(f"{output_dir}/{frame_name}", img)
                    frame_names.append(frame_name)
                    # print(f"Saved frame {frame_number} as {frame_name}")
                   
                   
            # extract coordinates and save them in a separate directory for each video
            if video_group in hdf_file and 'frames' in hdf_file:
                frames_dataset = hdf_file['frames']
                frame_references = {
                    frame['frame_id']: frame['frame_idx']
                    for frame in frames_dataset
                    if frame['video'] == int(video_group.replace('video', ''))
                }
 
                if len(frame_numbers) > 200:
                    print('Too many frames, skipping video {}, {}'.format(video_group, frame_numbers))
                    print(f"No labeled data found for video {video_filename}")
                    # Empty folder content and unlink folder
                    for file in os.listdir(output_dir):
                        os.unlink(os.path.join(output_dir, file))
                        os.rmdir(output_dir)
                    continue
 
 
                # Extract instances and points
                points_dataset = hdf_file['points']
                instances_dataset = hdf_file['instances']
                # print(instances_dataset)
               
                data = []
                for frame_id in tqdm(frame_references.keys()):
                    for i in range(len(instances_dataset)):
                        try:
                            if frame_id==instances_dataset[i]['frame_id']:
                                point_id_start = instances_dataset[i+1]['point_id_start']
                                point_id_end = instances_dataset[i+1]['point_id_end']
                                break
                        except IndexError:
                            continue
 
                    points = points_dataset[point_id_start:point_id_end]
 
                    keypoints_flat = []
                    for kp in points:
                        x, y, vis = kp['x'], kp['y'], kp['visible']
                        if np.isnan(x) or np.isnan(y) or vis==False:
                            x, y = None, None
                        keypoints_flat.extend([x, y])
 
                    frame_idx = frame_references[frame_id]
                    if len(keypoints_flat) > 0:
                        data.append([frame_idx] + keypoints_flat)
 
                if len(data) == 0:
                    print(f"No labeled data found for video {video_filename}")
                    # Empty folder content and unlink folder
                    for file in os.listdir(output_dir):
                        os.unlink(os.path.join(output_dir, file))
                    os.rmdir(output_dir)
                    continue
                       
                   
                # parse data
                print("parsing data")
                metadata_json = hdf_file['metadata'].attrs['json']
                metadata_dict = json.loads(metadata_json)
                nodes = metadata_dict['nodes']
                links = metadata_dict['skeletons'][0]['links']
               
                keypoints = [node['name'] for node in nodes]
                skeleton = [[keypoints[l['source']], keypoints[l['target']]] for l in links]
                config['skeleton'] = skeleton
               
                keypoints_ids = [n['id'] for n in metadata_dict['skeletons'][0]['nodes']]
                keypoints_ordered = [keypoints[idx] for idx in keypoints_ids]
                config['bodyparts'] = keypoints_ordered
 
                columns = [
                    'frame'
                ] + [
                    f'{kp}_x' for kp in keypoints_ordered
                ] + [
                    f'{kp}_y' for kp in keypoints_ordered
                ]
                # print(columns, '\n', [len(d) for d in data])
                scorer_row = ['scorer'] + [f'{scorer}'] * (len(columns) - 1)
                bodyparts_row = ['bodyparts'] + [f'{kp}' for kp in keypoints_ordered for _ in (0, 1)]
                coords_row = ['coords'] + ['x', 'y'] * len(keypoints_ordered)
 
                print("creating dataframe")
                labels_df = pd.DataFrame(data, columns=columns)
                video_base_name = os.path.basename(video_filename).split('.')[0]
                labels_df['frame'] = labels_df['frame'].apply(
                    lambda x: (
                        f"labeled-data/{video_base_name}/"
                        f"img{str(int(x)).zfill(8)}.png"
                    )
                )
                labels_df = labels_df.groupby('frame', as_index=False).first()
                data_frames.append(labels_df)
                   
                # Combine all data frames into a single DataFrame
                combined_df = pd.concat(data_frames, ignore_index=True)
                   
                header_df = pd.DataFrame(
                    [scorer_row, bodyparts_row, coords_row],
                    columns=combined_df.columns
                )
                final_df = pd.concat([header_df, combined_df], ignore_index=True)
                final_df.columns = [None] * len(final_df.columns)  # Set header to None
               
                # Save concatenated labels
                final_df.to_csv(os.path.join(output_dir, f"CollectedData_{scorer}.csv"), index=False, header=None)
    print("All folders written")
    with open(os.path.join(base_output_dir, 'config.yaml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
   
parser = argparse.ArgumentParser()
parser.add_argument("--slp_file", type=str)
parser.add_argument("--dlc_dir", type=str)
args = parser.parse_args()
slp_file = args.slp_file
dlc_dir = args.dlc_dir
 
print(f"Converting SLEAP project to DLC project located at {dlc_dir}")
 
# Check provided SLEAP path exists
if not os.path.exists(slp_file):
    raise NotADirectoryError(f"did not find the file {slp_file}")
 
# Extract and save labeled data from SLEAP project
extract_frames_from_pkg_slp(slp_file, dlc_dir)

'''