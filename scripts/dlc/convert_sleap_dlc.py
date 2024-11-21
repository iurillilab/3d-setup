# Convert SLEAP project to DLC project. Process ONLY user-defined instances (annotations),
# and save them with the relative frames images.
# The videos folder is created but is left empty atm, as we assume to be processing
# a .slp file with embedded frames and we have no access to original videos.

import argparse
import yaml
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

import cv2
from sleap.io.dataset import load_file


def _write_config(project_name, scorer, output_dir, bodyparts, skeleton):
    # Modify here to customize config.yaml
    config = {'Task': project_name,
            'scorer': scorer,
            'date': time.strftime("%Y-%m-%d"),
            'identity': None,
            'project_path': output_dir,
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
            'SuperAnimalConversionTables': None,
            "multianimalproject": False,
            'skeleton': skeleton,
            'bodyparts': bodyparts
            }

    # save DLC config.yaml
    print(f"Saving config.yaml to {output_dir}")
    with open(output_dir / "config.yaml", "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)


def _get_video_filename(video_obj):
    video_path = video_obj.backend.source_video_available.backend.filename
    video_filename = Path(video_path).stem
    return video_filename

def _get_img_from_frame(frame):
    img = np.array(frame.image)
    return img

def _get_bodyparts_from_frame(frame):
    bodyparts = [node.name for node in frame.user_instances[0].skeleton.nodes]
    return bodyparts

def _get_points_from_frame(frame):
    points = frame.user_instances[0].points
    x_arr = np.array([p.x for p in points])
    y_arr = np.array([p.y for p in points])
    visible_arr = np.array([p.visible for p in points])
    x_arr[~visible_arr] = np.nan
    y_arr[~visible_arr] = np.nan
    return x_arr, y_arr


def _create_labels_df(bodyparts, video_filenames, frame_names, all_x_arr, all_y_arr, scorer_name):
    column_names = []
    for bp in bodyparts:
        column_names.extend([(scorer_name, bp, 'x'), (scorer_name, bp, 'y')])
    columns = pd.MultiIndex.from_tuples(column_names, names=['scorer', 'bodypart', 'coord'])

    # Create MultiIndex rows 
    row_tuples = [("labeled-data", video_filename, frame_name) 
        for video_filename, frame_name in zip(video_filenames, frame_names)]
    rows = pd.MultiIndex.from_tuples(row_tuples, names=['subdir', 'video_filename', 'frame_idx'])

    # Create data array by interleaving x and y coordinates
    data = []
    for x_arr, y_arr in zip(all_x_arr, all_y_arr):
        frame_data = []
        for x, y in zip(x_arr, y_arr):
            frame_data.extend([x, y])
        data.append(frame_data)

    # Create DataFrame
    return pd.DataFrame(data, index=rows, columns=columns)


def convert_slep_to_dlc(sleap_file, output_dir, project_name=None, scorer='SLEAP-annotated'):
    
    sleap_file = Path(sleap_file)
    output_dir = Path(output_dir)
    assert sleap_file.exists(), f"Cannot find SLEAP file: {sleap_file}"
    
    if project_name is None:
        project_name = sleap_file.stem + "_converted"

    slp_labels = load_file(str(sleap_file))

    labels_folder = output_dir / "labeled-data"
    videos_folder = output_dir / "videos"
    labels_folder.mkdir(exist_ok=True, parents=True)
    videos_folder.mkdir(exist_ok=True, parents=True)

    # Read skeleton and bodyparts from SLEAP project:
    bodyparts = [node.name for node in slp_labels.skeleton.nodes]
    skeleton = [[node.name for node in edge] for edge in slp_labels.skeleton.edges]

    # Select frames with user-defined instances:
    selected_frames = []
    for frame in slp_labels.labeled_frames:
        if frame.n_user_instances > 0:
            selected_frames.append(frame)

    # Read skeleton and bodyparts from SLEAP project:
    bodyparts = [node.name for node in slp_labels.skeleton.nodes]
    skeleton = [[node.name for node in edge] for edge in slp_labels.skeleton.edges]

    # Loop over instances and extract x and y coordinates and frame names:
    all_x_arr = []
    all_y_arr = []
    frame_names = []
    video_filenames = []
    for frame in tqdm(selected_frames, desc="Processing labelled frames:"):
        bodyparts = _get_bodyparts_from_frame(frame)
        # assert bodyparts == bodyparts
        img = _get_img_from_frame(frame)
        x_arr, y_arr = _get_points_from_frame(frame)
        
        all_x_arr.append(x_arr)
        all_y_arr.append(y_arr)
        frame_name = f"img{str(frame.frame_idx).zfill(8)}.png"

        video_filename = _get_video_filename(frame.video)
        frame_names.append(frame_name)
        video_filenames.append(video_filename)

        # Save frame in labeled-data folder:
        frames_folder = labels_folder / video_filename
        frames_folder.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(frames_folder / frame_name), img)

    # Create dataframe of labels
    df = _create_labels_df(bodyparts, video_filenames, frame_names, all_x_arr, all_y_arr, scorer)

    # Loop over video_filenames and save relative labels dataframes:
    for video_filename in df.index.get_level_values('video_filename').unique():
        video_folder = labels_folder / video_filename
        df_sub = df[df.index.get_level_values('video_filename') == video_filename].copy()

        # drop rows index names
        df_sub.index.names = [None, None, None]
        
        # Save to h5 and csv:
        # df_sub.to_hdf(video_folder / f"CollectedData_{scorer}.h5", key="data")
        df_sub.to_csv(video_folder / f"CollectedData_{scorer}.csv", index=True, header=True)

    _write_config(project_name, scorer, output_dir, bodyparts, skeleton)


# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SLEAP project to DLC project."
    )
    parser.add_argument(
        "--slp-file",
        type=str,
        required=True,
        help="Path to the SLEAP project file (.pkg.slp)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output DLC project directory",
    )

    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Name of the DLC project",
    )

    parser.add_argument(
        "--scorer",
        type=str,
        default="SLEAP-annotated",
        help="Name of the scorer",
    )

    args = parser.parse_args()

    convert_slep_to_dlc(args.slp_file, args.output_dir, args.project_name, args.scorer)
