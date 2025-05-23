from movement.io.save_poses import to_dlc_file
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import cv2
import shutil

from pathlib import Path
import yaml
from datetime import datetime

# Stack keypoints and view dimensions to create combined labels
def stack_keypoints_views(dataset):
    keypoints = dataset.keypoints.values
    views = dataset.view.values

    # Create new combined labels
    new_keypoint_labels = []
    for view in views:
        for keypoint in keypoints:
            combined_label = f"{keypoint}_{view}"
            combined_label = combined_label.replace("-", "_")
            new_keypoint_labels.append(combined_label)
    print(new_keypoint_labels)

    # Stack position and confidence data using a temporary dimension name
    stacked_pos = dataset.position.stack(combined_keypoints=("view", "keypoints"))
    stacked_conf = dataset.confidence.stack(combined_keypoints=("view", "keypoints"))

    # Rename the stacked dimension coordinates
    stacked_pos = stacked_pos.assign_coords(combined_keypoints=new_keypoint_labels)
    stacked_conf = stacked_conf.assign_coords(combined_keypoints=new_keypoint_labels)

    # Create new dataset with stacked dimensions
    stacked_ds = xr.Dataset({"position": stacked_pos, "confidence": stacked_conf})

    # Rename the dimension to 'keypoints'
    stacked_ds = stacked_ds.rename({"combined_keypoints": "keypoints"})

    # Copy attributes from original dataset
    for attr_name, attr_value in dataset.attrs.items():
        stacked_ds.attrs[attr_name] = attr_value

    return stacked_ds


def export_ds_to_dlc_annotation(
    labels_ds: xr.Dataset,
    frames_names: list,
    main_output_folder: Path,
    video_name: str = "test_movie.avi",
):

    video_name = video_name.split(".")[0]
    #final_name_csv = f"{video_name}_movement.csv"
    #final_name_h5 = f"{video_name}_movement.h5"
    final_name_csv = f"CollectedData.csv"
    final_name_h5 = f"CollectedData.h5"

    main_output_folder = Path(main_output_folder)
    output_folder = main_output_folder / video_name
    output_folder.mkdir(parents=True, exist_ok=True)

    TEMP_LABELS_FILENAME = "labels.h5"
    fname, suffix = TEMP_LABELS_FILENAME.split(".")
    test_labels_individual_filename = f"{fname}_individual_0.{suffix}"

    to_dlc_file(labels_ds, output_folder / TEMP_LABELS_FILENAME, split_individuals=True)
    df = pd.read_hdf(output_folder / test_labels_individual_filename)

    # change df index to be multiindex with first level "labeled-data", second level movie_name.split(".")[0], third name frames_names:
    df.index = pd.MultiIndex.from_tuples(
        [
            ("labeled-data", video_name.split(".")[0], frames_names[i])
            for i in range(len(frames_names))
        ]
    )
    df.to_csv(output_folder / final_name_csv) 
    df.to_hdf(output_folder / final_name_h5, key="df", mode="w")

    (output_folder / test_labels_individual_filename).unlink()


def extract_keyframes(video_path, dataset, n_frames=20):
    """
    Extract every `stride`-th frame from video and matching prediction from dataset.

    Returns:
        keyframe_imgs (list of np.ndarray): Extracted frames as images
        keyframe_ds (xarray.Dataset): Subset of original dataset with only keyframes
        keyframe_indices (list of int): Indices of keyframes
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # total_frames = min(total_frames, 1000)  # Limit to first 1000 frames
    # keyframe_indices = list(range(0, total_frames, stride))
    keyframe_indices = np.random.randint(0, total_frames, size=n_frames).tolist()
    
    keyframe_imgs = []
    
    for i in keyframe_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {i} from {video_path}")
        keyframe_imgs.append(frame)

    cap.release()

    # Slice xarray dataset at those frames
    keyframe_ds = dataset.sel(time=keyframe_indices)

    return keyframe_imgs, keyframe_ds, keyframe_indices

# add folder from other script 
def save_keyframes_to_disk(frames, frame_names, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for frame, frame_name in zip(frames, frame_names):
        path = Path(output_dir) / frame_name
        # path = Path(output_dir) / f"{prefix}_{i:06d}.png"
        cv2.imwrite(str(path), frame)

def generate_config(scorer, project_path, video_paths, bodyparts):
    """
    Generate a DeepLabCut config.yaml file with specified parameters
    """
    # Get current date
    date = datetime.now().strftime("%b%d")
    
    # Create video_sets dict
    video_sets = {}
    for video_path in video_paths:
        video_sets[str(video_path)] = {"crop": "0, 1108, 0, 752"}
    
    # Create skeleton based on bodyparts structure
    skeleton = []
    
    # Group bodyparts by prefix
    grouped_bodyparts = {}
    for bp in bodyparts:
        prefix = bp.split('_')[0]
        if prefix not in grouped_bodyparts:
            grouped_bodyparts[prefix] = []
        grouped_bodyparts[prefix].append(bp)
    
    # Create skeleton for each group
    # for prefix, parts in grouped_bodyparts.items():
    #     if len(parts) >= 3:  # Only create skeleton if at least 3 points
    #         if parts[0] == parts[-1]:  # Skip if already a closed loop
    #             skeleton.append(parts)
    #         else:
    #             # Create a closed loop for certain parts
    #             if prefix in ["nose", "eye", "ear", "iris"]:
    #                 skeleton.append(parts + [parts[0]])
    #             else:
    #                 skeleton.append(parts)
    
    # Create config dictionary
    config = {
        "Task": "mouseface",
        "scorer": scorer,
        "date": date,
        "multianimalproject": False,
        "identity": None,
        
        "project_path": str(project_path),
        
        "video_sets": video_sets,
        "bodyparts": bodyparts,
        
        "start": 0,
        "stop": 1,
        "numframes2pick": 20,
        
        "skeleton": skeleton,
        "skeleton_color": "black",
        "pcutoff": 0.6,
        "dotsize": 12,
        "alphavalue": 0.7,
        "colormap": "rainbow",
        
        "TrainingFraction": [0.95],
        "iteration": 0,
        "default_net_type": "resnet_50",
        "default_augmenter": "default",
        "snapshotindex": -1,
        "batch_size": 8,
        
        "cropping": False,
        "x1": 0,
        "x2": 640,
        "y1": 277,
        "y2": 624,
        
        "corner2move2": [50, 50],
        "move2corner": True
    }
    
    return config


def save_config(config, output_folder: Path):
    """Save the config dictionary as a YAML file with proper formatting"""
    path = output_folder / "config.yaml"
    
    # Convert config to string with proper formatting
    yaml_str = ""
    for key, value in config.items():
        if value is None:
            yaml_str += f"{key}:\n"
        else:
            value_str = yaml.dump({key: value}, default_flow_style=False)
            yaml_str += value_str
    
    # Write to file
    path.write_text(yaml_str)
    print(f"Config saved to {path}")


####################
target_folder = Path(r'D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_project\project')
data_folder = Path(r'D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_project')

# create a new folder
target_folder.mkdir(parents=True, exist_ok=True)

labels_folder = target_folder / "labeled-data"
if labels_folder.exists():
    for file in labels_folder.rglob("*"):
        if file.is_file():
            file.unlink()
labels_folder.mkdir(parents=True, exist_ok=True)

videos_folder = target_folder / "videos"
if videos_folder.exists():
    for file in videos_folder.rglob("*"):
        if file.is_file():
            file.unlink()
videos_folder.mkdir(parents=True, exist_ok=True)


tiled_movies = [file if file.parent.name == data_folder.name else None for file in data_folder.rglob("*.mp4")]
tiled_datasets = [file if file.parent.name == data_folder.name else None for file in data_folder.rglob("*.h5")]
tiled_movies = [file for file in tiled_movies if file is not None]
tiled_datasets = [file for file in tiled_datasets if file is not None]

for video_path, ds_path in zip(tiled_movies, tiled_datasets):
    video_path = Path(video_path)
    ds_path = Path(ds_path)
    video_name = video_path.stem

    ds = xr.open_dataset(
        ds_path,
        #engine="h5netcdf",
    )

    frames, sliced_ds, frame_idxs = extract_keyframes(video_path, ds, n_frames=20)

    # along the keypoints dimension, drop all keypoints that contain "limbmid"
    sliced_ds = sliced_ds.isel(keypoints=~sliced_ds.keypoints.str.contains("limbmid"))

    # Stack entries along keypoint axis, making new labels
    ds_stacked = stack_keypoints_views(sliced_ds)

    # Name of the frames we'll save in the labels folder:
    frames_names = [f"frame_{i:06d}.png" for i in frame_idxs]

    output_folder = labels_folder / video_name
    output_folder.mkdir(exist_ok=True)

    save_keyframes_to_disk(frames, frames_names, output_folder)
    export_ds_to_dlc_annotation(ds_stacked, frames_names, labels_folder, video_name)

    # Copy video to the videos folder:
    shutil.copy(video_path, videos_folder / video_path.name)

config = generate_config(
        scorer="movement", 
        project_path=target_folder,
        video_paths=tiled_movies,
        bodyparts=ds_stacked.coords["keypoints"].values.tolist()
    )

save_config(config, target_folder)
