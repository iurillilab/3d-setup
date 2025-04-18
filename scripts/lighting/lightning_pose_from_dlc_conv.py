from movement.io.save_poses import to_dlc_file
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import cv2
import shutil


# Stack keypoints and view dimensions to create combined labels
def stack_keypoints_views(dataset):
    keypoints = dataset.keypoints.values
    views = dataset.view.values

    # Create new combined labels
    new_keypoint_labels = []
    for view in views:
        for keypoint in keypoints:
            combined_label = f"{keypoint}_{view}"
            new_keypoint_labels.append(combined_label)

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
    TEMP_LABELS_FILENAME = "labels.csv"
    TEMP_LABELS_INDIVIDUAL_FILENAME = "labels_individual_0.csv"

    # save as labels csv:
    to_dlc_file(labels_ds, output_folder / TEMP_LABELS_FILENAME, split_individuals=True)

    # Retarded fix for header:
    # Load csv as pandas DataFrame:
    df = pd.read_csv(output_folder / TEMP_LABELS_INDIVIDUAL_FILENAME, header=[0, 1, 2])
    # change df index to be multiindex with first level "labeled-data", second level movie_name.split(".")[0], third name frames_names:
    df.index = pd.MultiIndex.from_tuples(
        [
            ("labeled-data", video_name.split(".")[0], frames_names[i])
            for i in range(len(frames_names))
        ]
    )
    df.to_csv(output_folder / TEMP_LABELS_FILENAME)
    df = pd.read_csv(output_folder / TEMP_LABELS_FILENAME, header=None)
    df.iloc[:3, 0] = ["scorer", "bodyparts", "coords"]
    # drop fourth column by index:
    df = df.drop(columns=[3])
    df.to_csv(output_folder / f"{video_name}_movement.csv", header=False, index=False)

    # Remove "labels.csv" and "labels_individual_0.csv" using pathlib:
    (output_folder / TEMP_LABELS_FILENAME).unlink()
    (output_folder / TEMP_LABELS_INDIVIDUAL_FILENAME).unlink()


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
    total_frames = min(total_frames, 1000)  # Limit to first 1000 frames
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


####################
target_folder = Path(r'D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_project\project')

# create a new folder
target_folder.mkdir(parents=True, exist_ok=True)

labels_folder = target_folder / "labels"
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

# Integrate those:
# Example movement dataset:

# for moview in tiled_movies: 
# dataset, video 

# selecte the frames 

tiled_movies = [r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_project\multicam_video_2024-07-22T10_19_22_cropped_20250325101012.mp4"]
tiled_datasets = [r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_project\predictions_multicam_video_2024-07-22T10_19_22_cropped_20250325101012.h5"]
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

