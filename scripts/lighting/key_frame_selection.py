import cv2
import numpy as np
import xarray as xr
from pathlib import Path
import yaml
from pathlib import Path




def extract_keyframes(video_path, dataset, stride=50, individual=0):
    """
    Extract every `stride`-th frame from video and matching prediction from dataset.

    Returns:
        keyframe_imgs (list of np.ndarray): Extracted frames as images
        keyframe_ds (xarray.Dataset): Subset of original dataset with only keyframes
        keyframe_indices (list of int): Indices of keyframes
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keyframe_indices = list(range(0, total_frames, stride))
    
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


def save_keyframes_to_disk(frames, output_dir, prefix="frame"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        path = Path(output_dir) / f"{prefix}_{i:04d}.png"
        cv2.imwrite(str(path), frame)



def generate_minimal_dlc_config(bodyparts, video_path, scorer="TiledPose"):
    """
    Generate a minimal DLC config.yaml content.
    
    Args:
        bodyparts (list): List of keypoint names.
        video_path (str or Path): Path to the video file.
        scorer (str): Name of the scorer (used internally in DLC).
    
    Returns:
        dict: config.yaml content.
    """
    config = {
        "scorer": scorer,
        "bodyparts": bodyparts,
        "video_sets": {
            str(Path(video_path).resolve()): {
                "crop": None
            }
        }
    }
    return config


if __name__ == "__main__":
    video_path = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_project\tiled_output.mp4"
    dataset_path = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_project\predictions_multicam_video_2024-07-22T10_19_22_cropped_20250325101012.h5"
    stride = 50  # Every 50 frames
    output_dir = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_project\selected_frames"

    shifted_ds = xr.open_dataset(dataset_path)

    bodyparts = list(shifted_ds["keypoints"].values)  



    frames, ds_keyframes, frame_indices = extract_keyframes(video_path, shifted_ds, stride=stride)
    save_keyframes_to_disk(frames, output_dir)

    # Optionally save the new dataset
    save_path_predictions = Path(output_dir) / "keyframe_predictions.h5"
    ds_keyframes.to_netcdf(save_path_predictions)

    video_path = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\lighting_project\tiled_output.mp4"

    config = generate_minimal_dlc_config(bodyparts, video_path, scorer="TiledPose")
    config_path = Path(output_dir) / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

