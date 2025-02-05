# import necessary modules
import matplotlib.pyplot as plt
import numpy as np
import multicam_calibration as mcc
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
import re
import cv2
from threed_utils.io import read_calibration_toml
import xarray as xr
import plotly.graph_objects as go
import json
import argparse


def triangulate_all_keypoints(
    calib_uvs, extrinsics, intrinsics, progress_bar=True
):
    all_triang = []
    calib_uvs = calib_uvs.reshape((5, 1, 8, 2))
    progbar = tqdm if progress_bar else lambda x: x
    for i in progbar(range(calib_uvs.shape[2])):
        all_triang.append(
            mcc.triangulate(calib_uvs[:, :, i, :], extrinsics, intrinsics)
        )

    return np.array(all_triang)

def backproject_triangulated_points(xarray_ds, extrinsics, intrinsics, frame_n, cam_names, arena_3d):
    """
    Back-project 3D triangulated points to 2D camera coordinates for a specific frame.
    
    Args:
        xarray_ds (xarray.Dataset): Dataset containing triangulated points with dimensions:
            (time: 1001, keypoints: 13, individuals: 1, space: 3)
        extrinsics (list): List of extrinsic matrices for each camera
        intrinsics (list): List of intrinsic matrices for each camera
        frame_n (int): Frame number to process
        
    Returns:
        dict: Dictionary of back-projected points for each camera
    """
    video_paths = sorted(slp_dir.glob("*.mp4"))

    camera_matrices = [i[0] for i in intrinsics]
    dist_coef = [i[1] for i in intrinsics]
    # Extract 3D points for the specified frame
    # Shape will be (space, keypoints, individuals)
    frame_points = xarray_ds.position.values[frame_n, ...].squeeze().transpose(1, 0)
    # do semantic ordering of columns

     # Now shape is (keypoints, space)
    
    # Initialize dictionary to store back-projected points for each camera
    
    n_cameras = len(cam_names)
    n_frames = xarray_ds.time.size
    N = xarray_ds.keypoints.size

    reprojections = np.zeros((n_cameras, n_frames, N, 2))
    for cam in tqdm(range(len(checkboard_ds.view.values.tolist()))):
        reprojections[cam] = mcc.project_points(
            frame_points, extrinsics[cam], intrinsics[cam][0]
    )

    arena2d = {}
    for cam in tqdm(range(len(checkboard_ds.view.values.tolist()))):
        arena2d[cam] = mcc.project_points(
            arena_3d, extrinsics[cam], intrinsics[cam][0]
    )
    return arena2d, reprojections


def read_nth_frame(input_file, n):
    """
    Reads the nth frame from the input video file.

    Args:
        input_file (str or Path): Path to the video file.
        n (int): Frame number to read (0-indexed).

    Returns:
        np.ndarray: The nth frame as an image array.

    Raises:
        ValueError: If the frame cannot be read or the frame number is out of bounds.
    """
    # Open the video file
    cap = cv2.VideoCapture(str(input_file))
    
    # Check if the video was successfully opened
    if not cap.isOpened():
        raise ValueError(f"Failed to open the video file: {input_file}")
    
    # Set the frame position
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 0 or n >= total_frames:
        cap.release()
        raise ValueError(f"Frame number {n} is out of bounds for video with {total_frames} frames.")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Failed to read frame {n} from the video.")
    
    # Release the video capture object
    cap.release()
    
    return frame


def get_frames_camera(cam_names, n_frame, video_dir):
    """
    Associates video frames with their corresponding camera keys from backprojections.

    Args:
        cam_names (list): List of camera names in the desired order
        n_frame (int): Frame number to extract.
        video_dir (Path): Directory containing the video files.

    Returns:
        dict: Dictionary with keys as camera names (in original order) and values as video frames
    """
    video_paths = sorted(video_dir.glob("*.mp4"))

    # Regex to extract camera name from video file names
    camera_name_regex = re.compile(r".*_(\w+(?:-\w+)?)\.avi\.mp4$")

    # Create a map of camera name to video path
    video_camera_map = {}
    for video_path in video_paths:
        match = camera_name_regex.match(video_path.name)
        if match:
            camera_name = match.group(1)
            video_camera_map[camera_name] = video_path
        else:
            raise ValueError(f"Could not extract camera name from: {video_path.name}")

    # Create dictionary maintaining cam_names order
    camera_frames = {}
    for name in cam_names:
        if name not in video_camera_map:
            raise ValueError(f"Camera {name} not found in video directory")
        camera_frames[name] = read_nth_frame(video_camera_map[name], n_frame)

    return camera_frames




def plot_frames_with_points(cam_names, back_projected, video_camera_map, arena_2d, arena_points, frame_n):
    """
    Plots frames with tracked points in subplots, using camera names as titles.

    Args:
        camera_data (dict): A dictionary where keys are camera names and values are tuples:
                            (frame (numpy array), tracked points (list or numpy array)).
    """
    # Determine the number of subplots needed
    n_cameras = len(cam_names)
    n_cols = 3  # Number of columns in the grid
    n_rows = -(-n_cameras // n_cols)  # Ceiling division to get rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    arena_points = arena_points.squeeze()

    # Flatten axes for easy iteration
    axes = axes.flatten() if n_rows > 1 else [axes]
    arena_points_p = [arena_view.squeeze() for arena_view in arena_2d.values()]
    # Loop over cameras and plot data
    for idx in range(len(cam_names)):
        ax = axes[idx]
        ax.imshow(video_camera_map[cam_names[idx]])  # Display the frame
        points = np.array(back_projected[idx]) 
        print(points.shape) # Ensure points are in array form
        if len(points) > 0:
            ax.scatter(points[frame_n, : , 0], points[frame_n, :, 1], color='red', s=6)
            ax.scatter(arena_points_p[idx][:, 0], arena_points_p[idx][:, 1], color='blue', s=6)
            # ax.scatter(arena_2d[idx, :, 1], arena_2d[idx, :, 0], color='blue', s=10)  # Plot points
        ax.set_title(cam_names[idx])
        ax.axis('off')  # Turn off axis for a cleaner look

    # Hide unused subplots
    for i in range(len(cam_names), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def plot3d_frame(frame_points3d, arena_3d):
    frame_points3d = frame_points3d.squeeze()
    arena_3d = arena_3d.squeeze()
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=frame_points3d[:, 0], y=frame_points3d[:, 1], z=frame_points3d[:, 2], mode='markers', marker=dict(size=5, color='red')))
    #add arena points 
    fig.add_trace(go.Scatter3d(x=arena_3d[:, 0], y=arena_3d[:, 1], z=arena_3d[:, 2], mode='markers', marker=dict(size=5, color='blue')))
    fig.show()
# function that combines the previous functions into a single one by calling them:

def backprojections_plots(ds, n_frame, video_dir, calibration_dir, arena_path):


    # load the calibration matrices:
    calibration_path = Path(calibration_dir)
    calibration_paths = sorted(data_dir.glob("mc_calibration_output_*"))
    last_calibration_path = calibration_paths[-1]

    all_calib_uvs = np.load(last_calibration_path / "all_calib_uvs.npy")
    calib_toml_path = last_calibration_path / "calibration_from_mc.toml"
    cam_names, img_sizes, extrinsics, intrinsics = read_calibration_toml(calib_toml_path)
    with open(arena_path, 'r') as f:
        cropping_dict = json.load(f)
        old_arena = cropping_dict[-1]['points_coordinate']
    arr_arena = []
    for key, value in old_arena.items():
        arr_arena.append(value)
# load arena coordinates inot a movement dataset: with conf, etc. 
    arr_arena = np.array(arr_arena)
    arena_points_new = np.zeros_like(arr_arena)


    for i, cam in enumerate(cam_names):
        arena_points_new[i, ...] = old_arena[cam]


    arena_3d = triangulate_all_keypoints(arena_points_new[..., [1, 0]], extrinsics, intrinsics)

    arena_2d, back_projected = backproject_triangulated_points(ds, extrinsics, intrinsics, n_frame, cam_names, arena_3d)
    frame_points3d = ds.isel(time=n_frame).position.values.squeeze().transpose(1, 0)
    plot3d_frame(frame_points3d, arena_3d)



    video_camera_map = get_frames_camera(cam_names, n_frame, video_dir)


    plot_frames_with_points(cam_names, back_projected, video_camera_map, arena_2d, arena_points_new, n_frame)



def parse_args():
    parser = argparse.ArgumentParser(description='Backproject and plot 3D tracking data')
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing calibration data'
    )
    
    parser.add_argument(
        '--slp_dir',
        type=str,
        required=True,
        help='Directory containing video files'
    )
    
    parser.add_argument(
        '--frame_n',
        type=int,
        required=True,
        help='Frame number to process'
    )
    
    parser.add_argument(
        '--cropping_path',
        type=str,
        required=True,
        help='Path to the cropping parameters JSON file'
    )
    
    parser.add_argument(
        '--ds_path',
        type=str,
        required=True,
        help='Path to the xarray dataset containing tracking data'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Convert string paths to Path objects
    data_dir = Path(args.data_dir)
    slp_dir = Path(args.slp_dir)
    checkboard = data_dir / 'checkboard_triangulaiton.h5'
    checkboard_ds = xr.open_dataset(checkboard)

    cropping_path = Path(args.cropping_path)
        
    # Load the dataset
    ds = xr.open_dataset(args.ds_path)
    
    # Call your main function with the arguments
    backprojections_plots(
        ds=ds,
        n_frame=args.frame_n,
        video_dir=slp_dir,
        calibration_dir=data_dir,
        arena_path=args.cropping_path
    )


'''
Example of command:
python backprojection.py \
    --data_dir /Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/calibration \
    --slp_dir /Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/video_test \
    --frame_n 100 \
    --cropping_path /Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/uncropped_cal/multicam_video_2024-07-22T10_19_22_20241209-164946.json \
    --ds_path /Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/video_test/mcc_triangulated_ds.h5
'''