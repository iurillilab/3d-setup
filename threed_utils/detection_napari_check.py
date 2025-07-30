import h5py
import numpy as np
import napari
from pathlib import Path


def load_detections_from_h5(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load detections and frame indices from h5 file."""
    with h5py.File(h5_path, "r") as h5:
        uvs = h5["uvs"][:]  # (n_detections, 2) - x, y coordinates
        frame_ixs = h5["frame_ixs"][:]  # (n_detections,) - frame indices
    return uvs, frame_ixs


def view_detections(h5_path: Path):
    """
    Open napari viewer with video and overlaid detections as color-coded points.
    Uses napari's video extension for efficient video streaming without loading all frames into memory.
    
    Parameters
    ----------
    video_path : Path
        Path to the video file
    h5_path : Path
        Path to the h5 file containing detections with 'uvs' and 'frame_ixs' datasets
    """
    # Load detections
    print("Loading detections...")
    uvs, frame_ixs = load_detections_from_h5(h5_path)
    
    # Create napari viewer
    viewer = napari.Viewer()
    
    # uvs shape: (time, n_detections, 2) where time corresponds to indices in frame_ixs
    n_detection_frames, n_points, _ = uvs.shape
    max_frame = int(frame_ixs.max())
    
    # Reshape to (n_total_points, 3) format for napari: (time, y, x)
    points_data = []
    detection_ids = []
    
    for i, actual_frame in enumerate(frame_ixs):
        for p in range(n_points):
            x, y = uvs[i, p, 0], uvs[i, p, 1]
            # Skip NaN points if any
            if not (np.isnan(x) or np.isnan(y)):
                points_data.append([actual_frame, y, x])  # actual frame number, y, x
                detection_ids.append(p)  # progressive detection ID
    
    points_data = np.array(points_data)
    detection_ids = np.array(detection_ids)
    
    # Add detections as points layer
    viewer.add_points(
        points_data,
        features={"detection_id": detection_ids},
        face_color="detection_id",
        face_colormap="viridis",
        size=10,
        name="Detections"
    )
    
    print(f"Loaded detections from {len(frame_ixs)} detection frames")
    
    return viewer


def view_movement_3d(movement_ds: "xr.Dataset", viewer: napari.Viewer | None = None):
    """
    Open napari viewer with movement data as color-coded points (2D projection).
    
    Parameters
    ----------
    movement_ds : xarray.Dataset
        Movement dataset with dimensions (time, space, keypoints, individuals)
        where space=['x', 'y', 'z'] and individuals can be squeezed
    """
    import xarray as xr
    
    if viewer is None:
        # Create napari viewer
        viewer = napari.Viewer(ndisplay=3)
    
    # Squeeze individuals dimension and get position data
    # Shape: (time, space, keypoints)
    positions = movement_ds.position.squeeze('individuals')
    n_time, n_space, n_keypoints = positions.shape
    
    # Extract x,y coordinates only (drop z)
    x_coords = positions.sel(space='x').values  # (time, keypoints)
    y_coords = positions.sel(space='y').values  # (time, keypoints)
    z_coords = positions.sel(space='z').values  # (time, keypoints)
    
    # Create time and keypoint index arrays
    time_indices = np.repeat(np.arange(n_time), n_keypoints)  # [0,0,0,...,1,1,1,...] 
    keypoint_indices = np.tile(np.arange(n_keypoints), n_time)  # [0,1,2,...,0,1,2,...]
    
    # Flatten coordinate arrays
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    z_flat = z_coords.flatten()
    
    # Create mask for valid (non-NaN) points
    # valid_mask = ~(np.isnan(x_flat) | np.isnan(y_flat) | np.isnan(z_flat))
    
    # Filter to valid points only - format for napari: (time, y, x)
    points_data = np.column_stack([
        time_indices, #[valid_mask],
        y_flat, #[valid_mask], 
        x_flat, #[valid_mask],
        z_flat
        # z_flat[valid_mask]
    ])
    print(points_data.shape)
    keypoint_ids = keypoint_indices# [valid_mask]
    
    # Add detections as points layer
    viewer.add_points(
        points_data,
        features={"keypoint_id": keypoint_ids},
        face_color="keypoint_id",
        face_colormap="viridis",
        size=5,
        name="2D Keypoints"
    )
    
    print(f"Loaded {len(points_data)} valid 2D keypoints from {n_time} frames with {n_keypoints} keypoints each")
    print("Use the timeline slider to navigate through frames")
    
    return viewer


if __name__ == "__main__":
    # Example usage
    folder = Path("/Users/vigji/Desktop/test_3d/Calibration/20250509/multicam_video_2025-05-09T09_56_51_cropped-v2_20250710121328")
    view = "bottom"
    detection_file = next(folder.glob(f"*{view}*.detections.h5"))
    video_file = next(folder.glob(f"*{view}*.mp4"))
    
    if detection_file.exists():
        viewer = view_detections(detection_file)
        napari.run()
