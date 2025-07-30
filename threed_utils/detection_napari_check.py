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


if __name__ == "__main__":
    # Example usage
    folder = Path("/Users/vigji/Desktop/test_3d/Calibration/20250509/multicam_video_2025-05-09T09_56_51_cropped-v2_20250710121328")
    view = "bottom"
    detection_file = next(folder.glob(f"*{view}*.detections.h5"))
    video_file = next(folder.glob(f"*{view}*.mp4"))
    
    if detection_file.exists():
        viewer = view_detections(detection_file)
        napari.run()
