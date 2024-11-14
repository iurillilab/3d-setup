# %%
from pathlib import Path
import pickle
import numpy as np
from movement.io import load_poses

from threed_utils.triangulation import triangulate_all_keypoints

path = Path("/Users/vigji/Downloads/tracked_points_sample.pkl")
with open(path, "rb") as f:
    ds = pickle.load(f)
print(ds)

calibration_path = Path("/Users/vigji/code/3d-setup/tests/assets/arena_tracked_points.pkl")
with open(calibration_path, "rb") as f:
    calibration = pickle.load(f)

calibration["extrinsics"]

# %%
triangulated_points = triangulate_all_keypoints(calibration["points"]["tracked_points"], 
                                                calibration["extrinsics"],
                                                calibration["intrinsics"])


reordered_points = np.transpose(triangulated_points, (1, 0, 2))[:, None, :, :]
ds = load_poses.from_numpy(reordered_points)


# %%
