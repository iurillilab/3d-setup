# %%
from pathlib import Path
import pickle
import numpy as np
from movement.io import load_poses, save_poses

from threed_utils.old_triangulation import triangulate_all_keypoints

path = Path("/Users/vigji/Downloads/tracked_points_sample.pkl")
with open(path, "rb") as f:
    ds = pickle.load(f)
print(ds)

calibration_path = Path("/Users/vigji/code/3d-setup/tests/assets/arena_tracked_points.pkl")
with open(calibration_path, "rb") as f:
    calibration = pickle.load(f)

# %%
triangulated_points = triangulate_all_keypoints(calibration["points"]["tracked_points"], 
                                                calibration["extrinsics"],
                                                calibration["intrinsics"])

ds = load_poses.from_numpy(np.transpose(triangulated_points, (1, 0, 2))[:, None, :, :])


# %%
# Path with multiple recordings:
data_path = Path("/Users/vigji/Desktop/test_movement")
all_files = list(data_path.glob("*.slp"))
print(all_files)
ds = load_poses.from_file(all_files[0], source_software="SLEAP")
# %%
# take firxt 100 timepoints of xarray dataset:
ds = ds.isel(time=slice(100))
ds
# %%
view_labels = "mirror-left", "mirror-right", "mirror-top"
files_dict = {view_label: next(data_path.glob(f"*{view_label}*.slp")) for view_label in view_labels}
files_dict
ds = load_poses.from_multi_view_files(files_dict, source_software="SLEAP")

# %%
from movement.kinematics import compute_displacement, compute_velocity
compute_velocity(ds)
# %%
for file in all_files:
    ds = load_poses.from_file(file, source_software="SLEAP")
    save_poses.to_dlc_file(ds.isel(time=slice(200)), file.parent / "testing_dataset" / (file.stem + "_200.h5"))
# %%
ds
# %%
