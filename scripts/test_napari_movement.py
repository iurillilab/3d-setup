¬# %%
import napari
from movement.filtering import filter_by_confidence, interpolate_over_time

from movement import sample_data

# %%
# Default dataset:
ds_name = "SLEAP_single-mouse_EPM.predictions.slp"
ds = sample_data.fetch_dataset(ds_name)
ds = filter_by_confidence(ds, confidence=0.6, print_report=False)
ds = interpolate_over_time(ds, method="linear", print_report=False)
ds.position.shape
# %%
# Custom dataset:

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

triangulated_points = triangulate_all_keypoints(calibration["points"]["tracked_points"], 
                                                calibration["extrinsics"],
                                                calibration["intrinsics"])

ds = load_poses.from_numpy(np.transpose(triangulated_points, (1, 0, 2))[:, None, :, :])
ds = filter_by_confidence(ds, confidence=0.6, print_report=False)
ds = interpolate_over_time(ds, method="linear", print_report=False)
ds.position.shape

# %%
from movement_napari.layer_styles import TracksStyle, PointsStyle
from movement_napari.utils import columns_to_categorical_codes
from movement_napari.convert import ds_to_napari_tracks

track, props = ds_to_napari_tracks(ds)

points_style = PointsStyle(name=f"Keypoints - {ds_name}", properties=props, size=3)
points_style.set_color_by(prop="keypoint", cmap="turbo")

tracks_props = columns_to_categorical_codes(props, ["individual", "keypoint"])
tracks_props = columns_to_categorical_codes(props, ["individual", "keypoint"])

tracks_style = TracksStyle(name=f"Tracks - {ds_name}", properties=tracks_props, tail_width=2)
tracks_style.set_color_by(prop="keypoint", cmap="turbo")

viewer = napari.Viewer(ndisplay=3)
viewer.add_tracks(track, **tracks_style.as_kwargs())
viewer.add_points(track[:, 1:], size=4)  #**points_style.as_kwargs())
viewer.add_shapes

track.shape
# %%
from matplotlib import pyplot as plt
for i in range(track.shape[1]):
    plt.figure()
    plt.plot(track[:, i])
    plt.title(f"Track {i}")

# %%
tracks_style.as_kwargs().keys()
# %%
