# %%
# %%
import napari
from movement.filtering import filter_by_confidence, interpolate_over_time

from movement_napari.layer_styles import TracksStyle, PointsStyle
from movement_napari.utils import columns_to_categorical_codes
from movement_napari.convert import ds_to_napari_tracks
from movement.io.load_poses import from_file
from napari_video.napari_video import VideoReaderNP

from pathlib import Path
dir(napari_video.napari_hook_implementation)
# %%
data_path = Path("/Users/vigji/Desktop/multicam_video_2024-07-24T10_04_55_cropped_20241104101620")
video_path = data_path / "multicam_video_2024-07-24T10_04_55_central.avi.mp4"
slp_inference_path = data_path / "multicam_video_2024-07-24T10_04_55_cropped_20241104101620.slp"
dlc_inference_path = data_path / "multicam_video_2024-07-24T10_04_55_central.aviDLC_resnet50_labels.v001.pkg_converted2024-11-21shuffle1_50000.h5"

ds_sleap = from_file(slp_inference_path, source_software="SLEAP")
ds_dlc = from_file(dlc_inference_path, source_software="DeepLabCut")


# %%
viewer = napari.Viewer()  #ndisplay=3)
viewer.add_image(VideoReaderNP(str(video_path)))
# %%
for ds, ds_name in [(ds_sleap, "SLEAP"), (ds_dlc, "DeepLabCut")]:
    ds = filter_by_confidence(ds, confidence=0.6, print_report=False)
    ds = interpolate_over_time(ds, method="linear", print_report=False)

    track, props = ds_to_napari_tracks(ds)

    points_style = PointsStyle(name=f"Keypoints - {ds_name}", properties=props, size=3)
    points_style.set_color_by(prop="keypoint", cmap="turbo")

    tracks_props = columns_to_categorical_codes(props, ["individual", "keypoint"])
    tracks_style = TracksStyle(name=f"Tracks - {ds_name}", properties=tracks_props, tail_width=2)
    tracks_style.set_color_by(prop="keypoint", cmap="turbo")

    viewer.add_tracks(track, **tracks_style.as_kwargs())
    viewer.add_points(track[:, 1:], size=4)  # **points_style.as_kwargs())
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