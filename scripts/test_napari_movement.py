# %%
# %%
import napari
from movement.filtering import filter_by_confidence, interpolate_over_time

from movement_napari.layer_styles import TracksStyle, PointsStyle
from movement_napari.utils import columns_to_categorical_codes
from movement_napari.convert import ds_to_napari_tracks
from movement.io.load_poses import from_file
from napari_video.napari_video import VideoReaderNP
from matplotlib import pyplot as plt
from pathlib import Path
# %%
# data_path = Path("/Users/vigji/Desktop/multicam_video_2024-07-24T10_04_55_cropped_20241104101620")
data_path = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\test_cropping\sample_video_for_triangulation\multicam_video_2024-07-24T10_04_55_cropped_20241104101620")
video_path = data_path / "multicam_video_2024-07-24T10_04_55_mirror-bottom.avi.mp4"
slp_inference_path = next(video_path.parent.glob(f"{video_path.stem}*.slp"))  # data_path / "multicam_video_2024-07-24T10_04_55_cropped_20241104101620.slp"
dlc_inference_path = next(video_path.parent.glob(f"{video_path.stem}*.h5"))  # data_path / "multicam_video_2024-07-24T10_04_55_central.aviDLC_resnet50_labels.v001.pkg_converted2024-11-21shuffle1_50000.h5"

ds_sleap = from_file(slp_inference_path, source_software="SLEAP")
ds_dlc = from_file(dlc_inference_path, source_software="DeepLabCut")

# %%
nose_coords = ds_dlc.position.sel(individuals="individual_0", keypoints="nose", drop=True)
nose_conf = ds_dlc.confidence.sel(individuals="individual_0", keypoints="nose", drop=True)
ds_dlc.position.to_numpy()
# %%
plt.figure()
plt.scatter(nose_coords.sel(space="x"), nose_coords.sel(space="y"), s=1, c=nose_conf, cmap="turbo")
nose_coords.sel(space="x").shape


#%%
keys = [i for i in ds_dlc.keypoints.values]
ds_filts =[]
for key in keys:
    ds_filts.append(ds_dlc.sel(keypoints=[key], drop=False))


# %%
viewer = napari.Viewer()  #ndisplay=3)
viewer.add_image(VideoReaderNP(str(video_path)))
keys = [i for i in ds_dlc.keypoints.values]

for ds, ds_name, cmap in [(ds_sleap, "SLEAP", "Blues"), (ds_dlc, "DeepLabCut", "Reds")]:
    for key in keys:
    # ds = filter_by_confidence(ds, confidence=0.9, print_report=False)
        ds_filt = ds.sel(keypoints=[key], drop=False)
        ds_filt = interpolate_over_time(ds_filt, method="linear", print_report=False)

        track, props = ds_to_napari_tracks(ds_filt)
        props.face_colormap = cmap

        points_style = PointsStyle(name=f"confidence_{key} - {ds_name}", properties=props, size=3)
        points_style.face_colormap = cmap
        points_style.set_color_by(prop="confidence", cmap=cmap)

        # tracks_props = columns_to_categorical_codes(props, ["individual", "keypoint"])
        # tracks_style = TracksStyle(name=f"Tracks_{key} - {ds_filt}", properties=tracks_props, tail_width=2)
        # tracks_style.set_color_by(prop="confidence", cmap="turbo")

        # viewer.add_tracks(track, **tracks_style.as_kwargs())
        viewer.add_points(track[:, 1:], **points_style.as_kwargs())
        viewer.add_shapes

track.shape
# %%
#potential differences at frame 17306

#%%

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