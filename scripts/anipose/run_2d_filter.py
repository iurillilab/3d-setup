# %%
# %matplotlib widget
from anipose_filtering_2d import *
from movement.io.load_poses import from_file
from matplotlib import pyplot as plt
from pathlib import Path
import ocplot as ocp

# %%
data_path = Path("/Users/vigji/Desktop/multicam_video_2024-07-24T10_04_55_cropped_20241104101620")
# data_path = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\test_cropping\sample_video_for_triangulation\multicam_video_2024-07-24T10_04_55_cropped_20241104101620")
video_path = data_path / "multicam_video_2024-07-24T10_04_55_mirror-bottom.avi.mp4"
slp_inference_path = next(video_path.parent.glob(f"{video_path.stem}*.slp"))  # data_path / "multicam_video_2024-07-24T10_04_55_cropped_20241104101620.slp"
dlc_inference_path = next(video_path.parent.glob(f"{video_path.stem}*.h5"))  # data_path / "multicam_video_2024-07-24T10_04_55_central.aviDLC_resnet50_labels.v001.pkg_converted2024-11-21shuffle1_50000.h5"

ds_sleap = from_file(slp_inference_path, source_software="SLEAP")
ds_dlc = from_file(dlc_inference_path, source_software="DeepLabCut")

# %% napari view
# # %%
# import napari
# from movement.filtering import filter_by_confidence, interpolate_over_time
# from movement_napari.layer_styles import TracksStyle, PointsStyle
# from movement_napari.utils import columns_to_categorical_codes
# from movement_napari.convert import ds_to_napari_tracks
# from napari_video.napari_video import VideoReaderNP

# %%
t_slice = slice(500, 1000)
plt.figure()
for ds, cmap in zip([ds_sleap, ds_dlc], ["Reds_r", "Blues_r"]):
    to_plot = ds.position.sel(keypoints="nose", individuals="individual_0", time=t_slice, drop=True)
    confidence = ds.confidence.sel(keypoints="nose", individuals="individual_0", time=t_slice, drop=True)
    ocp.color_plot(to_plot.sel(space="x"), -to_plot.sel(space="y"), marker="o",
    lw=1, c=confidence, cmap=cmap, markersize=2)
    plt.show()

# %%
to_plot.coords["time"].values
# %%
def load_pose_2d(fname):
    data_orig = pd.read_hdf(fname)
    scorer = data_orig.columns.levels[0][0]
    data = data_orig.loc[:, scorer]

    bp_index = data.columns.names.index('bodyparts')
    coord_index = data.columns.names.index('coords')
    bodyparts = list(data.columns.get_level_values(bp_index).unique())
    n_possible = len(data.columns.levels[coord_index])//3

    n_frames = len(data)
    n_joints = len(bodyparts)
    test = np.array(data).reshape(n_frames, n_joints, n_possible, 3)

    metadata = {
        'bodyparts': bodyparts,
        'scorer': ds_dlc.attrs["source_software"],
        'index': data.index
    }

    return test, metadata

def load_pose_2d_movement(mov_ds):
    position_vals = mov_ds.position.transpose("time", "keypoints", "individuals", "space").values
    confidence_vals = mov_ds.confidence.transpose("time", "keypoints", "individuals").values
    test = np.concatenate([position_vals, confidence_vals[:, :, :, None]], axis=3)
    metadata = {
        'bodyparts': ds_dlc.coords["keypoints"].values,
        'scorer': ds_dlc.attrs["source_software"],
        'index': mov_ds.coords["time"].values
    }
    return test, metadata

test, metadata = load_pose_2d(dlc_inference_path)

test2, metadata2 = load_pose_2d_movement(ds_dlc)
print(test.shape, test2.shape)
assert np.allclose(test, test2)
# %%
f, axs = plt.subplots(2, 1)
axs[0].imshow(test[:, :, 0, 0], aspect="auto")
axs[1].imshow(test2[:, :, 0, 0], aspect="auto")
plt.show()


# %%
position_vals = ds_dlc.position.transpose("time", "individuals", "keypoints", "space").values
confidence_vals = ds_dlc.confidence.transpose("time", "individuals", "keypoints").values
np.concatenate([position_vals, confidence_vals[:, :, :, None]], axis=3).shape

# %%
ds_dlc.coords["time"].values
# %%
config = {
    "filter": {
        "score_threshold": 0.5,
        "offset_threshold": 10,
        "n_back": 10,
        "multiprocessing": False
    }
}

viterbi_points, viterbi_scores = filter_pose_viterbi(config, test, metadata["bodyparts"])

# %%
