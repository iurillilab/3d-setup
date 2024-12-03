# %%

import anipose_filtering_2d as af2d
from movement.io.load_poses import from_file
from matplotlib import pyplot as plt
from pathlib import Path
#import ocplot as ocp
import numpy as np

plots = False

# %%
#data_path = Path("/Users/vigji/Desktop/multicam_video_2024-07-24T10_04_55_cropped_20241104101620")
data_path = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\test_cropping\sample_video_for_triangulation\multicam_video_2024-07-24T10_04_55_cropped_20241104101620")
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
if plots:
    t_slice = slice(500, 1000)
    plt.figure()
    for ds, cmap in zip([ds_sleap, ds_dlc], ["Reds_r", "Blues_r"]):
        to_plot = ds.position.sel(keypoints="nose", individuals="individual_0", time=t_slice, drop=True)
        confidence = ds.confidence.sel(keypoints="nose", individuals="individual_0", time=t_slice, drop=True)
        ocp.color_plot(to_plot.sel(space="x"), -to_plot.sel(space="y"), marker="o",
        lw=1, c=confidence, cmap=cmap, markersize=2)
    plt.show()

# %%
def movement_to_anipose(mov_ds):
    position_vals = mov_ds.position.transpose("time", "keypoints", "individuals", "space").values
    confidence_vals = mov_ds.confidence.transpose("time", "keypoints", "individuals").values
    test = np.concatenate([position_vals, confidence_vals[:, :, :, None]], axis=3)
    metadata = {
        'bodyparts': ds_dlc.coords["keypoints"].values,
        'scorer': ds_dlc.attrs["source_software"],
        'index': mov_ds.coords["time"].values
    }
    return test, metadata

def anipose_to_movement(data, source_ds):
    """Convert anipose format back to movement format"""
    ds = source_ds.copy()
    # data that are 0 alongh both last dim values are set to nan
    print(np.sum(np.all(data == 0, axis=-1)))
    data[np.all(data == 0, axis=-1)] = np.nan

    ds.position.values = data[:, np.newaxis, :, :]
    
    return ds

# ds_to_filter = ds_dlc
# points, metadata = load_pose_2d_movement(ds_to_filter)


# %%
# ============================================
# Viterbi filter
config = {
    "filter": {
        "score_threshold": 0.5,
        "offset_threshold": 10,
        "n_back": 10,
        "multiprocessing": False
    }
}

if __name__ == "__main__":
    print("filtering...")
    ds_filtered = af2d.filter_pose_viterbi(config, ds_dlc)
#  %%

# add triangulation function from movement ds
from threed_utils.triangulation import triangulate_all_keypoints
import pickle 

with open(r"C:\Users\SNeurobiology\code\3d-setup\tests\assets\arena_tracked_points.pkl", "rb") as f:
    points = pickle.load(f)
    extrinsics = points["extrinsics"]
    intrinsics = points["intrinsics"]

print(extrinsics, intrinsics)






# %%
ds_filtered = anipose_to_movement(viterbi_points, ds_to_filter)
print(np.sum((viterbi_points[:, :, 0] == 0) & (viterbi_points[:, :, 1] == 0)))
print(np.min(ds_filtered.position.sel(keypoints="nose", individuals="individual_0").values[:, 0]))
# %%
plots = True
if plots:
    t_slice = slice(0, 500)
    plt.figure()
    for ds, cmap in zip([ds_to_filter, ds_filtered], ["Reds_r", "Blues_r"]):
        to_plot = ds.position.sel(keypoints="nose", individuals="individual_0", time=t_slice, drop=True)
        confidence = ds.confidence.sel(keypoints="nose", individuals="individual_0", time=t_slice, drop=True)
        ocp.color_plot(to_plot.sel(space="x"), -to_plot.sel(space="y"), marker="o",
        lw=1, c=confidence, cmap=cmap, markersize=2)
    plt.show()

# %%
# ============================================
# Autoencoder filter
# ============================================
config = {
    "filter": {
        "autoencoder_points_path": "/Users/vigji/Desktop/multicam_video_2024-07-24T10_04_55_cropped_20241104101620/autoencoder_points_model.pkl"
    }
}
autoencoder_points = af2d.filter_pose_autoencoder_points(config, points, metadata["bodyparts"])

# %%
