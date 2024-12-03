# %%

import threed_utils.anipose.anipose_filtering_2d as af2d
import threed_utils.movement_napari as mn
from movement.io.load_poses import from_file
from matplotlib import pyplot as plt
from pathlib import Path
import ocplot as ocp
import numpy as np

plots = False

# %%
data_path = Path("/Users/vigji/Desktop/multicam_video_2024-07-24T10_04_55_cropped_20241104101620")
# data_path = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\test_cropping\sample_video_for_triangulation\multicam_video_2024-07-24T10_04_55_cropped_20241104101620")
video_path = data_path / "multicam_video_2024-07-24T10_04_55_mirror-bottom.avi.mp4"
slp_inference_path = next(video_path.parent.glob(f"{video_path.stem}*.slp"))  # data_path / "multicam_video_2024-07-24T10_04_55_cropped_20241104101620.slp"
dlc_inference_path = next(video_path.parent.glob(f"{video_path.stem}*.h5"))  # data_path / "multicam_video_2024-07-24T10_04_55_central.aviDLC_resnet50_labels.v001.pkg_converted2024-11-21shuffle1_50000.h5"

ds_sleap = from_file(slp_inference_path, source_software="SLEAP")
ds_dlc = from_file(dlc_inference_path, source_software="DeepLabCut")

# %%
# ============================================
# Viterbi filter
# ============================================

config = {
    "filter": {
        "score_threshold": 0.1,
        "offset_threshold": 15,
        "n_back": 4,
        "multiprocessing": False
    }
}

if __name__ == "__main__":
    print("filtering...")
    ds_filtered = af2d.filter_pose_viterbi(config, ds_dlc)


# %%
# ============================================
# Visualize with napari
# ============================================

import napari
from movement.filtering import filter_by_confidence, interpolate_over_time
from threed_utils.movement_napari.layer_styles import TracksStyle, PointsStyle
from threed_utils.movement_napari.utils import columns_to_categorical_codes
from threed_utils.movement_napari.convert import ds_to_napari_tracks
from napari_video.napari_video import VideoReaderNP

# %%
viewer = napari.Viewer() 
viewer.add_image(VideoReaderNP(str(video_path)))
keys = [i for i in ds_dlc.keypoints.values]
tracks_on = False

to_compare = dict(original=ds_sleap, filtered=ds_filtered)

selected_keys = ["nose"]
for (ds_name, ds), cmap in zip(to_compare.items(), ["Blues", "Reds"]):
    for key in selected_keys:
    # ds = filter_by_confidence(ds, confidence=0.9, print_report=False)
        ds_filt = ds.sel(keypoints=[key], drop=False)
        # ds_filt = interpolate_over_time(ds_filt, method="linear", print_report=False)

        track, props = ds_to_napari_tracks(ds_filt)
        props.face_colormap = cmap

        points_style = PointsStyle(name=f"confidence_{key} - {ds_name}", properties=props, size=3)
        points_style.face_colormap = cmap
        points_style.set_color_by(prop="confidence", cmap=cmap)

        if tracks_on:   
            tracks_props = columns_to_categorical_codes(props, ["individual", "keypoint"])
            tracks_style = TracksStyle(name=f"Tracks_{key} - {ds_filt}", properties=tracks_props, tail_width=2)
            tracks_style.set_color_by(prop="confidence", cmap="turbo")

            viewer.add_tracks(track, **tracks_style.as_kwargs())
            
        viewer.add_points(track[:, 1:], **points_style.as_kwargs())



# # %%
# print(np.sum((viterbi_points[:, :, 0] == 0) & (viterbi_points[:, :, 1] == 0)))
# print(np.min(ds_filtered.position.sel(keypoints="nose", individuals="individual_0").values[:, 0]))
# # %%
# plots = True
# if plots:
#     t_slice = slice(0, 500)
#     plt.figure()
#     for ds, cmap in zip([ds_to_filter, ds_filtered], ["Reds_r", "Blues_r"]):
#         to_plot = ds.position.sel(keypoints="nose", individuals="individual_0", time=t_slice, drop=True)
#         confidence = ds.confidence.sel(keypoints="nose", individuals="individual_0", time=t_slice, drop=True)
#         ocp.color_plot(to_plot.sel(space="x"), -to_plot.sel(space="y"), marker="o",
#         lw=1, c=confidence, cmap=cmap, markersize=2)
#     plt.show()

# %%
