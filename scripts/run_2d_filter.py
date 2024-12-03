# %%

import threed_utils.anipose.anipose_filtering_2d as af2d
import threed_utils.movement_napari as mn
from movement.io.load_poses import from_file
from matplotlib import pyplot as plt
from pathlib import Path
import ocplot as ocp
import numpy as np
%matplotlib widget

plots = False

# %%
data_path = Path("/Users/vigji/Desktop/multicam_video_2024-07-24T10_04_55_cropped_20241104101620")
# data_path = Path(r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\test_cropping\sample_video_for_triangulation\multicam_video_2024-07-24T10_04_55_cropped_20241104101620")
video_path = data_path / "multicam_video_2024-07-24T10_04_55_mirror-bottom.avi.mp4"
slp_inference_path = next(video_path.parent.glob(f"{video_path.stem}*.slp"))  # data_path / "multicam_video_2024-07-24T10_04_55_cropped_20241104101620.slp"
dlc_inference_path = next(video_path.parent.glob(f"{video_path.stem}*.h5"))  # data_path / "multicam_video_2024-07-24T10_04_55_central.aviDLC_resnet50_labels.v001.pkg_converted2024-11-21shuffle1_50000.h5"

ds_sleap_full = from_file(slp_inference_path, source_software="SLEAP")
# ds_dlc = from_file(dlc_inference_path, source_software="DeepLabCut")

#restrict ds_sleap to 23559 to 24559
ds_sleap = ds_sleap_full#.sel(time=slice(23559, 28559))
# %%
# ============================================
# Viterbi filter
# ============================================
conf_threshold = 0.2
config = {
    "filter": {
        "score_threshold": conf_threshold,
        "offset_threshold": 5,
        "n_back": 4,
        "multiprocessing": False
    }
}

if __name__ == "__main__":
    print("filtering...")
    ds_filtered = af2d.filter_pose_viterbi(config, ds_sleap)

# %%
# make a plot showing raw and filtered trajectories between times 23559 and 23663
# time_slice = slice(23559, 23663)

to_compare = dict(original=ds_sleap, filtered=ds_filtered)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

individuals = "individual_0"
# Plot x coordinates
for ax, coord in zip([ax1, ax2], ["x", "y"]):
    for ds_name, ds in to_compare.items():
        ax.plot((ds_filtered.confidence.sel(keypoints="nose", individuals=individuals) > conf_threshold).astype(int)*ds.position.sel(keypoints="nose", individuals=individuals, space=coord).max(),
                alpha=0.1, color="k", lw=0.3)
        ax.plot(ds.position.sel(keypoints="nose", individuals=individuals, space=coord), 
                alpha=0.5, label=ds_name)
    ax.set_ylabel(f"{coord} position")

# plot confidence
ax3.plot(ds_filtered.confidence.sel(keypoints="nose", individuals=individuals), label="filtered")
ax3.set_ylabel("Confidence")
ax3.set_xlabel("Frame")
ax1.legend()
ax2.set_xlabel("Frame")

plt.tight_layout()
plt.show()



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
# Create custom colormap that highlights low confidence in purple
def _custom_colormap(base_cmap_name, confidence, conf_threshold=0.2, low_conf_color=[0.5, 0, 0.5]):
    base_cmap = plt.get_cmap(base_cmap_name)
    colors = np.zeros((len(confidence), 3))
    low_conf = confidence < conf_threshold
    colors[low_conf] = low_conf_color
    colors[~low_conf] = base_cmap(confidence[~low_conf])[:, :3]
    return colors

viewer = napari.Viewer() 
viewer.add_image(VideoReaderNP(str(video_path)))
keys = [i for i in list(to_compare.values())[0].keypoints.values]
tracks_on = False

selected_keys = ["nose"]
for (ds_name, ds), cmap in zip(to_compare.items(), ["Blues", "Greens"]):
    for key in selected_keys:
        ds_filt = ds.sel(keypoints=[key], drop=False)

        track, props = ds_to_napari_tracks(ds_filt)
        props.face_colormap = cmap

        points_style = PointsStyle(name=f"confidence_{key} - {ds_name}", properties=props, size=3)
        points_style.face_color = _custom_colormap(cmap, props["confidence"], conf_threshold=conf_threshold)

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
