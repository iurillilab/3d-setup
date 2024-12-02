# %%
from anipose_filtering_2d import *
from movement.io.load_poses import from_file
from matplotlib import pyplot as plt
from pathlib import Path
from movement.io.load_poses import from_file

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