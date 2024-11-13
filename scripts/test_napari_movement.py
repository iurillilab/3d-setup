import napari
from movement.napari.layer_styles import TracksStyle, PointsStyle
from movement.napari.utils import columns_to_categorical_codes
from movement.napari.convert import ds_to_napari_tracks
from movement.filtering import filter_by_confidence, interpolate_over_time

from movement import sample_data

ds_name = "SLEAP_single-mouse_EPM.predictions.slp"
ds = sample_data.fetch_dataset(ds_name)
ds = filter_by_confidence(ds, threshold=0.6, print_report=False)
ds = interpolate_over_time(ds, method="linear", print_report=False)
track, props = ds_to_napari_tracks(ds)

viewer = napari.Viewer()
points_style = PointsStyle(name=f"Keypoints - {ds_name}", properties=props)
points_style.set_color_by(prop="keypoint", cmap="turbo")

tracks_props = columns_to_categorical_codes(props, ["individual", "keypoint"])
tracks_props = columns_to_categorical_codes(props, ["individual", "keypoint"])

tracks_style = TracksStyle(name=f"Tracks - {ds_name}", properties=tracks_props)
tracks_style.set_color_by(prop="keypoint", cmap="turbo")

viewer = napari.Viewer()
viewer.add_tracks(track)  # , **tracks_style.as_kwargs())
viewer.add_points(track[:, 1:])  # , **points_style.as_kwargs())
