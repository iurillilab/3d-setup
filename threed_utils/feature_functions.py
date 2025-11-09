from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr


SessionData = Dict[str, object]


def compute_velocity(
    session_data: SessionData, time_slice: Optional[int] = None
) -> np.ndarray:
    session = _get_session_dataset(session_data)
    coordinates = _select_position_values(session, time_slice)
    assert coordinates.shape[2] > 0
    centroids = coordinates.mean(axis=2).squeeze()
    velocity = np.vstack([np.zeros((1, 3)), np.diff(centroids, axis=0)])
    return np.linalg.norm(velocity, axis=1)


def compute_acceleration(
    session_data: SessionData, time_slice: Optional[int] = None
) -> np.ndarray:
    session = _get_session_dataset(session_data)
    coordinates = _select_position_values(session, time_slice)
    assert coordinates.shape[2] > 0
    centroids = coordinates.mean(axis=2).squeeze()
    displacement = np.vstack([np.zeros((1, 3)), np.diff(centroids, axis=0)])
    acceleration_vec = np.vstack([np.zeros((1, 3)), np.diff(displacement, axis=0)])
    return np.linalg.norm(acceleration_vec, axis=1)


def compute_head_rear(
    session_data: SessionData, time_slice: Optional[int] = None
) -> np.ndarray:
    session = _get_session_dataset(session_data)
    assert "position" in session
    assert "z" in session.position.space
    selection = session["position"].sel(keypoints=["nose", "ear_lf", "ear_rt"], space="z")
    if time_slice is not None:
        assert time_slice > 0
        selection = selection.isel(time=slice(0, time_slice))
    return selection.values.squeeze()


def compute_theta(
    session_data: SessionData,
    keypoints: Tuple[str, str],
    time_slice: Optional[int] = None,
) -> np.ndarray:
    session = _get_session_dataset(session_data)
    assert len(keypoints) == 2
    keypoint_1, keypoint_2 = keypoints
    assert keypoint_1 in session.position.keypoints
    assert keypoint_2 in session.position.keypoints
    diff = (
        session["position"].sel(keypoints=keypoint_1).values
        - session["position"].sel(keypoints=keypoint_2).values
    )
    if time_slice is not None:
        assert time_slice > 0
        diff = diff[:time_slice]
    assert diff.shape[1] >= 2
    diff_norm = np.linalg.norm(diff, axis=1)
    v_x = diff[:, 0] / diff_norm
    v_y = diff[:, 1] / diff_norm
    return np.arctan2(v_y, v_x)


def compute_turning_rate(
    session_data: SessionData,
    keypoints: Tuple[str, str],
    time_slice: Optional[int] = None,
) -> np.ndarray:
    theta = compute_theta(session_data, keypoints, time_slice)
    dtheta = np.vstack([np.zeros((1,)), np.diff(theta, axis=0)])
    return np.mod(dtheta + np.pi, 2 * np.pi) - np.pi


def compute_yaw_offset(
    session_data: SessionData, time_slice: Optional[int] = None
) -> np.ndarray:
    theta_body = compute_theta(session_data, ("back_mid", "tailbase"), time_slice)
    theta_head = compute_theta(session_data, ("nose", "tailbase"), time_slice)
    delta_theta = theta_head - theta_body
    return np.mod(delta_theta + np.pi, 2 * np.pi) - np.pi


def compute_pitch_angle(
    session_data: SessionData,
    keypoints: Tuple[str, str],
    time_slice: Optional[int] = None,
) -> np.ndarray:
    session = _get_session_dataset(session_data)
    assert len(keypoints) == 2
    keypoint_1, keypoint_2 = keypoints
    assert keypoint_1 in session.position.keypoints
    assert keypoint_2 in session.position.keypoints
    assert "z" in session.position.space
    v_nose = session.position.sel(keypoints=keypoint_1).values
    v_tailbase = session.position.sel(keypoints=keypoint_2).values
    if time_slice is not None:
        assert time_slice > 0
        v_nose = v_nose[:time_slice]
        v_tailbase = v_tailbase[:time_slice]
    v_body = v_nose - v_tailbase
    assert v_body.shape[1] >= 3
    length_xy = np.linalg.norm(v_body[:, :2], axis=1)
    return np.arctan2(v_body[:, 2], length_xy)


def compute_velocity_components(
    session_data: SessionData, time_slice: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    session = _get_session_dataset(session_data)
    assert "nose" in session.position.keypoints
    assert "tailbase" in session.position.keypoints

    if time_slice is not None:
        assert time_slice > 0
        position = session.position.isel(time=slice(0, time_slice))
        centroids = position.values.mean(axis=2).squeeze()
        nose = session.position.sel(keypoints="nose").isel(time=slice(0, time_slice)).values
        tail = (
            session.position.sel(keypoints="tailbase")
            .isel(time=slice(0, time_slice))
            .values
        )
    else:
        position = session.position
        centroids = position.values.mean(axis=2).squeeze()
        nose = session.position.sel(keypoints="nose").values
        tail = session.position.sel(keypoints="tailbase").values

    displacement = np.vstack([np.zeros((1, 3)), np.diff(centroids, axis=0)])
    nose = nose.squeeze(-1)
    tail = tail.squeeze(-1)

    body_vector = nose - tail
    body_vector_norm = np.linalg.norm(body_vector, axis=1)
    body_vector_unit = body_vector / body_vector_norm[:, None]

    z_axis = np.array([0.0, 0.0, 1.0])
    side_axis = np.cross(z_axis, body_vector_unit)
    side_axis_norm = np.linalg.norm(side_axis, axis=1)
    side_axis_unit = side_axis / side_axis_norm[:, None]

    forward_velocity = np.sum(displacement * body_vector_unit, axis=1)
    side_velocity = np.sum(displacement * side_axis_unit, axis=1)
    vertical_velocity = displacement[:, 2]
    return forward_velocity, side_velocity, vertical_velocity


def compute_manipulation_index_paws(
    session_data: SessionData,
    time_slice: Optional[int] = None,
    target_distance: Optional[np.ndarray] = None,
    quantile_manipulation: float = 0.75,
    quantile_distance: float = 0.1,
) -> np.ndarray:
    session = _get_session_dataset(session_data)
    assert "forepaw_lf" in session.position.keypoints
    assert "forepaw_rt" in session.position.keypoints

    v_lf = session.position.sel(keypoints="forepaw_lf").values.squeeze()
    v_rt = session.position.sel(keypoints="forepaw_rt").values.squeeze()
    if time_slice is not None:
        assert time_slice > 0
        v_lf = v_lf[:time_slice]
        v_rt = v_rt[:time_slice]

    body_speed = compute_velocity(session_data, time_slice)
    if target_distance is None:
        target_distance = compute_distance_mouse_cricket(session_data, time_slice)
    target_distance_slice = target_distance[:time_slice]

    lf_disp = np.vstack([np.zeros((1, 3)), np.diff(v_lf, axis=0)])
    rt_disp = np.vstack([np.zeros((1, 3)), np.diff(v_rt, axis=0)])
    lf_speed = np.linalg.norm(lf_disp, axis=1)
    rt_speed = np.linalg.norm(rt_disp, axis=1)

    body_speed_min = 1e-3
    body_speed_safe = np.where(body_speed > 0, body_speed, body_speed_min)
    manipulation_idx = (lf_speed / body_speed_safe + rt_speed / body_speed_safe) / 2
    manipulation_threshold = np.quantile(manipulation_idx, quantile_manipulation)
    distance_threshold = np.quantile(target_distance_slice, quantile_distance)
    manipulation_idx_filtered = np.where(
        manipulation_idx > manipulation_threshold, manipulation_idx, 0
    )
    mask_manipulation_close = (target_distance_slice < distance_threshold) & (
        manipulation_idx_filtered > manipulation_threshold
    )
    mask_manipulation_far = (target_distance_slice >= distance_threshold) & (
        manipulation_idx_filtered > manipulation_threshold
    )
    manipulation_idx_filtered = np.where(
        mask_manipulation_close, 1, manipulation_idx_filtered
    )
    manipulation_idx_filtered = np.where(
        mask_manipulation_far, -1, manipulation_idx_filtered
    )
    return manipulation_idx_filtered


def compute_freezing(
    session_data: SessionData, time_slice: Optional[int] = None
) -> np.ndarray:
    velocity = compute_velocity(session_data, time_slice)
    threshold = np.percentile(velocity, 10)
    return velocity < threshold


def compute_curvature(
    session_data: SessionData, time_slice: Optional[int] = None
) -> np.ndarray:
    turning_rate = compute_turning_rate(session_data, ("nose", "tailbase"), time_slice)
    velocity = compute_velocity(session_data, time_slice)
    assert len(turning_rate) == len(velocity)
    turning_rate = np.abs(turning_rate.squeeze())
    velocity_safe = np.where(velocity > 0, velocity, 1.0)
    return turning_rate / velocity_safe + np.random.randn(len(velocity)) * 0.01


def compute_position_centroid(
    session_data: SessionData, time_slice: Optional[int] = None
) -> np.ndarray:
    session = _get_session_dataset(session_data)
    position = session.position
    if time_slice is not None:
        assert time_slice > 0
        position = position.isel(time=slice(0, time_slice))
    centroid = position.values.mean(axis=2).squeeze()
    return centroid


def compute_distance_to_walls(
    session_data: SessionData, time_slice: Optional[int] = None
) -> np.ndarray:
    arena_3d = _get_arena_dataset(session_data)
    assert "position" in arena_3d
    assert "x" in arena_3d.position.space and "y" in arena_3d.position.space

    centroid_position = compute_position_centroid(session_data, time_slice)
    arena_2d = arena_3d.position.sel(space=["x", "y"]).values.squeeze()
    x_min, x_max = arena_2d[:, 0].min(), arena_2d[:, 0].max()
    y_min, y_max = arena_2d[:, 1].min(), arena_2d[:, 1].max()

    centroid_2d = centroid_position[:, :2]
    dist_to_x_min = np.abs(x_min - centroid_2d[:, 0])
    dist_to_x_max = np.abs(x_max - centroid_2d[:, 0])
    dist_to_y_min = np.abs(y_min - centroid_2d[:, 1])
    dist_to_y_max = np.abs(y_max - centroid_2d[:, 1])
    return np.minimum.reduce(
        [dist_to_x_min, dist_to_x_max, dist_to_y_min, dist_to_y_max]
    )


def compute_distance_mouse_cricket(
    session_data: SessionData, time_slice: Optional[int] = None
) -> np.ndarray:
    cricket_coords_2d = session_data.get("cricket_coords_2d")
    mouse_coords_2d = session_data.get("mouse_coords_2d")
    if cricket_coords_2d is not None and mouse_coords_2d is not None:
        return compute_distance_mouse_cricket_2d(session_data, time_slice)

    cricket_coords_3d = session_data.get("cricket_coords_3d")
    if cricket_coords_3d is None:
        raise ValueError(
            "Cricket coordinates not available for this session (object session?)"
        )
    mouse_centroid = compute_position_centroid(session_data, time_slice)
    min_len = min(len(mouse_centroid), len(cricket_coords_3d))
    mouse_centroid = mouse_centroid[:min_len]
    cricket_coords = cricket_coords_3d[:min_len]

    assert mouse_centroid.shape[1] >= 2
    assert cricket_coords.shape[1] == 2
    if mouse_centroid.ndim == 3:
        mouse_centroid = np.nanmean(mouse_centroid[:, :, :2], axis=1)
    else:
        mouse_centroid = mouse_centroid[:, :2]
    return np.linalg.norm(mouse_centroid - cricket_coords, axis=1)


def compute_mouse_centroid_2d(
    session_data: SessionData, time_slice: Optional[int] = None
) -> np.ndarray:
    mouse_coords_2d = session_data.get("mouse_coords_2d")
    if mouse_coords_2d is None:
        raise ValueError("Mouse 2D coordinates not available for this session")
    coords = mouse_coords_2d
    bodyparts = session_data.get("mouse_coords_2d_bodyparts") or []
    if time_slice is not None:
        assert time_slice > 0
        coords = coords[:time_slice]
    if coords.ndim == 2:
        return coords

    torso_keypoints = {
        "nose",
        "back_mid",
        "tailbase",
        "spine_mid",
        "spine_base",
    }
    torso_indices = [
        idx for idx, name in enumerate(bodyparts) if name in torso_keypoints
    ]
    coords_subset = coords[:, torso_indices, :] if torso_indices else coords
    with np.errstate(invalid="ignore"):
        centroid = np.nanmean(coords_subset, axis=1)
    return centroid


def compute_cricket_centroid_2d(
    session_data: SessionData, time_slice: Optional[int] = None
) -> np.ndarray:
    cricket_coords_2d = session_data.get("cricket_coords_2d")
    if cricket_coords_2d is None:
        raise ValueError(
            "Cricket coordinates not available for this session (object session?)"
        )
    coords = cricket_coords_2d
    if time_slice is not None:
        assert time_slice > 0
        coords = coords[:time_slice]
    if coords.ndim == 2:
        return coords
    with np.errstate(invalid="ignore"):
        centroid = np.nanmean(coords, axis=1)
    return centroid


def compute_distance_mouse_cricket_2d(
    session_data: SessionData, time_slice: Optional[int] = None
) -> np.ndarray:
    mouse_centroid = compute_mouse_centroid_2d(session_data, time_slice)
    prey_centroid = compute_cricket_centroid_2d(session_data, time_slice)
    min_len = min(len(mouse_centroid), len(prey_centroid))
    if min_len == 0:
        return np.array([])
    mouse_centroid = mouse_centroid[:min_len]
    prey_centroid = prey_centroid[:min_len]
    assert mouse_centroid.shape[1] == 2
    assert prey_centroid.shape[1] == 2
    return np.linalg.norm(mouse_centroid - prey_centroid, axis=1)


def extract_all_features_to_dataframe(
    session_data: SessionData,
    time_slice: Optional[int] = None,
) -> pd.DataFrame:
    features = {}

    features["velocity"] = compute_velocity(session_data, time_slice)
    features["acceleration"] = compute_acceleration(session_data, time_slice)

    head_rear = compute_head_rear(session_data, time_slice)
    if head_rear.ndim == 2:
        for i, kp in enumerate(["nose_z", "ear_lf_z", "ear_rt_z"]):
            if i < head_rear.shape[1]:
                features[f"head_rear_{kp}"] = head_rear[:, i]
    else:
        features["head_rear"] = head_rear

    keypoints = ("nose", "tailbase")
    features["theta"] = compute_theta(session_data, keypoints, time_slice)
    features["turning_rate"] = compute_turning_rate(session_data, keypoints, time_slice)
    features["yaw_offset"] = compute_yaw_offset(session_data, time_slice)
    features["pitch_angle"] = compute_pitch_angle(session_data, keypoints, time_slice)

    forward_vel, side_vel, vertical_vel = compute_velocity_components(
        session_data, time_slice
    )
    features["velocity_forward"] = forward_vel
    features["velocity_side"] = side_vel
    features["velocity_vertical"] = vertical_vel

    features["manipulation_index"] = compute_manipulation_index_paws(
        session_data,
        time_slice,
        target_distance=compute_distance_mouse_cricket(session_data, time_slice),
    )

    features["freezing"] = compute_freezing(session_data, time_slice)
    features["curvature"] = compute_curvature(session_data, time_slice)

    centroid = compute_position_centroid(session_data, time_slice).squeeze()
    if centroid.ndim == 2:
        features["centroid_x"] = centroid[:, 0]
        features["centroid_y"] = centroid[:, 1]
        if centroid.shape[1] > 2:
            features["centroid_z"] = centroid[:, 2]
    elif centroid.ndim == 1:
        features["centroid"] = centroid

    dist_walls = compute_distance_to_walls(session_data, time_slice).squeeze()
    if dist_walls.ndim == 2:
        features["dist_wall_x_min"] = dist_walls[:, 0]
        features["dist_wall_x_max"] = dist_walls[:, 1]
        features["dist_wall_y_min"] = dist_walls[:, 2]
        features["dist_wall_y_max"] = dist_walls[:, 3]
    elif dist_walls.ndim == 1:
        features["dist_wall"] = dist_walls

    try:
        features["dist_mouse_cricket"] = compute_distance_mouse_cricket(
            session_data, time_slice
        )
    except (ValueError, KeyError):
        pass

    min_len = min(
        len(v.flatten()) if isinstance(v, np.ndarray) else len(v)
        for v in features.values()
        if isinstance(v, np.ndarray)
    )

    aligned_features = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            aligned_features[key] = value.flatten()[:min_len]
        else:
            aligned_features[key] = value
    return pd.DataFrame(aligned_features)


# ------------------------------------------------------------------------- #
# Helpers
# ------------------------------------------------------------------------- #

def _get_session_dataset(session_data: SessionData) -> xr.Dataset:
    session = session_data["session"]
    if not isinstance(session, xr.Dataset):
        raise TypeError("session_data['session'] must be an xarray Dataset")
    return session


def _get_arena_dataset(session_data: SessionData) -> xr.Dataset:
    arena_3d = session_data.get("arena_3d")
    if not isinstance(arena_3d, xr.Dataset):
        raise TypeError("session_data['arena_3d'] must be an xarray Dataset")
    return arena_3d


def _select_position_values(
    session: xr.Dataset, time_slice: Optional[int]
) -> np.ndarray:
    position = session.position
    if time_slice is not None:
        assert time_slice > 0
        position = position.isel(time=slice(0, time_slice))
    return position.values


