import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path
import pickle
from argparse import ArgumentParser
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

def get_velocity(session:xr.Dataset, time_slice:int)->np.ndarray:
    """
    Calculate velocity from position data
    
    Args:
        session (xr.Dataset): The input dataset containing position data
        time_slice (slice): The time slice to calculate velocity over
        
    Returns:
        np.ndarray: The velocity data
    """
    assert "position" in session, "session must contain 'position' variable"
    assert time_slice > 0, "time_slice must be positive"
    if time_slice:
        coordinates = session.position.isel(time=slice(0,time_slice)).values
    else:
        coordinates = session.position.values
    assert coordinates.shape[2] > 0, "position must have individuals dimension"
    centroids = coordinates.mean(axis=2).squeeze()
    velocity = np.vstack([np.zeros((1, 3)), np.diff(centroids, axis=0)])
    velocity = np.linalg.norm(velocity, axis=1)
    return velocity

def get_accelleration(session:xr.Dataset, time_slice:int)->np.ndarray:
    """
    Calculate accelleration from position data
    
    Args:
        session (xr.Dataset): The input dataset containing position data
        time_slice (slice): The time slice to calculate accelleration over
        
    Returns:
        np.ndarray: The accelleration data
    """
    assert "position" in session, "session must contain 'position' variable"
    assert time_slice > 0, "time_slice must be positive"
    if time_slice:
        coordinates = session.position.isel(time=slice(0,time_slice)).values
    else:
        coordinates = session.position.values
    assert coordinates.shape[2] > 0, "position must have individuals dimension"
    centroids = coordinates.mean(axis=2).squeeze()
    displacement = np.vstack([np.zeros((1, 3)), np.diff(centroids, axis=0)])
    velocity = displacement 
    accelleration = np.vstack([np.zeros((1, 3)), np.diff(velocity, axis=0)]) 
    accelleration = np.linalg.norm(accelleration, axis=1)
    return accelleration

def get_head_rear(session:xr.Dataset, time_slice:int)->np.ndarray:
    """
    Computes the verrtifical movement of the head as an heuristic of rear

    Args:
        session(xr.Dataset): input xrarray 
        time_slice: int max frame
    Returns:
        np.ndarray
    """
    assert "position" in session, "session must contain 'position' variable"
    assert time_slice > 0, "time_slice must be positive"
    assert "z" in session.position.space, "position must have 'z' in space dimension"
    rear = session["position"].sel(keypoints=["nose", "ear_lf", "ear_rt"], space="z").isel(time=slice(0, time_slice)).values.squeeze()
    return rear
def get_theta(session:xr.Dataset, time_slice:int, keypoints:tuple)->np.ndarray:
    """
    Computes the theta angle of the head in the 2D xy space
    
    Args:
        session(xr.Dataset): input xr.Dataset
        time_slice: int max frame
    Returns:
        The direction fo the head in the 2D xy space [-pi, pi]
    """
    assert "position" in session, "session must contain 'position' variable"
    assert time_slice > 0, "time_slice must be positive"
    assert len(keypoints) == 2, "keypoints must be tuple of 2 keypoints"
    keypoint_1, keypoint_2 = keypoints
    assert keypoint_1 in session.position.keypoints, f"keypoint '{keypoint_1}' not found"
    assert keypoint_2 in session.position.keypoints, f"keypoint '{keypoint_2}' not found"
    diff = session["position"].sel(keypoints=keypoint_1).values - session["position"].sel(keypoints=keypoint_2).values
    diff = diff[:time_slice]
    assert diff.shape[1] >= 2, "position must have at least 2 spatial dimensions (x, y)"
    diff_norm = np.linalg.norm(diff, axis=1)
    v_x = diff[:, 0] / diff_norm
    v_y = diff[:, 1] / diff_norm
    theta_head = np.arctan2(v_y, v_x)
    return theta_head
    
def get_turning_rate(session:xr.Dataset, time_slice:int, keypoints:tuple)->np.ndarray:
    """
    Computes the turning rate of the head
    Args:
        session(xr.Dataset): input xr.Dataset
        time_slice: int max frame
        keypoints: tuple of keypoints
    Returns:
        np.ndarray: turning rate
    """
    theta_head = get_theta(session, time_slice, keypoints)
    dtheta = np.vstack([np.zeros((1,)), np.diff(theta_head, axis=0)])
    #wrapped to [-pi, pi]
    dtheta = np.mod(dtheta + np.pi, 2 * np.pi) - np.pi
    dtheta = dtheta 
    return dtheta
def get_yaw_offset(session:xr.Dataset, time_slice:int)->np.ndarray:
    """
    Computes the yaw offset of the head
    Args:
        session(xr.Dataset): input xr.Dataset
        time_slice: int max frame
    Returns:
        np.ndarray: 0 (body and head are aligned), >0 (head is left of body), <0 (head is right of body)
    """
    theta_body = get_theta(session, time_slice, ("back_mid", "tailbase"))
    theta_head = get_theta(session, time_slice, ("nose", "tailbase"))
    delta_theta = theta_head - theta_body
    #wrap to [-pi, pi]
    delta_theta = np.mod(delta_theta + np.pi, 2 * np.pi) - np.pi
    return delta_theta

def get_pitch_angle(session:xr.Dataset, time_slice:int, keypoints:tuple)->np.ndarray:
    """
    Computes the pitch angle of the head

    Args:
        session(xr.Dataset): input xr.Dataset
        time_slice: int max frame
        keypoints: tuple of keypoints
    Returns:
        np.ndarray: pitch angle
        pitch = 0 when head and body are aligned, pitch > 0 when head is up, pitch < 0 when head is down
    """
    assert "position" in session, "session must contain 'position' variable"
    assert len(keypoints) == 2, "keypoints must be tuple of 2 keypoints"
    keypoint_1, keypoint_2 = keypoints
    assert keypoint_1 in session.position.keypoints, f"keypoint '{keypoint_1}' not found"
    assert keypoint_2 in session.position.keypoints, f"keypoint '{keypoint_2}' not found"
    assert "z" in session.position.space, "position must have 'z' in space dimension"
    if time_slice:
        v_nose = session.position.sel(keypoints=keypoint_1).values[:time_slice]
        v_tailbase = session.position.sel(keypoints=keypoint_2).values[:time_slice]
    else:
        v_nose = session.position.sel(keypoints=keypoint_1).values
        v_tailbase = session.position.sel(keypoints=keypoint_2).values
    v_body = v_nose - v_tailbase
    assert v_body.shape[1] >= 3, "position must have at least 3 spatial dimensions (x, y, z)"
    # compute horizontal (ground-plane) lenght 
    L_xy = np.linalg.norm(v_body[:, :2], axis=1)
    #compute pitch angle
    pitch_angle = np.arctan2(v_body[:, 2], L_xy)
    return pitch_angle
def get_velocity_components(session:xr.Dataset, time_slice:int)->np.ndarray:
    """
    Computes the forward, sideways, and vertical velocity components

    Args:
        session(xr.Dataset): input xr.Dataset
        time_slice: int max frame
    Returns:
        np.ndarray: forward, sideways, and vertical velocity components
    """
    assert "position" in session, "session must contain 'position' variable"
    assert time_slice > 0, "time_slice must be positive"
    assert "nose" in session.position.keypoints, "position must have 'nose' keypoint"
    assert "tailbase" in session.position.keypoints, "position must have 'tailbase' keypoint"
    # 1. Centroid positions and 3D displacement (velocity vector per frame)
    centroids = session.position.isel(time=slice(0, time_slice)).values.mean(axis=2).squeeze()
    # centroids.shape -> (T, 3)

    displacement = np.vstack([np.zeros((1, 3)), np.diff(centroids, axis=0)])
    # displacement.shape -> (T, 3)

    # 2. Body direction: tailbase -> nose, unit vector per frame
    nose = session.position.sel(keypoints="nose").isel(time=slice(0, time_slice)).values
    tail = session.position.sel(keypoints="tailbase").isel(time=slice(0, time_slice)).values

    # nose, tail: (T, 3, 1) -> drop individuals dim
    nose = nose.squeeze(-1)   # (T, 3)
    tail = tail.squeeze(-1)   # (T, 3)

    body_vector = nose - tail           # (T, 3)
    body_vector_norm = np.linalg.norm(body_vector, axis=1)  # (T,)
    body_vector_unit = body_vector / body_vector_norm[:, None]  # (T, 3)

    # 3. Sideways axis from cross product with global z-axis
    z_axis = np.array([0.0, 0.0, 1.0])  # (3,)

    side_axis = np.cross(z_axis, body_vector_unit)      # (T, 3)
    side_axis_norm = np.linalg.norm(side_axis, axis=1)  # (T,)
    side_axis_unit = side_axis / side_axis_norm[:, None]  # (T, 3)

    # 4. Per-frame projections (dot products)
    # forward: along body_vector_unit
    forward_velocity = np.sum(displacement * body_vector_unit, axis=1)  # (T,)

    # sideways: along side_axis_unit
    side_velocity = np.sum(displacement * side_axis_unit, axis=1)       # (T,)

    # vertical: along z-axis (just the z component of displacement)
    vertical_velocity = displacement[:, 2]                               # (T,)
    # or: vertical_velocity = np.sum(displacement * z_axis, axis=1)
    return forward_velocity, side_velocity, vertical_velocity
def get_manipulation_index_paws(session:xr.Dataset, time_slice:int)->np.ndarray:
    """
    Computes the manipulation index of the paws
    """
    assert "position" in session, "session must contain 'position' variable"
    assert time_slice > 0, "time_slice must be positive"
    assert "forepaw_lf" in session.position.keypoints, "position must have 'forepaw_lf' keypoint"
    assert "forepaw_rt" in session.position.keypoints, "position must have 'forepaw_rt' keypoint"
    v_lf = session.position.sel(keypoints="forepaw_lf").values.squeeze()[:time_slice]
    v_rt = session.position.sel(keypoints="forepaw_rt").values.squeeze()[:time_slice]
    body_speed = get_velocity(session, time_slice)
    lf_disp = np.vstack([np.zeros((1, 3)), np.diff(v_lf, axis=0)])
    rt_disp = np.vstack([np.zeros((1, 3)), np.diff(v_rt, axis=0)])
    lf_speed = np.linalg.norm(lf_disp, axis=1)
    rt_speed = np.linalg.norm(rt_disp, axis=1)
    manipulation_idx_lf = lf_speed / body_speed
    manipulation_idx_rt = rt_speed / body_speed
    return manipulation_idx_lf, manipulation_idx_rt

def get_freezing(session:xr.Dataset, time_slice:int)->np.ndarray:
    """
    Computes the freezing of the animal
    """
    velocity = get_velocity(session, time_slice)
    threshold = np.percentile(velocity, 10)
    freezing = velocity < threshold
    return freezing
def get_curvature(session:xr.Dataset, time_slice:int)->np.ndarray:
    """
    Computes the curvature of the path
    """
    turning_rate = get_turning_rate(session, time_slice,  keypoints=("nose", "tailbase"))
    velocity = get_velocity(session, time_slice)
    assert len(turning_rate) == len(velocity), "turning_rate and velocity must have same length"
    curvature = np.abs(turning_rate.squeeze()) / velocity + np.random.randn(len(velocity)) * 0.01
    return curvature
def get_position_centroid(session:xr.Dataset, time_slice:int)->np.ndarray:
    """
    Computes the centroid of the position
    """
    assert "position" in session, "session must contain 'position' variable"
    assert time_slice > 0, "time_slice must be positive"
    centroid = session.position.isel(time=slice(0, time_slice)).values.mean(axis=2).squeeze()
    return centroid

def get_distance_to_walls(session:xr.Dataset, arena:xr.Dataset, time_slice:int)->np.ndarray:
    """
    Computes the distance to the walls
    """
    assert "position" in arena, "arena must contain 'position' variable"
    assert "x" in arena.position.space and "y" in arena.position.space, "arena position must have 'x' and 'y' in space dimension"
    centroid_position = get_position_centroid(session, time_slice=time_slice)
    arena_2d = arena.position.sel(space=["x", "y"]).values.squeeze()
    x_min, x_max = arena_2d[:, 0].min(), arena_2d[:, 0].max()
    y_min, y_max = arena_2d[:, 1].min(), arena_2d[:, 1].max()

    # Use only 2D position from centroids (first two dimensions)
    centroid_2d = centroid_position[:, :2]  # shape: (frame, 2)

    # Compute distance to nearest wall for each frame
    dist_to_x_min = np.abs(x_min - centroid_2d[:, 0])
    dist_to_x_max = np.abs(x_max - centroid_2d[:, 0])
    dist_to_y_min = np.abs(y_min - centroid_2d[:, 1])
    dist_to_y_max = np.abs(y_max - centroid_2d[:, 1])

    # For each frame, find the minimum distance to any wall
    d_wall = np.minimum.reduce([dist_to_x_min, dist_to_x_max, dist_to_y_min, dist_to_y_max])  # shape: (frame,)
    return d_wall

def load_cricket_coordinates(pickle_path:Path)->np.ndarray:
    """
    Loads the coordinates of the cricket
    """
    assert pickle_path.exists(), f"pickle_path does not exist: {pickle_path}"
    data_cricket = pickle.load(open(pickle_path, "rb"))
    assert isinstance(data_cricket, dict), "pickle file must contain a dictionary"
    # we have a dict with "metadata" and one key for each frame:
    # we want to extract the coordinates for each frame and put them in an array:
    coordinates = []
    for frame in data_cricket.keys():
        #skip metadata
        if frame == "metadata":
            continue
        # get the coordinates:
        assert "coordinates" in data_cricket[frame], f"frame {frame} must contain 'coordinates'"
        coordinates.append(data_cricket[frame]["coordinates"][0][0])
    return np.array(coordinates)
def convert_cricket_coordinates(cricket_coordinates:np.ndarray, arena_3d:xr.Dataset, arena_views:xr.Dataset)->np.ndarray:
    """
    Converts the cricket coordinates to arena floor coordinates
    """
    assert "position" in arena_3d, "arena_3d must contain 'position' variable"
    assert "position" in arena_views, "arena_views must contain 'position' variable"
    assert "x" in arena_3d.position.space and "y" in arena_3d.position.space, "arena_3d position must have 'x' and 'y' in space dimension"
    assert "central" in arena_views.position.view, "arena_views must have 'central' view"
    assert cricket_coordinates.shape[1] == 2, "cricket_coordinates must have shape (N, 2)"
    # Extract 4 corner correspondences
    arena_corners_3d = arena_3d["position"].sel(space=["x", "y"]).values.squeeze()[:, :4]  # shape: (2, 4) - [x, y] x 4 corners
    arena_corners_2d = arena_views["position"].sel(view="central").values.squeeze()[:, :4]  # shape: (2, 4) - [x, y] x 4 corners
        # For 3D arena coordinates (destination)
    arena_3d_homogeneous = np.vstack([arena_corners_3d, np.ones((1, 4))])  # shape: (3, 4) - each column is [x, y, 1]
    # For 2D image coordinates (source)
    arena_2d_homogeneous = np.vstack([arena_corners_2d, np.ones((1, 4))])  # shape: (3, 4) - each column is [x, y, 1]

    # Compute homography using Direct Linear Transform (DLT)
    # H maps from 2D image to 3D arena: [x', y', 1]^T = H @ [x, y, 1]^T
    # Using pseudo-inverse: H = arena_3d_homogeneous @ pinv(arena_2d_homogeneous)
    H = arena_3d_homogeneous @ np.linalg.pinv(arena_2d_homogeneous)

    # Apply homography to cricket coordinates
    # Prepare cricket coordinates in homogeneous form: each column is [x, y, 1]
    cricket_2d = cricket_coordinates.squeeze()  # shape: (N, 2)
    cricket_2d_homogeneous = np.vstack([cricket_2d.T, np.ones((1, cricket_2d.shape[0]))])  # shape: (3, N)

    # Transform: cricket_3d_homogeneous = H @ cricket_2d_homogeneous
    cricket_3d_homogeneous = H @ cricket_2d_homogeneous  # shape: (3, N)

    # Normalize by homogeneous coordinate (last row) to get [x, y, 1]
    # Divide each column by its third element (the homogeneous coordinate)
    cricket_3d_homogeneous = cricket_3d_homogeneous / cricket_3d_homogeneous[2, :]  # shape: (3, N)

    # Extract x, y coordinates (first two rows)
    cricket_converted = cricket_3d_homogeneous[:2, :].T  # shape: (N, 2)
    return cricket_converted
def get_distance_mouse_cricket(mouse_coordinates:np.ndarray, cricket_coordinates:np.ndarray)->np.ndarray:
    """
    Computes the distance between the mouse and the cricket
    """
    assert mouse_coordinates.shape[1] >= 2, "mouse_coordinates must have at least 2 columns (x, y)"
    assert cricket_coordinates.shape[1] == 2, "cricket_coordinates must have shape (N, 2)"
    assert len(mouse_coordinates) == len(cricket_coordinates), "mouse and cricket coordinates must have same length"
    return np.linalg.norm(mouse_coordinates[:, :2] - cricket_coordinates, axis=1)


if __name__ == "__main__":
    # let's hardoce the session and the other paths for testing:
    session = xr.open_dataset("/Users/thomasbush/Downloads/multicam_video_2025-05-07T12_16_20_cropped-v2_20250701121021_triangulated_points_20250802-065459.h5")
    arena_3d = xr.open_dataset("/Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/newarena.h5")
    arena_views = xr.open_dataset("/Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/3d-setup/tests/assets/arena_views.h5")
    # let's get the velocity:
    velocity = get_velocity(session, time_slice=1000)
    print(velocity.shape)
    # let's get the accelleration:
    accelleration = get_accelleration(session, time_slice=1000)
    print(accelleration.shape)
    # let's get the head rear:
    head_rear = get_head_rear(session, time_slice=1000)
    print(head_rear.shape)
    # let's print the other features:
    print(get_theta(session, time_slice=1000, keypoints=("nose", "tailbase")).shape)
    print(get_turning_rate(session, time_slice=1000, keypoints=("nose", "tailbase")).shape)
    print(get_yaw_offset(session, time_slice=1000).shape)
    print(get_pitch_angle(session, time_slice=1000, keypoints=("nose", "tailbase")).shape)
    print(get_velocity_components(session, time_slice=1000)[0].shape)
    print(get_manipulation_index_paws(session, time_slice=1000)[0].shape)
    print(get_freezing(session, time_slice=1000).shape)
    print(get_curvature(session, time_slice=1000).shape)
    print(get_position_centroid(session, time_slice=1000).shape)
    print(get_distance_to_walls(session, arena_3d, time_slice=1000).shape)

    