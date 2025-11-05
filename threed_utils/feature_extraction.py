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
from typing import List, Tuple, Dict, Any, Optional

class FeatureExtractor:
    def __init__(self, base_dir: Path):
        """
        Initialize FeatureExtractor with base directory.
        
        Args:
            base_dir: Directory like M30/20250507/cricket/115438
        """
        self.base_dir = Path(base_dir)
        assert self.base_dir.exists(), f"Base directory does not exist: {self.base_dir}"
        
        self.session_dir = self._find_latest_session()
        self.session = None
        self.arena_3d = None
        self.arena_views = None
        self.cricket_coords_2d = None
        self.cricket_coords_3d = None
        
        self._load_data()
    
    def _find_latest_session(self) -> Path:
        """Find the most recent session directory (multicam_video_*_cropped*)"""
        candidates = [d for d in self.base_dir.iterdir() 
                     if d.is_dir() and "multicam_video" in d.name and "cropped" in d.name]
        if not candidates:
            raise FileNotFoundError(f"No session directories found in {self.base_dir}")
        
        v2_candidates = [d for d in candidates if "-v2" in d.name]
        if v2_candidates:
            return max(v2_candidates, key=lambda p: p.stat().st_mtime)
        return max(candidates, key=lambda p: p.stat().st_mtime)
    
    def _find_latest_pickle(self, pattern: str) -> Path:
        """Find the most recent pickle file matching pattern"""
        candidates = list(self.session_dir.rglob(f"*{pattern}*_full.pickle"))
        if not candidates:
            raise FileNotFoundError(f"No pickle files found matching pattern '*{pattern}*_full.pickle' in {self.session_dir}")
        return max(candidates, key=lambda p: p.stat().st_mtime)
    
    def _find_triangulated_h5(self) -> Path:
        """Find the triangulated h5 file"""
        candidates = list(self.session_dir.rglob("*triangulated*.h5"))
        if not candidates:
            raise FileNotFoundError(f"No triangulated h5 files found in {self.session_dir}")
        return max(candidates, key=lambda p: p.stat().st_mtime)
    
    def _load_data(self):
        """Load all required data files"""
        script_dir = Path(__file__).parent.parent
        arena_3d_path = script_dir / "data" / "newarena.h5"
        if not arena_3d_path.exists():
            arena_3d_path = script_dir / "tests" / "assets" / "newarena.h5"
        arena_views_path = script_dir / "tests" / "assets" / "arena_views.h5"
        
        assert arena_3d_path.exists(), f"Arena 3D file not found: {arena_3d_path}"
        assert arena_views_path.exists(), f"Arena views file not found: {arena_views_path}"
        
        triangulated_h5 = self._find_triangulated_h5()
        cricket_pickle = self._find_latest_pickle("cricket")
        
        self.session = xr.open_dataset(triangulated_h5)
        self.arena_3d = xr.open_dataset(arena_3d_path)
        self.arena_views = xr.open_dataset(arena_views_path)
        
        self.cricket_coords_2d = self._load_2d_coordinates(cricket_pickle)
        cricket_coords_2d_mean = np.nanmean(self.cricket_coords_2d, axis=1)
        self.cricket_coords_3d = self._convert_cricket_coordinates(cricket_coords_2d_mean, self.arena_3d, self.arena_views)
    
    def _load_2d_coordinates(self, pickle_path: Path) -> np.ndarray:
        """Loads the coordinates from pickle file"""
        data = pickle.load(open(pickle_path, "rb"))
        coordinates = []
        for frame in data.keys():
            if frame == "metadata":
                continue
            frame_coords = data[frame]["coordinates"]
            if isinstance(frame_coords, list):
                arr = np.array(frame_coords)
                if arr.ndim == 3:
                    arr = arr[:, 0, :]
                elif arr.ndim == 2:
                    pass
                else:
                    arr = arr.reshape(-1, 2)
            else:
                arr = np.array(frame_coords)
                if arr.ndim == 3:
                    arr = arr[:, 0, :]
                elif arr.ndim == 2:
                    pass
                else:
                    arr = arr.reshape(-1, 2)
            coordinates.append(arr)
        max_n = max(coord.shape[0] for coord in coordinates)
        coords_padded = []
        for arr in coordinates:
            if arr.shape[0] < max_n:
                pad_width = ((0, max_n - arr.shape[0]), (0, 0))
                arr = np.pad(arr, pad_width, mode='constant', constant_values=np.nan)
            coords_padded.append(arr)
        return np.array(coords_padded)
    
    def _convert_cricket_coordinates(self, cricket_coordinates: np.ndarray, arena_3d: xr.Dataset, arena_views: xr.Dataset) -> np.ndarray:
        """Converts cricket coordinates to arena floor coordinates"""
        assert "position" in arena_3d, "arena_3d must contain 'position' variable"
        assert "position" in arena_views, "arena_views must contain 'position' variable"
        assert "x" in arena_3d.position.space and "y" in arena_3d.position.space
        assert "central" in arena_views.position.view
        assert cricket_coordinates.shape[1] == 2, "cricket_coordinates must have shape (N, 2)"
        
        arena_corners_3d = arena_3d["position"].sel(space=["x", "y"]).values.squeeze()[:, :4]
        arena_corners_2d = arena_views["position"].sel(view="central").values.squeeze()[:, :4]
        
        arena_3d_homogeneous = np.vstack([arena_corners_3d, np.ones((1, 4))])
        arena_2d_homogeneous = np.vstack([arena_corners_2d, np.ones((1, 4))])
        
        H = arena_3d_homogeneous @ np.linalg.pinv(arena_2d_homogeneous)
        
        cricket_2d = cricket_coordinates.squeeze()
        cricket_2d_homogeneous = np.vstack([cricket_2d.T, np.ones((1, cricket_2d.shape[0]))])
        
        cricket_3d_homogeneous = H @ cricket_2d_homogeneous
        cricket_3d_homogeneous = cricket_3d_homogeneous / cricket_3d_homogeneous[2, :]
        
        return cricket_3d_homogeneous[:2, :].T
    
    def get_velocity(self, time_slice: Optional[int] = None) -> np.ndarray:
        """Calculate velocity from position data"""
        assert "position" in self.session
        if time_slice is not None:
            assert time_slice > 0
            coordinates = self.session.position.isel(time=slice(0, time_slice)).values
        else:
            coordinates = self.session.position.values
        assert coordinates.shape[2] > 0
        centroids = coordinates.mean(axis=2).squeeze()
        velocity = np.vstack([np.zeros((1, 3)), np.diff(centroids, axis=0)])
        velocity = np.linalg.norm(velocity, axis=1)
        return velocity
    
    def get_accelleration(self, time_slice: Optional[int] = None) -> np.ndarray:
        """Calculate acceleration from position data"""
        assert "position" in self.session
        if time_slice is not None:
            assert time_slice > 0
            coordinates = self.session.position.isel(time=slice(0, time_slice)).values
        else:
            coordinates = self.session.position.values
        assert coordinates.shape[2] > 0
        centroids = coordinates.mean(axis=2).squeeze()
        displacement = np.vstack([np.zeros((1, 3)), np.diff(centroids, axis=0)])
        velocity = displacement
        accelleration = np.vstack([np.zeros((1, 3)), np.diff(velocity, axis=0)])
        accelleration = np.linalg.norm(accelleration, axis=1)
        return accelleration
    
    def get_head_rear(self, time_slice: Optional[int] = None) -> np.ndarray:
        """Computes the vertical movement of the head as heuristic of rear"""
        assert "position" in self.session
        assert "z" in self.session.position.space
        if time_slice is not None:
            assert time_slice > 0
            rear = self.session["position"].sel(keypoints=["nose", "ear_lf", "ear_rt"], space="z").isel(time=slice(0, time_slice)).values.squeeze()
        else:
            rear = self.session["position"].sel(keypoints=["nose", "ear_lf", "ear_rt"], space="z").values.squeeze()
        return rear
    
    def get_theta(self, keypoints: tuple, time_slice: Optional[int] = None) -> np.ndarray:
        """Computes the theta angle of the head in the 2D xy space"""
        assert "position" in self.session
        assert len(keypoints) == 2
        keypoint_1, keypoint_2 = keypoints
        assert keypoint_1 in self.session.position.keypoints
        assert keypoint_2 in self.session.position.keypoints
        diff = self.session["position"].sel(keypoints=keypoint_1).values - self.session["position"].sel(keypoints=keypoint_2).values
        if time_slice is not None:
            assert time_slice > 0
            diff = diff[:time_slice]
        assert diff.shape[1] >= 2
        diff_norm = np.linalg.norm(diff, axis=1)
        v_x = diff[:, 0] / diff_norm
        v_y = diff[:, 1] / diff_norm
        theta_head = np.arctan2(v_y, v_x)
        return theta_head
    
    def get_turning_rate(self, keypoints: tuple, time_slice: Optional[int] = None) -> np.ndarray:
        """Computes the turning rate of the head"""
        theta_head = self.get_theta(keypoints, time_slice)
        dtheta = np.vstack([np.zeros((1,)), np.diff(theta_head, axis=0)])
        dtheta = np.mod(dtheta + np.pi, 2 * np.pi) - np.pi
        return dtheta
    
    def get_yaw_offset(self, time_slice: Optional[int] = None) -> np.ndarray:
        """Computes the yaw offset of the head"""
        theta_body = self.get_theta(("back_mid", "tailbase"), time_slice)
        theta_head = self.get_theta(("nose", "tailbase"), time_slice)
        delta_theta = theta_head - theta_body
        delta_theta = np.mod(delta_theta + np.pi, 2 * np.pi) - np.pi
        return delta_theta
    
    def get_pitch_angle(self, keypoints: tuple, time_slice: Optional[int] = None) -> np.ndarray:
        """Computes the pitch angle of the head"""
        assert "position" in self.session
        assert len(keypoints) == 2
        keypoint_1, keypoint_2 = keypoints
        assert keypoint_1 in self.session.position.keypoints
        assert keypoint_2 in self.session.position.keypoints
        assert "z" in self.session.position.space
        if time_slice is not None:
            assert time_slice > 0
            v_nose = self.session.position.sel(keypoints=keypoint_1).values[:time_slice]
            v_tailbase = self.session.position.sel(keypoints=keypoint_2).values[:time_slice]
        else:
            v_nose = self.session.position.sel(keypoints=keypoint_1).values
            v_tailbase = self.session.position.sel(keypoints=keypoint_2).values
        v_body = v_nose - v_tailbase
        assert v_body.shape[1] >= 3
        L_xy = np.linalg.norm(v_body[:, :2], axis=1)
        pitch_angle = np.arctan2(v_body[:, 2], L_xy)
        return pitch_angle
    
    def get_velocity_components(self, time_slice: Optional[int] = None) -> tuple:
        """Computes the forward, sideways, and vertical velocity components"""
        assert "position" in self.session
        assert "nose" in self.session.position.keypoints
        assert "tailbase" in self.session.position.keypoints
        if time_slice is not None:
            assert time_slice > 0
            centroids = self.session.position.isel(time=slice(0, time_slice)).values.mean(axis=2).squeeze()
            nose = self.session.position.sel(keypoints="nose").isel(time=slice(0, time_slice)).values
            tail = self.session.position.sel(keypoints="tailbase").isel(time=slice(0, time_slice)).values
        else:
            centroids = self.session.position.values.mean(axis=2).squeeze()
            nose = self.session.position.sel(keypoints="nose").values
            tail = self.session.position.sel(keypoints="tailbase").values
        
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
    
    def get_manipulation_index_paws(self, time_slice: Optional[int] = None) -> tuple:
        """Computes the manipulation index of the paws"""
        assert "position" in self.session
        assert "forepaw_lf" in self.session.position.keypoints
        assert "forepaw_rt" in self.session.position.keypoints
        v_lf = self.session.position.sel(keypoints="forepaw_lf").values.squeeze()
        v_rt = self.session.position.sel(keypoints="forepaw_rt").values.squeeze()
        if time_slice is not None:
            assert time_slice > 0
            v_lf = v_lf[:time_slice]
            v_rt = v_rt[:time_slice]
        body_speed = self.get_velocity(time_slice)
        lf_disp = np.vstack([np.zeros((1, 3)), np.diff(v_lf, axis=0)])
        rt_disp = np.vstack([np.zeros((1, 3)), np.diff(v_rt, axis=0)])
        lf_speed = np.linalg.norm(lf_disp, axis=1)
        rt_speed = np.linalg.norm(rt_disp, axis=1)
        manipulation_idx_lf = lf_speed / body_speed
        manipulation_idx_rt = rt_speed / body_speed
        return manipulation_idx_lf, manipulation_idx_rt
    
    def get_freezing(self, time_slice: Optional[int] = None) -> np.ndarray:
        """Computes the freezing of the animal"""
        velocity = self.get_velocity(time_slice)
        threshold = np.percentile(velocity, 10)
        freezing = velocity < threshold
        return freezing
    
    def get_curvature(self, time_slice: Optional[int] = None) -> np.ndarray:
        """Computes the curvature of the path"""
        turning_rate = self.get_turning_rate(("nose", "tailbase"), time_slice)
        velocity = self.get_velocity(time_slice)
        assert len(turning_rate) == len(velocity)
        curvature = np.abs(turning_rate.squeeze()) / velocity + np.random.randn(len(velocity)) * 0.01
        return curvature
    
    def get_position_centroid(self, time_slice: Optional[int] = None) -> np.ndarray:
        """Computes the centroid of the position"""
        assert "position" in self.session
        if time_slice is not None:
            assert time_slice > 0
            centroid = self.session.position.isel(time=slice(0, time_slice)).values.mean(axis=2).squeeze()
        else:
            centroid = self.session.position.values.mean(axis=2).squeeze()
        return centroid
    
    def get_distance_to_walls(self, time_slice: Optional[int] = None) -> np.ndarray:
        """Computes the distance to the walls"""
        assert "position" in self.arena_3d
        assert "x" in self.arena_3d.position.space and "y" in self.arena_3d.position.space
        centroid_position = self.get_position_centroid(time_slice=time_slice)
        arena_2d = self.arena_3d.position.sel(space=["x", "y"]).values.squeeze()
        x_min, x_max = arena_2d[:, 0].min(), arena_2d[:, 0].max()
        y_min, y_max = arena_2d[:, 1].min(), arena_2d[:, 1].max()
        
        centroid_2d = centroid_position[:, :2]
        
        dist_to_x_min = np.abs(x_min - centroid_2d[:, 0])
        dist_to_x_max = np.abs(x_max - centroid_2d[:, 0])
        dist_to_y_min = np.abs(y_min - centroid_2d[:, 1])
        dist_to_y_max = np.abs(y_max - centroid_2d[:, 1])
        
        d_wall = np.minimum.reduce([dist_to_x_min, dist_to_x_max, dist_to_y_min, dist_to_y_max])
        return d_wall
    
    def get_distance_mouse_cricket(self, time_slice: Optional[int] = None) -> np.ndarray:
        """Computes the distance between the mouse and the cricket"""
        mouse_centroid = self.get_position_centroid(time_slice)
        min_len = min(len(mouse_centroid), len(self.cricket_coords_3d))
        mouse_centroid = mouse_centroid[:min_len]
        cricket_coords = self.cricket_coords_3d[:min_len]
        
        assert mouse_centroid.shape[1] >= 2
        assert cricket_coords.shape[1] == 2
        assert len(mouse_centroid) == len(cricket_coords)
        
        if mouse_centroid.ndim == 3:
            mouse_centroid = np.nanmean(mouse_centroid[:, :, :2], axis=1)
        else:
            mouse_centroid = mouse_centroid[:, :2]
        
        return np.linalg.norm(mouse_centroid - cricket_coords, axis=1)


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract features from tracking data")
    parser.add_argument("directory", type=str, help="Base directory path (e.g., M30/20250507/cricket/115438)")
    parser.add_argument("--time-slice", type=int, default=None, help="Time slice to process (default: all data)")
    args = parser.parse_args()
    
    extractor = FeatureExtractor(args.directory)
    
    print(f"Velocity: {extractor.get_velocity(args.time_slice).shape}")
    print(f"Acceleration: {extractor.get_accelleration(args.time_slice).shape}")
    print(f"Head rear: {extractor.get_head_rear(args.time_slice).shape}")
    print(f"Theta: {extractor.get_theta(('nose', 'tailbase'), args.time_slice).shape}")
    print(f"Turning rate: {extractor.get_turning_rate(('nose', 'tailbase'), args.time_slice).shape}")
    print(f"Yaw offset: {extractor.get_yaw_offset(args.time_slice).shape}")
    print(f"Pitch angle: {extractor.get_pitch_angle(('nose', 'tailbase'), args.time_slice).shape}")
    print(f"Velocity components: {extractor.get_velocity_components(args.time_slice)[0].shape}")
    print(f"Manipulation index: {extractor.get_manipulation_index_paws(args.time_slice)[0].shape}")
    print(f"Freezing: {extractor.get_freezing(args.time_slice).shape}")
    print(f"Curvature: {extractor.get_curvature(args.time_slice).shape}")
    print(f"Position centroid: {extractor.get_position_centroid(args.time_slice).shape}")
    print(f"Distance to walls: {extractor.get_distance_to_walls(args.time_slice).shape}")
    print(f"Distance mouse-cricket: {extractor.get_distance_mouse_cricket(args.time_slice).shape}")
