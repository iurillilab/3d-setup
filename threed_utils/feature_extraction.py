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
            base_dir: Directory like M30/20250507/cricket/115438 or M30 (for multi-session)
        """
        self.base_dir = Path(base_dir)
        assert self.base_dir.exists(), f"Base directory does not exist: {self.base_dir}"
        
        self.sessions = {}
        self.arena_3d = None
        self.arena_views = None
        
        if self._is_multi_session_dir():
            self._load_all_sessions()
        else:
            self._load_single_session()
    
    def _is_multi_session_dir(self) -> bool:
        """Check if directory is a top-level directory (M30) with date subdirectories"""
        date_dirs = [d for d in self.base_dir.iterdir() 
                     if d.is_dir() and d.name.isdigit() and len(d.name) == 8]
        return len(date_dirs) > 0
    
    def _find_all_session_dirs(self) -> list:
        """Find all session directories recursively"""
        pattern = "**/multicam_video*cropped*"
        all_dirs = [d for d in self.base_dir.glob(pattern) if d.is_dir() and "multicam_video" in d.name and "cropped" in d.name]
        
        v2_dirs = [d for d in all_dirs if "-v2" in d.name]
        non_v2_dirs = [d for d in all_dirs if "-v2" not in d.name]
        
        session_dirs = []
        seen_parents = set()
        
        for d in v2_dirs:
            parent_dir = d.parent
            if parent_dir not in seen_parents:
                v2_candidates = [sd for sd in parent_dir.iterdir() 
                               if sd.is_dir() and "multicam_video" in sd.name and "cropped" in sd.name and "-v2" in sd.name]
                if v2_candidates:
                    session_dirs.append(max(v2_candidates, key=lambda p: p.stat().st_mtime))
                    seen_parents.add(parent_dir)
        
        for d in non_v2_dirs:
            parent_dir = d.parent
            if parent_dir not in seen_parents:
                candidates = [sd for sd in parent_dir.iterdir() 
                            if sd.is_dir() and "multicam_video" in sd.name and "cropped" in sd.name]
                if candidates:
                    session_dirs.append(max(candidates, key=lambda p: p.stat().st_mtime))
                    seen_parents.add(parent_dir)
        
        return session_dirs
    
    def _load_all_sessions(self):
        """Load all sessions and store them in self.sessions dict"""
        session_dirs = self._find_all_session_dirs()
        if not session_dirs:
            raise FileNotFoundError(f"No session directories found in {self.base_dir}")
        
        script_dir = Path(__file__).parent.parent
        arena_3d_path = script_dir / "data" / "newarena.h5"
        if not arena_3d_path.exists():
            arena_3d_path = script_dir / "tests" / "assets" / "newarena.h5"
        arena_views_path = script_dir / "tests" / "assets" / "arena_views.h5"
        
        assert arena_3d_path.exists(), f"Arena 3D file not found: {arena_3d_path}"
        assert arena_views_path.exists(), f"Arena views file not found: {arena_views_path}"
        
        self.arena_3d = xr.open_dataset(arena_3d_path)
        self.arena_views = xr.open_dataset(arena_views_path)
        
        print(f"Found {len(session_dirs)} session directories to process")
        loaded = 0
        failed = []
        for session_dir in tqdm(session_dirs, desc="Loading sessions"):
            try:
                session_data = self._load_session_data(session_dir, self.arena_3d, self.arena_views)
                self.sessions[str(session_dir)] = session_data
                loaded += 1
            except Exception as e:
                failed.append((session_dir, str(e)))
                continue
        
        print(f"\nSuccessfully loaded {loaded}/{len(session_dirs)} sessions")
        if failed:
            print(f"\nFailed to load {len(failed)} sessions (missing triangulated h5 files):")
            for session_dir, error in failed[:5]:  # Show first 5
                print(f"  - {session_dir.parent.name}/{session_dir.name}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")
    
    def _load_single_session(self):
        """Load single session (backward compatibility)"""
        script_dir = Path(__file__).parent.parent
        arena_3d_path = script_dir / "data" / "newarena.h5"
        if not arena_3d_path.exists():
            arena_3d_path = script_dir / "tests" / "assets" / "newarena.h5"
        arena_views_path = script_dir / "tests" / "assets" / "arena_views.h5"
        
        assert arena_3d_path.exists(), f"Arena 3D file not found: {arena_3d_path}"
        assert arena_views_path.exists(), f"Arena views file not found: {arena_views_path}"
        
        self.arena_3d = xr.open_dataset(arena_3d_path)
        self.arena_views = xr.open_dataset(arena_views_path)
        
        self.session_dir = self._find_latest_session()
        session_data = self._load_session_data(self.session_dir, self.arena_3d, self.arena_views)
        self.sessions[str(self.session_dir)] = session_data
        self.session = session_data["session"]
        self.cricket_coords_2d = session_data["cricket_coords_2d"]
        self.cricket_coords_3d = session_data["cricket_coords_3d"]
    
    def _is_cricket_session(self, session_dir: Path) -> bool:
        """Check if session is a cricket session based on path"""
        return "cricket" in str(session_dir).lower()
    
    def _load_session_data(self, session_dir: Path, arena_3d: xr.Dataset, arena_views: xr.Dataset) -> dict:
        """Load data for a single session"""
        triangulated_h5 = self._find_triangulated_h5_in_dir(session_dir)
        session = xr.open_dataset(triangulated_h5)
        
        result = {
            "session_dir": session_dir,
            "session": session,
            "arena_3d": arena_3d,
            "arena_views": arena_views,
            "cricket_coords_2d": None,
            "cricket_coords_3d": None
        }
        
        if self._is_cricket_session(session_dir):
            try:
                cricket_pickle = self._find_latest_pickle_in_dir(session_dir, "cricket")
                cricket_coords_2d = self._load_2d_coordinates(cricket_pickle)
                cricket_coords_2d_mean = np.nanmean(cricket_coords_2d, axis=1)
                cricket_coords_3d = self._convert_cricket_coordinates(cricket_coords_2d_mean, arena_3d, arena_views)
                result["cricket_coords_2d"] = cricket_coords_2d
                result["cricket_coords_3d"] = cricket_coords_3d
            except Exception as e:
                print(f"Warning: Could not load cricket coordinates for {session_dir}: {e}")
        
        return result
    
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
    
    def _find_latest_pickle_in_dir(self, session_dir: Path, pattern: str) -> Path:
        """Find the most recent pickle file matching pattern in given directory"""
        candidates = list(session_dir.rglob(f"*{pattern}*_full.pickle"))
        if not candidates:
            raise FileNotFoundError(f"No pickle files found matching pattern '*{pattern}*_full.pickle' in {session_dir}")
        return max(candidates, key=lambda p: p.stat().st_mtime)
    
    def _find_triangulated_h5_in_dir(self, session_dir: Path) -> Path:
        """Find the triangulated h5 file in given directory"""
        candidates = list(session_dir.rglob("*triangulated*.h5"))
        if not candidates:
            raise FileNotFoundError(f"No triangulated h5 files found in {session_dir}")
        return max(candidates, key=lambda p: p.stat().st_mtime)
    
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
    
    def _get_session_data(self, session_path: Optional[str] = None):
        """Get session data for a specific session path, or current session if single mode"""
        if session_path:
            if session_path not in self.sessions:
                raise KeyError(f"Session path not found: {session_path}")
            return self.sessions[session_path]
        if hasattr(self, 'session') and self.session is not None:
            return {
                "session": self.session,
                "arena_3d": self.arena_3d,
                "arena_views": self.arena_views,
                "cricket_coords_2d": getattr(self, 'cricket_coords_2d', None),
                "cricket_coords_3d": getattr(self, 'cricket_coords_3d', None)
            }
        if len(self.sessions) == 1:
            return list(self.sessions.values())[0]
        raise ValueError("Must specify session_path when multiple sessions are loaded")
    
    def get_velocity(self, time_slice: Optional[int] = None, session_path: Optional[str] = None) -> np.ndarray:
        """Calculate velocity from position data"""
        session_data = self._get_session_data(session_path)
        session = session_data["session"]
        assert "position" in session
        if time_slice is not None:
            assert time_slice > 0
            coordinates = session.position.isel(time=slice(0, time_slice)).values
        else:
            coordinates = session.position.values
        assert coordinates.shape[2] > 0
        centroids = coordinates.mean(axis=2).squeeze()
        velocity = np.vstack([np.zeros((1, 3)), np.diff(centroids, axis=0)])
        velocity = np.linalg.norm(velocity, axis=1)
        return velocity
    
    def get_accelleration(self, time_slice: Optional[int] = None, session_path: Optional[str] = None) -> np.ndarray:
        """Calculate acceleration from position data"""
        session_data = self._get_session_data(session_path)
        session = session_data["session"]
        assert "position" in session
        if time_slice is not None:
            assert time_slice > 0
            coordinates = session.position.isel(time=slice(0, time_slice)).values
        else:
            coordinates = session.position.values
        assert coordinates.shape[2] > 0
        centroids = coordinates.mean(axis=2).squeeze()
        displacement = np.vstack([np.zeros((1, 3)), np.diff(centroids, axis=0)])
        velocity = displacement
        accelleration = np.vstack([np.zeros((1, 3)), np.diff(velocity, axis=0)])
        accelleration = np.linalg.norm(accelleration, axis=1)
        return accelleration
    
    def get_head_rear(self, time_slice: Optional[int] = None, session_path: Optional[str] = None) -> np.ndarray:
        """Computes the vertical movement of the head as heuristic of rear"""
        session_data = self._get_session_data(session_path)
        session = session_data["session"]
        assert "position" in session
        assert "z" in session.position.space
        if time_slice is not None:
            assert time_slice > 0
            rear = session["position"].sel(keypoints=["nose", "ear_lf", "ear_rt"], space="z").isel(time=slice(0, time_slice)).values.squeeze()
        else:
            rear = session["position"].sel(keypoints=["nose", "ear_lf", "ear_rt"], space="z").values.squeeze()
        return rear
    
    def get_theta(self, keypoints: tuple, time_slice: Optional[int] = None, session_path: Optional[str] = None) -> np.ndarray:
        """Computes the theta angle of the head in the 2D xy space"""
        session_data = self._get_session_data(session_path)
        session = session_data["session"]
        assert "position" in session
        assert len(keypoints) == 2
        keypoint_1, keypoint_2 = keypoints
        assert keypoint_1 in session.position.keypoints
        assert keypoint_2 in session.position.keypoints
        diff = session["position"].sel(keypoints=keypoint_1).values - session["position"].sel(keypoints=keypoint_2).values
        if time_slice is not None:
            assert time_slice > 0
            diff = diff[:time_slice]
        assert diff.shape[1] >= 2
        diff_norm = np.linalg.norm(diff, axis=1)
        v_x = diff[:, 0] / diff_norm
        v_y = diff[:, 1] / diff_norm
        theta_head = np.arctan2(v_y, v_x)
        return theta_head
    
    def get_turning_rate(self, keypoints: tuple, time_slice: Optional[int] = None, session_path: Optional[str] = None) -> np.ndarray:
        """Computes the turning rate of the head"""
        theta_head = self.get_theta(keypoints, time_slice, session_path)
        dtheta = np.vstack([np.zeros((1,)), np.diff(theta_head, axis=0)])
        dtheta = np.mod(dtheta + np.pi, 2 * np.pi) - np.pi
        return dtheta
    
    def get_yaw_offset(self, time_slice: Optional[int] = None, session_path: Optional[str] = None) -> np.ndarray:
        """Computes the yaw offset of the head"""
        theta_body = self.get_theta(("back_mid", "tailbase"), time_slice, session_path)
        theta_head = self.get_theta(("nose", "tailbase"), time_slice, session_path)
        delta_theta = theta_head - theta_body
        delta_theta = np.mod(delta_theta + np.pi, 2 * np.pi) - np.pi
        return delta_theta

    def get_pitch_angle(self, keypoints: tuple, time_slice: Optional[int] = None, session_path: Optional[str] = None) -> np.ndarray:
        """Computes the pitch angle of the head"""
        session_data = self._get_session_data(session_path)
        session = session_data["session"]
        assert "position" in session
        assert len(keypoints) == 2
        keypoint_1, keypoint_2 = keypoints
        assert keypoint_1 in session.position.keypoints
        assert keypoint_2 in session.position.keypoints
        assert "z" in session.position.space
        if time_slice is not None:
            assert time_slice > 0
            v_nose = session.position.sel(keypoints=keypoint_1).values[:time_slice]
            v_tailbase = session.position.sel(keypoints=keypoint_2).values[:time_slice]
        else:
            v_nose = session.position.sel(keypoints=keypoint_1).values
            v_tailbase = session.position.sel(keypoints=keypoint_2).values
        v_body = v_nose - v_tailbase
        assert v_body.shape[1] >= 3
        L_xy = np.linalg.norm(v_body[:, :2], axis=1)
        pitch_angle = np.arctan2(v_body[:, 2], L_xy)
        return pitch_angle
    
    def get_velocity_components(self, time_slice: Optional[int] = None, session_path: Optional[str] = None) -> tuple:
        """Computes the forward, sideways, and vertical velocity components"""
        session_data = self._get_session_data(session_path)
        session = session_data["session"]
        assert "position" in session
        assert "nose" in session.position.keypoints
        assert "tailbase" in session.position.keypoints
        if time_slice is not None:
            assert time_slice > 0
            centroids = session.position.isel(time=slice(0, time_slice)).values.mean(axis=2).squeeze()
            nose = session.position.sel(keypoints="nose").isel(time=slice(0, time_slice)).values
            tail = session.position.sel(keypoints="tailbase").isel(time=slice(0, time_slice)).values
        else:
            centroids = session.position.values.mean(axis=2).squeeze()
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
    
    def get_manipulation_index_paws(self, time_slice: Optional[int] = None, session_path: Optional[str] = None) -> tuple:
        """Computes the manipulation index of the paws"""
        session_data = self._get_session_data(session_path)
        session = session_data["session"]
        assert "position" in session
        assert "forepaw_lf" in session.position.keypoints
        assert "forepaw_rt" in session.position.keypoints
        v_lf = session.position.sel(keypoints="forepaw_lf").values.squeeze()
        v_rt = session.position.sel(keypoints="forepaw_rt").values.squeeze()
        if time_slice is not None:
            assert time_slice > 0
            v_lf = v_lf[:time_slice]
            v_rt = v_rt[:time_slice]
        body_speed = self.get_velocity(time_slice, session_path)
        lf_disp = np.vstack([np.zeros((1, 3)), np.diff(v_lf, axis=0)])
        rt_disp = np.vstack([np.zeros((1, 3)), np.diff(v_rt, axis=0)])
        lf_speed = np.linalg.norm(lf_disp, axis=1)
        rt_speed = np.linalg.norm(rt_disp, axis=1)
        # Avoid division by zero
        body_speed_safe = np.where(body_speed > 0, body_speed, 1.0)
        manipulation_idx_lf = lf_speed / body_speed_safe
        manipulation_idx_rt = rt_speed / body_speed_safe
        return manipulation_idx_lf, manipulation_idx_rt
    
    def get_freezing(self, time_slice: Optional[int] = None, session_path: Optional[str] = None) -> np.ndarray:
        """Computes the freezing of the animal"""
        velocity = self.get_velocity(time_slice, session_path)
        threshold = np.percentile(velocity, 10)
        freezing = velocity < threshold
        return freezing
    
    def get_curvature(self, time_slice: Optional[int] = None, session_path: Optional[str] = None) -> np.ndarray:
        """Computes the curvature of the path"""
        turning_rate = self.get_turning_rate(("nose", "tailbase"), time_slice, session_path)
        velocity = self.get_velocity(time_slice, session_path)
        assert len(turning_rate) == len(velocity)
        turning_rate = np.abs(turning_rate.squeeze())
        # Avoid division by zero
        velocity_safe = np.where(velocity > 0, velocity, 1.0)
        curvature = turning_rate / velocity_safe + np.random.randn(len(velocity)) * 0.01
        return curvature
    
    def get_position_centroid(self, time_slice: Optional[int] = None, session_path: Optional[str] = None) -> np.ndarray:
        """Computes the centroid of the position"""
        session_data = self._get_session_data(session_path)
        session = session_data["session"]
        assert "position" in session
        if time_slice is not None:
            assert time_slice > 0
            centroid = session.position.isel(time=slice(0, time_slice)).values.mean(axis=2).squeeze()
        else:
            centroid = session.position.values.mean(axis=2).squeeze()
        return centroid

    def get_distance_to_walls(self, time_slice: Optional[int] = None, session_path: Optional[str] = None) -> np.ndarray:
        """Computes the distance to the walls"""
        session_data = self._get_session_data(session_path)
        arena_3d = session_data["arena_3d"]
        assert "position" in arena_3d
        assert "x" in arena_3d.position.space and "y" in arena_3d.position.space
        centroid_position = self.get_position_centroid(time_slice=time_slice, session_path=session_path)
        arena_2d = arena_3d.position.sel(space=["x", "y"]).values.squeeze()
        x_min, x_max = arena_2d[:, 0].min(), arena_2d[:, 0].max()
        y_min, y_max = arena_2d[:, 1].min(), arena_2d[:, 1].max()
        
        centroid_2d = centroid_position[:, :2]
        
        dist_to_x_min = np.abs(x_min - centroid_2d[:, 0])
        dist_to_x_max = np.abs(x_max - centroid_2d[:, 0])
        dist_to_y_min = np.abs(y_min - centroid_2d[:, 1])
        dist_to_y_max = np.abs(y_max - centroid_2d[:, 1])
        
        d_wall = np.minimum.reduce([dist_to_x_min, dist_to_x_max, dist_to_y_min, dist_to_y_max])
        return d_wall

    def get_distance_mouse_cricket(self, time_slice: Optional[int] = None, session_path: Optional[str] = None) -> np.ndarray:
        """Computes the distance between the mouse and the cricket"""
        session_data = self._get_session_data(session_path)
        cricket_coords_3d = session_data["cricket_coords_3d"]
        if cricket_coords_3d is None:
            raise ValueError("Cricket coordinates not available for this session (object session?)")
        mouse_centroid = self.get_position_centroid(time_slice, session_path)
        min_len = min(len(mouse_centroid), len(cricket_coords_3d))
        mouse_centroid = mouse_centroid[:min_len]
        cricket_coords = cricket_coords_3d[:min_len]
        
        assert mouse_centroid.shape[1] >= 2
        assert cricket_coords.shape[1] == 2
        assert len(mouse_centroid) == len(cricket_coords)
        
        if mouse_centroid.ndim == 3:
            mouse_centroid = np.nanmean(mouse_centroid[:, :, :2], axis=1)
        else:
            mouse_centroid = mouse_centroid[:, :2]
        
        return np.linalg.norm(mouse_centroid - cricket_coords, axis=1)
    
    def extract_all_features_to_dataframe(self, time_slice: Optional[int] = None, session_path: Optional[str] = None) -> pd.DataFrame:
        """Extract all features for a session and return as DataFrame"""
        features = {}
        
        # Basic features
        features['velocity'] = self.get_velocity(time_slice, session_path)
        features['acceleration'] = self.get_accelleration(time_slice, session_path)
        
        # Head rear (might be multi-dimensional)
        head_rear = self.get_head_rear(time_slice, session_path)
        if head_rear.ndim == 2:
            for i, kp in enumerate(['nose_z', 'ear_lf_z', 'ear_rt_z']):
                if i < head_rear.shape[1]:
                    features[f'head_rear_{kp}'] = head_rear[:, i]
        else:
            features['head_rear'] = head_rear
        
        # Theta and related
        keypoints = ('nose', 'tailbase')
        features['theta'] = self.get_theta(keypoints, time_slice, session_path)
        features['turning_rate'] = self.get_turning_rate(keypoints, time_slice, session_path)
        features['yaw_offset'] = self.get_yaw_offset(time_slice, session_path)
        features['pitch_angle'] = self.get_pitch_angle(keypoints, time_slice, session_path)
        
        # Velocity components (returns tuple)
        forward_vel, side_vel, vertical_vel = self.get_velocity_components(time_slice, session_path)
        features['velocity_forward'] = forward_vel
        features['velocity_side'] = side_vel
        features['velocity_vertical'] = vertical_vel
        
        # Manipulation index (returns tuple of two arrays)
        manip_idx_lf, manip_idx_rt = self.get_manipulation_index_paws(time_slice, session_path)
        features['manipulation_index_lf'] = manip_idx_lf
        features['manipulation_index_rt'] = manip_idx_rt
        
        # Freezing
        features['freezing'] = self.get_freezing(time_slice, session_path)
        
        # Curvature
        features['curvature'] = self.get_curvature(time_slice, session_path)
        
        # Position centroid (3D)
        centroid = self.get_position_centroid(time_slice, session_path)
        centroid = centroid.squeeze()  # Remove any singleton dimensions
        if centroid.ndim == 2:
            features['centroid_x'] = centroid[:, 0]
            features['centroid_y'] = centroid[:, 1]
            if centroid.shape[1] > 2:
                features['centroid_z'] = centroid[:, 2]
        elif centroid.ndim == 1:
            # If 1D, it's likely just x, or we need to handle it differently
            features['centroid'] = centroid
        
        # Distance to walls (4D)
        dist_walls = self.get_distance_to_walls(time_slice, session_path)
        dist_walls = dist_walls.squeeze()  # Remove any singleton dimensions
        if dist_walls.ndim == 2:
            features['dist_wall_x_min'] = dist_walls[:, 0]
            features['dist_wall_x_max'] = dist_walls[:, 1]
            features['dist_wall_y_min'] = dist_walls[:, 2]
            features['dist_wall_y_max'] = dist_walls[:, 3]
        elif dist_walls.ndim == 1:
            features['dist_wall'] = dist_walls
        
        # Distance mouse-cricket (only for cricket sessions)
        try:
            features['dist_mouse_cricket'] = self.get_distance_mouse_cricket(time_slice, session_path)
        except (ValueError, KeyError):
            pass  # Skip if not available (object sessions)
        
        # Find minimum length to align all features and ensure all are 1D
        min_len = min(len(v.flatten()) if isinstance(v, np.ndarray) else len(v) 
                     for v in features.values() if isinstance(v, np.ndarray))
        
        aligned_features = {}
        for k, v in features.items():
            if isinstance(v, np.ndarray):
                # Flatten to 1D and trim to min_len
                v_flat = v.flatten()
                aligned_features[k] = v_flat[:min_len]
            else:
                aligned_features[k] = v
        
        return pd.DataFrame(aligned_features)


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract features from tracking data")
    parser.add_argument("directory", type=str, help="Base directory path (e.g., M30 or M30/20250507/cricket/115438)")
    parser.add_argument("--time-slice", type=int, default=None, help="Time slice to process (default: all data)")
    parser.add_argument("--output", type=str, default=None, help="Output path. If ends with .csv, save each session as separate CSV. Otherwise, save single file with all sessions.")
    args = parser.parse_args()
    
    extractor = FeatureExtractor(args.directory)
    
    if args.output:
        output_path = Path(args.output)
        
        if output_path.suffix == '.csv':
            # Save each session as separate CSV file
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = output_path.stem
            
            for session_path in extractor.sessions.keys():
                df = extractor.extract_all_features_to_dataframe(args.time_slice, session_path)
                session_name = Path(session_path).name
                csv_path = output_dir / f"{base_name}_{session_name}.csv"
                df.to_csv(csv_path, index=False)
                print(f"Saved features for {session_name} to {csv_path}")
        else:
            # Check if path is a directory (existing or should be created)
            if output_path.exists() and output_path.is_dir():
                # If it's a directory, save as pickle file inside it
                output_file = output_path / "features_all_sessions.pkl"
            elif not output_path.suffix:
                # If no extension, treat as directory and create pickle inside
                output_path.mkdir(parents=True, exist_ok=True)
                output_file = output_path / "features_all_sessions.pkl"
            else:
                # It's a file path
                output_file = output_path
            
            # Save single file with all sessions
            all_sessions_data = {}
            for session_path in extractor.sessions.keys():
                df = extractor.extract_all_features_to_dataframe(args.time_slice, session_path)
                session_name = Path(session_path).name
                all_sessions_data[session_name] = df
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if output_file.suffix == '.pkl' or output_file.suffix == '.pickle':
                with open(output_file, 'wb') as f:
                    pickle.dump(all_sessions_data, f)
                print(f"Saved all sessions to {output_file} (pickle format)")
            else:
                # Default: save as pickle if unknown extension
                output_file = output_file.with_suffix('.pkl')
                with open(output_file, 'wb') as f:
                    pickle.dump(all_sessions_data, f)
                print(f"Saved all sessions to {output_file} (pickle format)")
    
    if len(extractor.sessions) > 1:
        print(f"\nLoaded {len(extractor.sessions)} sessions")
        for session_path in extractor.sessions.keys():
            print(f"\nSession: {session_path}")
            print(f"  Velocity: {extractor.get_velocity(args.time_slice, session_path).shape}")
            print(f"  Acceleration: {extractor.get_accelleration(args.time_slice, session_path).shape}")
    else:
        session_path = list(extractor.sessions.keys())[0] if extractor.sessions else None
        print(f"Velocity: {extractor.get_velocity(args.time_slice, session_path).shape}")
        print(f"Acceleration: {extractor.get_accelleration(args.time_slice, session_path).shape}")
        print(f"Head rear: {extractor.get_head_rear(args.time_slice, session_path).shape}")
        print(f"Theta: {extractor.get_theta(('nose', 'tailbase'), args.time_slice, session_path).shape}")
        print(f"Turning rate: {extractor.get_turning_rate(('nose', 'tailbase'), args.time_slice, session_path).shape}")
        print(f"Yaw offset: {extractor.get_yaw_offset(args.time_slice, session_path).shape}")
        print(f"Pitch angle: {extractor.get_pitch_angle(('nose', 'tailbase'), args.time_slice, session_path).shape}")
        print(f"Velocity components: {extractor.get_velocity_components(args.time_slice, session_path)[0].shape}")
        print(f"Manipulation index: {extractor.get_manipulation_index_paws(args.time_slice, session_path)[0].shape}")
        print(f"Freezing: {extractor.get_freezing(args.time_slice, session_path).shape}")
        print(f"Curvature: {extractor.get_curvature(args.time_slice, session_path).shape}")
        print(f"Position centroid: {extractor.get_position_centroid(args.time_slice, session_path).shape}")
        print(f"Distance to walls: {extractor.get_distance_to_walls(args.time_slice, session_path).shape}")
        try:
            print(f"Distance mouse-cricket: {extractor.get_distance_mouse_cricket(args.time_slice, session_path).shape}")
        except (ValueError, KeyError):
            print("Distance mouse-cricket: Not available (object session)")
    