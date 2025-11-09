import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import feature_functions as feat
from .session_loader import SessionLoader


class FeatureExtractor:
    def __init__(self, base_dir: Path):
        """
        Initialize FeatureExtractor with base directory.

        Args:
            base_dir: Directory like M30/20250507/cricket/115438 or M30 (for multi-session)
        """
        self.base_dir = Path(base_dir)
        self.loader = SessionLoader(self.base_dir)
        self.sessions = self.loader.load()
        self.arena_3d = self.loader.arena_3d
        self.arena_views = self.loader.arena_views

        self.session_dir = self.loader.session_dir
        self.session = None
        self.cricket_coords_2d = None
        self.cricket_coords_3d = None
        self.mouse_coords_2d = None
        self.prey_label = None

        if len(self.sessions) == 1:
            session_path, session_data = next(iter(self.sessions.items()))
            self.session_dir = Path(session_path)
            self.session = session_data["session"]
            self.cricket_coords_2d = session_data.get("cricket_coords_2d")
            self.cricket_coords_3d = session_data.get("cricket_coords_3d")
            self.mouse_coords_2d = session_data.get("mouse_coords_2d")
            self.prey_label = session_data.get("prey_label")

    # Session loading helpers are provided by SessionLoader.










    def _get_session_data(self, session_path: Optional[str] = None):
        """Get session data for a specific session path, or current session if single mode"""
        if session_path:
            if session_path not in self.sessions:
                raise KeyError(f"Session path not found: {session_path}")
            return self.sessions[session_path]
        if hasattr(self, "session") and self.session is not None:
            return {
                "session": self.session,
                "arena_3d": self.arena_3d,
                "arena_views": self.arena_views,
                "mouse_coords_2d": getattr(self, "mouse_coords_2d", None),
                "cricket_coords_2d": getattr(self, "cricket_coords_2d", None),
                "cricket_coords_3d": getattr(self, "cricket_coords_3d", None),
                "prey_label": getattr(self, "prey_label", None),
            }
        if len(self.sessions) == 1:
            return list(self.sessions.values())[0]
        raise ValueError("Must specify session_path when multiple sessions are loaded")

    def get_velocity(
        self, time_slice: Optional[int] = None, session_path: Optional[str] = None
    ) -> np.ndarray:
        """Calculate velocity from position data"""
        session_data = self._get_session_data(session_path)
        return feat.compute_velocity(session_data, time_slice)

    def get_accelleration(
        self, time_slice: Optional[int] = None, session_path: Optional[str] = None
    ) -> np.ndarray:
        """Calculate acceleration from position data"""
        session_data = self._get_session_data(session_path)
        return feat.compute_acceleration(session_data, time_slice)

    def get_head_rear(
        self, time_slice: Optional[int] = None, session_path: Optional[str] = None
    ) -> np.ndarray:
        """Computes the vertical movement of the head as heuristic of rear"""
        session_data = self._get_session_data(session_path)
        return feat.compute_head_rear(session_data, time_slice)

    def get_theta(
        self,
        keypoints: tuple,
        time_slice: Optional[int] = None,
        session_path: Optional[str] = None,
    ) -> np.ndarray:
        """Computes the theta angle of the head in the 2D xy space"""
        session_data = self._get_session_data(session_path)
        return feat.compute_theta(session_data, keypoints, time_slice)

    def get_turning_rate(
        self,
        keypoints: tuple,
        time_slice: Optional[int] = None,
        session_path: Optional[str] = None,
    ) -> np.ndarray:
        """Computes the turning rate of the head"""
        session_data = self._get_session_data(session_path)
        return feat.compute_turning_rate(session_data, keypoints, time_slice)

    def get_yaw_offset(
        self, time_slice: Optional[int] = None, session_path: Optional[str] = None
    ) -> np.ndarray:
        """Computes the yaw offset of the head"""
        session_data = self._get_session_data(session_path)
        return feat.compute_yaw_offset(session_data, time_slice)

    def get_pitch_angle(
        self,
        keypoints: tuple,
        time_slice: Optional[int] = None,
        session_path: Optional[str] = None,
    ) -> np.ndarray:
        """Computes the pitch angle of the head"""
        session_data = self._get_session_data(session_path)
        return feat.compute_pitch_angle(session_data, keypoints, time_slice)

    def get_velocity_components(
        self, time_slice: Optional[int] = None, session_path: Optional[str] = None
    ) -> tuple:
        """Computes the forward, sideways, and vertical velocity components"""
        session_data = self._get_session_data(session_path)
        return feat.compute_velocity_components(session_data, time_slice)

    def get_manipulation_index_paws(
        self,
        time_slice: Optional[int] = None,
        session_path: Optional[str] = None,
        target_distance: np.ndarray = None,
        quantile_manipulation: float = 0.75,
        quantile_distance: float = 0.1,
    ) -> tuple:
        """Computes the manipulation index of the paws"""
        session_data = self._get_session_data(session_path)
        return feat.compute_manipulation_index_paws(
            session_data,
            time_slice,
            target_distance=target_distance,
            quantile_manipulation=quantile_manipulation,
            quantile_distance=quantile_distance,
        )

    def get_freezing(
        self, time_slice: Optional[int] = None, session_path: Optional[str] = None
    ) -> np.ndarray:
        """Computes the freezing of the animal"""
        session_data = self._get_session_data(session_path)
        return feat.compute_freezing(session_data, time_slice)

    def get_curvature(
        self, time_slice: Optional[int] = None, session_path: Optional[str] = None
    ) -> np.ndarray:
        """Computes the curvature of the path"""
        session_data = self._get_session_data(session_path)
        return feat.compute_curvature(session_data, time_slice)

    def get_position_centroid(
        self, time_slice: Optional[int] = None, session_path: Optional[str] = None
    ) -> np.ndarray:
        """Computes the centroid of the position"""
        session_data = self._get_session_data(session_path)
        return feat.compute_position_centroid(session_data, time_slice)

    def get_distance_to_walls(
        self, time_slice: Optional[int] = None, session_path: Optional[str] = None
    ) -> np.ndarray:
        """Computes the distance to the walls"""
        session_data = self._get_session_data(session_path)
        return feat.compute_distance_to_walls(session_data, time_slice)

    def get_distance_mouse_cricket(
        self, time_slice: Optional[int] = None, session_path: Optional[str] = None
    ) -> np.ndarray:
        """Computes the distance between the mouse and the cricket"""
        session_data = self._get_session_data(session_path)
        return feat.compute_distance_mouse_cricket(session_data, time_slice)

    def get_mouse_centroid_2d(
        self, time_slice: Optional[int] = None, session_path: Optional[str] = None
    ) -> np.ndarray:
        """Computes the 2D centroid of the mouse from central view coordinates"""
        session_data = self._get_session_data(session_path)
        return feat.compute_mouse_centroid_2d(session_data, time_slice)

    def get_cricket_centroid_2d(
        self, time_slice: Optional[int] = None, session_path: Optional[str] = None
    ) -> np.ndarray:
        """Computes the 2D centroid of the prey (cricket/object) from central view coordinates"""
        session_data = self._get_session_data(session_path)
        return feat.compute_cricket_centroid_2d(session_data, time_slice)

    def get_distance_mouse_cricket_2d(
        self, time_slice: Optional[int] = None, session_path: Optional[str] = None
    ) -> np.ndarray:
        """Computes the distance between the mouse centroid and prey in central 2D view"""
        session_data = self._get_session_data(session_path)
        return feat.compute_distance_mouse_cricket_2d(session_data, time_slice)

    def extract_all_features_to_dataframe(
        self, time_slice: Optional[int] = None, session_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Extract all features for a session and return as DataFrame"""
        session_data = self._get_session_data(session_path)
        return feat.extract_all_features_to_dataframe(session_data, time_slice)


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract features from tracking data")
    parser.add_argument(
        "directory",
        type=str,
        help="Base directory path (e.g., M30 or M30/20250507/cricket/115438)",
    )
    parser.add_argument(
        "--time-slice",
        type=int,
        default=None,
        help="Time slice to process (default: all data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path. If ends with .csv, save each session as separate CSV. Otherwise, save single file with all sessions.",
    )
    args = parser.parse_args()

    extractor = FeatureExtractor(args.directory)

    if args.output:
        output_path = Path(args.output)

        if output_path.suffix == ".csv":
            # Save each session as separate CSV file
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = output_path.stem

            for session_path in extractor.sessions.keys():
                df = extractor.extract_all_features_to_dataframe(
                    args.time_slice, session_path
                )
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
                df = extractor.extract_all_features_to_dataframe(
                    args.time_slice, session_path
                )
                session_name = Path(session_path).name
                all_sessions_data[session_name] = df

            output_file.parent.mkdir(parents=True, exist_ok=True)

            if output_file.suffix == ".pkl" or output_file.suffix == ".pickle":
                with open(output_file, "wb") as f:
                    pickle.dump(all_sessions_data, f)
                print(f"Saved all sessions to {output_file} (pickle format)")
            else:
                # Default: save as pickle if unknown extension
                output_file = output_file.with_suffix(".pkl")
                with open(output_file, "wb") as f:
                    pickle.dump(all_sessions_data, f)
                print(f"Saved all sessions to {output_file} (pickle format)")

    if len(extractor.sessions) > 1:
        print(f"\nLoaded {len(extractor.sessions)} sessions")
        for session_path in extractor.sessions.keys():
            print(f"\nSession: {session_path}")
            print(
                f"  Velocity: {extractor.get_velocity(args.time_slice, session_path).shape}"
            )
            print(
                f"  Acceleration: {extractor.get_accelleration(args.time_slice, session_path).shape}"
            )
    else:
        session_path = (
            list(extractor.sessions.keys())[0] if extractor.sessions else None
        )
        print(
            f"Velocity: {extractor.get_velocity(args.time_slice, session_path).shape}"
        )
        print(
            f"Acceleration: {extractor.get_accelleration(args.time_slice, session_path).shape}"
        )
        print(
            f"Head rear: {extractor.get_head_rear(args.time_slice, session_path).shape}"
        )
        print(
            f"Theta: {extractor.get_theta(('nose', 'tailbase'), args.time_slice, session_path).shape}"
        )
        print(
            f"Turning rate: {extractor.get_turning_rate(('nose', 'tailbase'), args.time_slice, session_path).shape}"
        )
        print(
            f"Yaw offset: {extractor.get_yaw_offset(args.time_slice, session_path).shape}"
        )
        print(
            f"Pitch angle: {extractor.get_pitch_angle(('nose', 'tailbase'), args.time_slice, session_path).shape}"
        )
        print(
            f"Velocity components: {extractor.get_velocity_components(args.time_slice, session_path)[0].shape}"
        )
        print(
            f"Manipulation index: {extractor.get_manipulation_index_paws(args.time_slice, session_path).shape}"
        )
        print(
            f"Freezing: {extractor.get_freezing(args.time_slice, session_path).shape}"
        )
        print(
            f"Curvature: {extractor.get_curvature(args.time_slice, session_path).shape}"
        )
        print(
            f"Position centroid: {extractor.get_position_centroid(args.time_slice, session_path).shape}"
        )
        print(
            f"Distance to walls: {extractor.get_distance_to_walls(args.time_slice, session_path).shape}"
        )
        try:
            print(
                f"Distance mouse-cricket: {extractor.get_distance_mouse_cricket(args.time_slice, session_path).shape}"
            )
        except (ValueError, KeyError):
            print("Distance mouse-cricket: Not available (object session)")
