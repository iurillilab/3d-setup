from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import xarray as xr
from tqdm import tqdm


class SessionLoader:
    """Handle discovery and loading of tracking sessions."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        assert self.base_dir.exists(), f"Base directory does not exist: {self.base_dir}"

        self.sessions: Dict[str, dict] = {}
        self.arena_3d: Optional[xr.Dataset] = None
        self.arena_views: Optional[xr.Dataset] = None
        self.session_dir: Optional[Path] = None

    def load(self) -> Dict[str, dict]:
        if self._is_multi_session_dir():
            self._load_all_sessions()
        else:
            self._load_single_session()
        return self.sessions

    # --------------------------------------------------------------------- #
    # Discovery
    # --------------------------------------------------------------------- #

    def _is_multi_session_dir(self) -> bool:
        date_dirs = [
            d
            for d in self.base_dir.iterdir()
            if d.is_dir() and d.name.isdigit() and len(d.name) == 8
        ]
        return len(date_dirs) > 0

    def _find_all_session_dirs(self) -> List[Path]:
        pattern = "**/multicam_video*cropped*"
        all_dirs = [
            d
            for d in self.base_dir.glob(pattern)
            if d.is_dir() and "multicam_video" in d.name and "cropped" in d.name
        ]

        v2_dirs = [d for d in all_dirs if "-v2" in d.name]
        non_v2_dirs = [d for d in all_dirs if "-v2" not in d.name]

        session_dirs: List[Path] = []
        seen_parents = set()

        for d in v2_dirs:
            parent_dir = d.parent
            if parent_dir in seen_parents:
                continue
            v2_candidates = self._iter_session_dir_variants(parent_dir, include_v2=True)
            if v2_candidates:
                session_dirs.append(max(v2_candidates, key=lambda p: p.stat().st_mtime))
                seen_parents.add(parent_dir)

        for d in non_v2_dirs:
            parent_dir = d.parent
            if parent_dir in seen_parents:
                continue
            candidates = self._iter_session_dir_variants(parent_dir, include_v2=False)
            if candidates:
                session_dirs.append(max(candidates, key=lambda p: p.stat().st_mtime))
                seen_parents.add(parent_dir)

        return session_dirs

    @staticmethod
    def _iter_session_dir_variants(parent_dir: Path, include_v2: bool) -> List[Path]:
        variants = []
        for sd in parent_dir.iterdir():
            if not sd.is_dir():
                continue
            name = sd.name
            if "multicam_video" not in name or "cropped" not in name:
                continue
            if include_v2 and "-v2" not in name:
                continue
            if not include_v2 and "-v2" in name:
                continue
            variants.append(sd)
        return variants

    def _find_latest_session(self) -> Path:
        candidates = [
            d
            for d in self.base_dir.iterdir()
            if d.is_dir() and "multicam_video" in d.name and "cropped" in d.name
        ]
        if not candidates:
            raise FileNotFoundError(f"No session directories found in {self.base_dir}")

        v2_candidates = [d for d in candidates if "-v2" in d.name]
        if v2_candidates:
            return max(v2_candidates, key=lambda p: p.stat().st_mtime)
        return max(candidates, key=lambda p: p.stat().st_mtime)

    # --------------------------------------------------------------------- #
    # Loading helpers
    # --------------------------------------------------------------------- #

    def _ensure_arena_datasets(self) -> Tuple[xr.Dataset, xr.Dataset]:
        if self.arena_3d is not None and self.arena_views is not None:
            return self.arena_3d, self.arena_views

        script_dir = Path(__file__).parent.parent
        arena_3d_path = script_dir / "data" / "newarena.h5"
        if not arena_3d_path.exists():
            arena_3d_path = script_dir / "tests" / "assets" / "newarena.h5"
        arena_views_path = script_dir / "tests" / "assets" / "arena_views.h5"

        assert arena_3d_path.exists(), f"Arena 3D file not found: {arena_3d_path}"
        assert arena_views_path.exists(), (
            f"Arena views file not found: {arena_views_path}"
        )

        self.arena_3d = xr.open_dataset(arena_3d_path)
        self.arena_views = xr.open_dataset(arena_views_path)
        return self.arena_3d, self.arena_views

    def _load_all_sessions(self) -> None:
        arena_3d, arena_views = self._ensure_arena_datasets()
        session_dirs = self._find_all_session_dirs()
        if not session_dirs:
            raise FileNotFoundError(f"No session directories found in {self.base_dir}")

        print(f"Found {len(session_dirs)} session directories to process")
        loaded = 0
        failed: List[Tuple[Path, str]] = []

        for session_dir in tqdm(session_dirs, desc="Loading sessions"):
            try:
                session_data = self._load_session_data(session_dir, arena_3d, arena_views)
            except Exception as exc:  # pragma: no cover - guarding multi-run feedback
                failed.append((session_dir, str(exc)))
                continue

            self.sessions[str(session_dir)] = session_data
            loaded += 1

        print(f"\nSuccessfully loaded {loaded}/{len(session_dirs)} sessions")
        if failed:
            print(
                f"\nFailed to load {len(failed)} sessions (missing triangulated h5 files):"
            )
            for session_dir, error in failed[:5]:
                print(f"  - {session_dir.parent.name}/{session_dir.name}")
                print(f"    {error}")
            if len(failed) > 5:
                remaining = len(failed) - 5
                print(f"  ... and {remaining} more")

    def _load_single_session(self) -> None:
        arena_3d, arena_views = self._ensure_arena_datasets()
        self.session_dir = self._find_latest_session()
        session_data = self._load_session_data(self.session_dir, arena_3d, arena_views)
        self.sessions[str(self.session_dir)] = session_data

    def _load_session_data(
        self, session_dir: Path, arena_3d: xr.Dataset, arena_views: xr.Dataset
    ) -> dict:
        triangulated_h5 = self._find_triangulated_h5_in_dir(session_dir)
        session = xr.open_dataset(triangulated_h5)

        result = {
            "session_dir": session_dir,
            "session": session,
            "arena_3d": arena_3d,
            "arena_views": arena_views,
            "mouse_coords_2d": None,
            "cricket_coords_2d": None,
            "cricket_coords_3d": None,
            "prey_label": None,
        }

        try:
            mouse_pickle = self._find_latest_mouse_pickle_in_dir(session_dir)
            mouse_coords_2d, mouse_bodyparts = self._load_2d_coordinates(mouse_pickle)
            result["mouse_coords_2d"] = mouse_coords_2d
            result["mouse_coords_2d_bodyparts"] = mouse_bodyparts
        except FileNotFoundError as exc:
            print(
                f"Warning: Could not find mouse central 2D pickle for {session_dir}: {exc}"
            )
        except Exception as exc:  # pragma: no cover - emphasizing robustness
            print(
                f"Warning: Failed to load mouse 2D coordinates for {session_dir}: {exc}"
            )

        if self._is_cricket_session(session_dir):
            try:
                prey_pickle, prey_label = self._find_latest_prey_pickle_in_dir(
                    session_dir
                )
                cricket_coords_2d, cricket_bodyparts = self._load_2d_coordinates(
                    prey_pickle
                )
                with np.errstate(invalid="ignore"):
                    cricket_coords_2d_mean = np.nanmean(cricket_coords_2d, axis=1)
                cricket_coords_3d = self._convert_cricket_coordinates(
                    cricket_coords_2d_mean, arena_3d, arena_views
                )
                result["cricket_coords_2d"] = cricket_coords_2d
                result["cricket_coords_2d_bodyparts"] = cricket_bodyparts
                result["cricket_coords_3d"] = cricket_coords_3d
                result["prey_label"] = prey_label
            except Exception as exc:
                print(
                    f"Warning: Could not load cricket coordinates for {session_dir}: {exc}"
                )

        return result

    # --------------------------------------------------------------------- #
    # Pickle and dataset helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _find_latest_pickle_in_dir_with_keywords(
        session_dir: Path,
        include_keywords: Iterable[str],
        exclude_keywords: Optional[Iterable[str]] = None,
    ) -> Path:
        patterns = ["*.pickle", "*.pkl"]
        candidates: List[Path] = []
        for pattern in patterns:
            candidates.extend(session_dir.rglob(pattern))

        include_lower = [kw.lower() for kw in include_keywords]
        exclude_lower = [kw.lower() for kw in (exclude_keywords or [])]

        matches = []
        for candidate in candidates:
            name_lower = candidate.name.lower()
            if all(keyword in name_lower for keyword in include_lower):
                if all(keyword not in name_lower for keyword in exclude_lower):
                    matches.append(candidate)

        if not matches:
            raise FileNotFoundError(
                f"No pickle files found in {session_dir} containing keywords {include_lower}"
            )
        return max(matches, key=lambda p: p.stat().st_mtime)

    @classmethod
    def _find_latest_mouse_pickle_in_dir(cls, session_dir: Path) -> Path:
        search_orders = [
            ["centraldlc", "mouse"],
            ["central", "mouse"],
            ["mouse"],
        ]
        for keywords in search_orders:
            try:
                return cls._find_latest_pickle_in_dir_with_keywords(session_dir, keywords)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"No mouse pickle files found in {session_dir}")

    @classmethod
    def _find_latest_prey_pickle_in_dir(cls, session_dir: Path) -> Tuple[Path, str]:
        search_orders = [
            (["centraldlc", "cricket"], "cricket"),
            (["central", "cricket"], "cricket"),
            (["cricket"], "cricket"),
            (["centraldlc", "object"], "object"),
            (["central", "object"], "object"),
            (["object"], "object"),
        ]
        for keywords, label in search_orders:
            try:
                path = cls._find_latest_pickle_in_dir_with_keywords(
                    session_dir, keywords
                )
                return path, label
            except FileNotFoundError:
                continue
        raise FileNotFoundError(
            f"No cricket/object pickle files found in {session_dir}"
        )

    @staticmethod
    def _find_triangulated_h5_in_dir(session_dir: Path) -> Path:
        candidates = list(session_dir.rglob("*triangulated*.h5"))
        if not candidates:
            raise FileNotFoundError(f"No triangulated h5 files found in {session_dir}")
        return max(candidates, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _is_cricket_session(session_dir: Path) -> bool:
        session_str = str(session_dir).lower()
        return "cricket" in session_str or "object" in session_str

    @staticmethod
    def _load_2d_coordinates(
        pickle_path: Path,
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        data = pickle.load(open(pickle_path, "rb"))
        bodyparts: Optional[List[str]] = None
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            bodyparts_value = metadata.get("bodyparts")
            if isinstance(bodyparts_value, (list, tuple)):
                bodyparts = list(bodyparts_value)

        coordinates = []
        for frame in data.keys():
            if frame == "metadata":
                continue
            frame_coords = data[frame]["coordinates"]
            arr = SessionLoader._normalize_coordinate_array(frame_coords)
            coordinates.append(arr)

        max_n = max(coord.shape[0] for coord in coordinates)
        coords_padded = []
        for arr in coordinates:
            if arr.shape[0] < max_n:
                pad_width = ((0, max_n - arr.shape[0]), (0, 0))
                arr = np.pad(arr, pad_width, mode="constant", constant_values=np.nan)
            coords_padded.append(arr)
        return np.array(coords_padded), bodyparts

    @staticmethod
    def _normalize_coordinate_array(frame_coords) -> np.ndarray:
        arr = np.array(frame_coords)
        if arr.ndim == 3:
            return arr[:, 0, :]
        if arr.ndim == 2:
            return arr
        return arr.reshape(-1, 2)

    @staticmethod
    def _convert_cricket_coordinates(
        cricket_coordinates: np.ndarray,
        arena_3d: xr.Dataset,
        arena_views: xr.Dataset,
    ) -> np.ndarray:
        assert "position" in arena_3d, "arena_3d must contain 'position' variable"
        assert "position" in arena_views, "arena_views must contain 'position' variable"
        assert "x" in arena_3d.position.space and "y" in arena_3d.position.space
        assert "central" in arena_views.position.view
        assert cricket_coordinates.shape[1] == 2, (
            "cricket_coordinates must have shape (N, 2)"
        )

        arena_corners_3d = (
            arena_3d["position"].sel(space=["x", "y"]).values.squeeze()[:, :4]
        )
        arena_corners_2d = (
            arena_views["position"].sel(view="central").values.squeeze()[:, :4]
        )

        arena_3d_homogeneous = np.vstack([arena_corners_3d, np.ones((1, 4))])
        arena_2d_homogeneous = np.vstack([arena_corners_2d, np.ones((1, 4))])

        H = arena_3d_homogeneous @ np.linalg.pinv(arena_2d_homogeneous)

        cricket_2d = cricket_coordinates.squeeze()
        cricket_2d_homogeneous = np.vstack(
            [cricket_2d.T, np.ones((1, cricket_2d.shape[0]))]
        )

        cricket_3d_homogeneous = H @ cricket_2d_homogeneous
        cricket_3d_homogeneous = cricket_3d_homogeneous / cricket_3d_homogeneous[2, :]
        return cricket_3d_homogeneous[:2, :].T


