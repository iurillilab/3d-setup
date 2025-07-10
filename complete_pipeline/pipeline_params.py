from calendar import c
from dataclasses import dataclass


@dataclass
class CroppingOptions:
    crop_folder_pattern: str = "cropped-v2"
    expected_views: tuple[str] = ("mirror-left", "mirror-right", "mirror-top", "mirror-bottom", "central")

@dataclass
class ProcessingOptions:
    n_workers: int = 6

@dataclass
class DetectionOptions:
    board_shape: tuple[int, int] = (5, 7)
    match_score_min_diff: float = 0.2
    match_score_min: float = 0.4


@dataclass
class DetectionRunnerOptions:
    video_extension: str = "mp4"
    overwrite: bool = False


@dataclass
class CalibrationOptions:
    square_size: float = 12.5
    scale_factor: float = 0.5
    n_samples_for_intrinsics: int = 100
    ftol: float = 1e-4
    overlay: bool = False
    n_frames: int | None = None