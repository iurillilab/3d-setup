from dataclasses import dataclass

@dataclass
class DetectionOptions:
    board_shape: tuple[int, int] = (5, 7)
    match_score_min_diff: float = 0.2
    match_score_min: float = 0.7


@dataclass
class DetectionRunnerOptions:
    video_extension: str = "mp4"
    overwrite: bool = False


