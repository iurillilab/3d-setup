import argparse
from pathlib import Path
from typing import List
from dataclasses import dataclass, asdict

from threed_utils.multiview_calibration.detection import (
    detect_chessboard,
    # process_video,
    run_calibration_detection,
)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

@dataclass
class DetectionOptions:
    board_shape: tuple[int, int] = (5, 7)
    match_score_min_diff: float = 0.2
    match_score_min: float = 0.7
    video_extension: str = "mp4"


DEFAULT_N_WORKERS = 12

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def run_checkerboard_detection(
    folder: Path,
    detection_options: dict = None,
    n_workers: int = DEFAULT_N_WORKERS,
) -> None:
    """
    Run checkerboard detection on all video files in *folder*.
    """

    options_dict = asdict(detection_options)
    extension = options_dict.pop("video_extension")

    video_files = list(folder.rglob(f"*.{extension}"))
    
    if not video_files:
        raise ValueError(f"No video files found in {folder}")
    
    print(f"Found {len(video_files)} video files in {folder}")
    
    all_videos = [str(video_file) for video_file in video_files]
    
    run_calibration_detection(
        all_videos,
        detect_chessboard,
        detection_options=options_dict,
        n_workers=n_workers,
    )

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract checkerboards from video files.")
    parser.add_argument(
        "folder", 
        type=Path, 
        help="Folder containing video files for checkerboard detection."
    )
    parser.add_argument(
        "--board-shape", 
        type=int, 
        nargs=2, 
        default=DetectionOptions.board_shape,
        help=f"Checkerboard shape (rows, cols). Default: {DetectionOptions.board_shape}"
    )
    parser.add_argument(
        "--match-score-min", 
        type=float, 
        default=DetectionOptions.match_score_min,
        help=f"Minimum match score. Default: {DetectionOptions.match_score_min}"
    )
    parser.add_argument(
        "--match-score-min-diff", 
        type=float, 
        default=DetectionOptions.match_score_min_diff,
        help=f"Minimum match score difference. Default: {DetectionOptions.match_score_min_diff}"
    )
    parser.add_argument(
        "--n-workers", 
        type=int, 
        default=DEFAULT_N_WORKERS,
        help=f"Number of worker processes. Default: {DEFAULT_N_WORKERS}"
    )
    parser.add_argument(
        "--video-extension",
        type=str,
        default=DetectionOptions.video_extension,
        help=f"Video file extension. Default: {DetectionOptions.video_extension}"
    )

    args = parser.parse_args()

    if not args.folder.exists():
        raise FileNotFoundError(f"Folder not found: {args.folder}")

    detection_options = DetectionOptions(
        board_shape=tuple(args.board_shape),
        match_score_min=args.match_score_min,
        match_score_min_diff=args.match_score_min_diff,
        video_extension=args.video_extension,
    )

    run_checkerboard_detection(
        folder=args.folder,
        detection_options=detection_options,
        n_workers=args.n_workers,
    )
