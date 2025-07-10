import argparse
from pathlib import Path
from typing import List
from dataclasses import dataclass, asdict
from pipeline_params import DetectionOptions, DetectionRunnerOptions

from threed_utils.multiview_calibration.detection import run_checkerboard_detection


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
        "--video-extension",
        type=str,
        default=DetectionRunnerOptions.video_extension,
        help=f"Video file extension. Default: {DetectionRunnerOptions.video_extension}"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=DetectionRunnerOptions.overwrite,
        help="Overwrite existing detections."
    )

    args = parser.parse_args()

    if not args.folder.exists():
        raise FileNotFoundError(f"Folder not found: {args.folder}")

    detection_options = DetectionOptions(
        board_shape=tuple(args.board_shape),
        match_score_min=args.match_score_min,
        match_score_min_diff=args.match_score_min_diff,
    )

    run_checkerboard_detection(
        folder=args.folder,
        detection_options=asdict(detection_options),
        extension=args.video_extension,
        overwrite=args.overwrite,
    )
