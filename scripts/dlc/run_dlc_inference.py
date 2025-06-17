#!/usr/bin/env python3
"""
Simple DeepLabCut inference helper
----------------------------------

Usage (examples):

    # Run inference on all .mp4 files inside a folder, create labelled videos too
    python run_dlc_inference.py config.yaml /path/to/videos --make-labeled-video \
           --shuffle-n 2 --batch-size 4

    # Same, but skip labelled‐video generation
    python run_dlc_inference.py config.yaml /path/to/videos \
           --shuffle-n 1 --batch-size 8

Positional arguments
~~~~~~~~~~~~~~~~~~~~
* ``config_file`` – path to the DLC ``config.yaml``
* ``videos_dir``  – directory containing the videos

Optional arguments
~~~~~~~~~~~~~~~~~~
* ``--make-labeled-video`` – create labelled videos (off by default)
* ``--shuffle-n`` – which shuffle index to use (default 2)
* ``--batch-size`` – batch size for inference (default 2)

The script processes *all* ``*.mp4`` files found in ``videos_dir``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run DeepLabCut inference on a folder of videos",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# positional
parser.add_argument("config_file", type=str, help="Path to DLC config.yaml file")
parser.add_argument("videos_dir", type=str, help="Directory containing videos")

# optional
parser.add_argument(
    "--make-labeled-video",
    dest="make_labeled_video",
    action="store_true",
    help="Create labeled video output",
)
parser.add_argument(
    "--shuffle_n",
    type=int,
    default=2,
    help="Shuffle index to use (integer)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Batch size used by DLC during inference",
)

args = parser.parse_args()

# -----------------------------------------------------------------------------
# Path handling & video list
# -----------------------------------------------------------------------------
videos_dir = Path(args.videos_dir)  # .expanduser().resolve()
if not videos_dir.is_dir():
    sys.exit(f"[ERROR] Videos directory not found: {videos_dir}")

path_config_file = Path(args.config_file) # .expanduser().resolve()
if not path_config_file.is_file():
    sys.exit(f"[ERROR] config.yaml not found: {path_config_file}")

video_paths = sorted(videos_dir.glob("*.mp4"))
if not video_paths:
    sys.exit(f"[ERROR] No .mp4 files found in {videos_dir}")

videofile_path = [str(p) for p in video_paths]

# -----------------------------------------------------------------------------
# DLC inference
# -----------------------------------------------------------------------------

import deeplabcut as dlc
print("Running analyze_videos on", len(videofile_path), "files …")

dlc.analyze_videos(
    str(path_config_file),
    videofile_path,
    shuffle=args.shuffle_n,
    batchsize=args.batch_size,
)

if args.make_labeled_video:
    print("\nCreating labelled videos …")
    dlc.create_labeled_video(
        str(path_config_file),
        videofile_path,
        shuffle=args.shuffle_n,
    )

print("\n✅ Done.")
