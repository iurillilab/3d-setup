#!/usr/bin/env python3
"""
Parallel DeepLabCut video processing.

Usage:
    python dlc_parallel.py /path/to/project/config.yaml /path/to/videos \
           --output /path/to/output --processes 4 --ext mp4

    eg 
python scripts/dlc/run_dlc_inference_parallel.py /mnt/d/Luigi/dlc3_mouse-YaduLuigi-2025-06-10/config.yaml /mnt/d/Luigi/test-videos

"""
import argparse
import multiprocessing as mp
import os
from pathlib import Path

import deeplabcut as dlc


def _process(args):
    config, video, outdir = args
    video = str(video)

    # Analyze
    dlc.analyze_videos(
        config,
        [video],
        save_as_csv=True,
        destfolder=outdir,
        batchsize=1,
        shuffle=2
    )

    # Create labeled video (optional; comment out if not needed)
    dlc.create_labeled_video(
        config,
        [video],
        destfolder=outdir,
        draw_skeleton=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run DeepLabCut on multiple videos in parallel."
    )
    parser.add_argument("config", help="Path to DLC config.yaml")
    parser.add_argument("video_dir", help="Directory containing videos")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Destination folder for DLC outputs (default: project’s auto locations)",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=max(1, os.cpu_count() // 2),
        help="Number of worker processes (default: half your CPUs)",
    )
    parser.add_argument(
        "-e",
        "--ext",
        default="mp4",
        help="Video extension to match (default: mp4)",
    )
    args = parser.parse_args()

    videos = sorted(Path(args.video_dir).glob(f"*.{args.ext}"))
    if not videos:
        raise SystemExit("No videos found with extension " + args.ext)

    iterable = [(args.config, v, args.output) for v in videos]

    mp.set_start_method("spawn", force=True)  # safe on all OSes
    with mp.Pool(processes=args.processes) as pool:
        pool.map(_process, iterable)


if __name__ == "__main__":
    main()
