import deeplabcut
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Run DLC inference on videos")
parser.add_argument(
    "--videos_dir",
    type=str,
    required=True,
    help="Path to directory containing videos",
)
parser.add_argument(
    "--config_file",
    type=str,
    required=True,
    help="Path to DLC config.yaml file",
)
parser.add_argument(
    "--make_labeled_video",
    action="store_true",
    default=True,
    help="Create labeled video output",
)

args = parser.parse_args()

videos_dir = Path(args.videos_dir)
path_config_file = args.config_file
videos_pattern_to_match = "*mirror*.mp4"

videofile_path_list = [str(filename) for filename in (videos_dir.rglob(videos_pattern_to_match))]

print("analysing videos: ")
for video in videofile_path_list:
    print("  - ", video)

for video in videofile_path_list:
    deeplabcut.analyze_videos(path_config_file, [video], videotype=".mp4")

    if args.make_labeled_video:
        deeplabcut.create_labeled_video(path_config_file, [video])
