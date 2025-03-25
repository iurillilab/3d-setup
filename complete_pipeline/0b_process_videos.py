import datetime
import json
from pathlib import Path
import os
import re
import subprocess

from pathlib import Path
import numpy as np

from tqdm import tqdm
from utils import crop_all_views

# change location of models
MODELS_LOCATIONS = {
    "side": r"D:\SLEAP_models\SLEAP_side_models\models\241007_120850.single_instance.n=500",
    "bottom": r"D:\SLEAP_models\SLEAP_bottom_model\models\241106_104724.single_instance.n=161",
}

MODELS_MAP_TO_VIEW = {
    "side": ['mirror-top', 'mirror-bottom', 'mirror-left', 'mirror-right'],
    "bottom": ["central"]
}
def run_inference(video, model):
    """
    Input: video_paths: list: list of paths to all the videos in the main folder
    It runs the model inference on the encoded videos and saves the results in a specific folder.
    """
    video_path = str(video)
    output_folder = video_path.replace('.avi.mp4', "predictions.slp")
    conda_act = 'conda activate sleap'
    conda_deact = 'conda deactivate'
    command = f"{conda_act} && sleap-track -m {model} -o {output_folder} {video} && {conda_deact}"

    print(f"Running inference on video: {video}")
    try:
        subprocess.run(command, check=True, shell=True, text=True)
        print(f"Inference results saved to: {output_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Error running comman {output_folder}, error: {e}, for video: {video}")


def process_videos_in_folder(folder, json_file, timestamp, skip_existing=True):
    avi_files = list(Path(folder).rglob("*.avi"))
    # filter out files from previous runs, if in the name there's
    # central, mirror-top, mirror-bottom, mirror-left, mirror-right:
    avi_files = [
        f
        for f in avi_files
        if not any(
            view in f.stem
            for view in [
                "central",
                "mirror-top",
                "mirror-bottom",
                "mirror-left",
                "mirror-right",
            ]
        )
    ]

    for avi_file in tqdm(avi_files):
        existing_cropped_dirs = list(avi_file.parent.glob(f"*_cropped_*"))
        if len(existing_cropped_dirs) > 0 and skip_existing:
            print(f"Skipping {avi_file} as it has already been processed")
            continue
        output_dir = avi_file.parent / f"{avi_file.stem}_cropped_{timestamp}"
        cropped_filenames = crop_all_views(avi_file, output_dir, json_file, verbose=False)
        print(cropped_filenames, '\n', type(cropped_filenames))
        cropped_filenames = [f.result() for f in cropped_filenames]

        # TODO process cropped videos usin run_inference:
        for model_name, views in MODELS_MAP_TO_VIEW.items():
            videos_to_process = [video_path for video_path in cropped_filenames if any([view in video_path.name for view in views])]
            for video_path in videos_to_process:
                run_inference(video_path, MODELS_LOCATIONS[model_name])
        

        
        



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process AVI files with cropping parameters"
    )
    parser.add_argument(
        "folder", type=str, help="Folder containing AVI files to process"
    )
    parser.add_argument(
        "json_file", type=str, help="JSON file with cropping parameters"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_false",
        help="Process all files, including ones that have been processed before",
        dest="skip_existing",
        default=False,
    )

    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    process_videos_in_folder(Path(args.folder), Path(args.json_file), timestamp, skip_existing=args.skip_existing)
