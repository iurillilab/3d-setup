import datetime
import json
from pathlib import Path

from tqdm import tqdm
from utils import crop_all_views

MODELS_LOCATIONS = {
    "side": r"D:\SLEAP_models\SLEAP_side_models\models\241007_120850.single_instance.n=500",
    "bottom": r"D:\SLEAP_models\SLEAP_bottom_model\models\241106_104724.single_instance.n=161",
}

MODELS_MAP_TO_VIEW = {
    "side": [],
    "bottom": ["central"]
}


def process_videos_in_folder(folder, json_file, timestamp):
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
        output_dir = avi_file.parent / f"{avi_file.stem}_cropped_{timestamp}"
        cropped_filenames = crop_all_views(avi_file, output_dir, json_file, verbose=False)

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

    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    process_videos_in_folder(Path(args.folder), Path(args.json_file), timestamp)
