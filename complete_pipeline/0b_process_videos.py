import json
from pathlib import Path
from utils import crop_all_views
from tqdm import tqdm


def process_videos_in_folder(folder, json_file):
    
    avi_files = list(Path(folder).rglob('*.avi'))
    
    for avi_file in tqdm(avi_files):
        output_dir = avi_file.parent / f"{avi_file.stem}_cropped"
        crop_all_views(avi_file, output_dir, json_file, verbose=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process AVI files with cropping parameters")
    parser.add_argument('folder', type=str, help='Folder containing AVI files to process')
    parser.add_argument('json_file', type=str, help='JSON file with cropping parameters')

    args = parser.parse_args()
    process_videos_in_folder(Path(args.folder), Path(args.json_file))

