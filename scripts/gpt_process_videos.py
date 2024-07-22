import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from utils import apply_transformations

def process_videos_in_folder(folder, json_file):
    with open(json_file, 'r') as f:
        cropping_specs = json.load(f)
    
    avi_files = list(Path(folder).glob('*.avi'))
    
    for avi_file in avi_files:
        output_dir = avi_file.parent / (avi_file.stem + '_cropped')
        output_dir.mkdir(exist_ok=True)
        
        # Use ThreadPoolExecutor to run the tasks in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for spec in cropping_specs:
                output_file = output_dir / spec['output_file']
                futures.append(executor.submit(apply_transformations, avi_file, output_file, 
                                               spec['crop_width'], spec['crop_height'], spec['crop_x'], spec['crop_y'], spec['transform']))
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process AVI files with cropping parameters")
    parser.add_argument('folder', type=str, help='Folder containing AVI files to process')
    parser.add_argument('json_file', type=str, help='JSON file with cropping parameters')

    args = parser.parse_args()
    process_videos_in_folder(Path(args.folder), Path(args.json_file))
