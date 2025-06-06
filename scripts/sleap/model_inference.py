import os
import pathlib
import re
import subprocess
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def get_video_paths(main_path):
    """
    Finds video paths for SLEAP processing, excluding already processed ones.
    """
    side_paths = []
    bottom_paths = []
    main_path = pathlib.Path(main_path)

    all_candidate_folders = [f for f in main_path.rglob("multicam_video_*_cropped_*") if f.is_dir()]
    parent_dict = {folder.parent: [] for folder in all_candidate_folders}
    
    for candidate_folder in all_candidate_folders:
        parent_dict[candidate_folder.parent].append(candidate_folder)
    
    last_folders = [sorted(folders)[-1] for folders in parent_dict.values()]
    
    for candidate_folder in last_folders:        
        for video in candidate_folder.glob("*"):
            if len(list(candidate_folder.glob(f"{video.name.split('.avi')[0]}*.slp"))) > 0: 
                continue
            if "calibration" in [parent.name.lower() for parent in video.parents]:
                continue
                
            if video.is_file() and video.name.endswith(".mp4") and "central" not in video.name:
                side_paths.append(str(video))
            if "central" in video.name:
                bottom_paths.append(str(video))
            
    return side_paths, bottom_paths

def check_sleap_files_complete(main_path):
    """
    Checks if the most recent directories have exactly 5 .slp files each.
    Returns a list of directories that don't have exactly 5 .slp files.
    """
    main_path = pathlib.Path(main_path)
    incomplete_dirs = []

    # Find all candidate folders (same logic as get_video_paths)
    all_candidate_folders = [f for f in main_path.rglob("multicam_video_*_cropped_*") if f.is_dir()]
    parent_dict = {folder.parent: [] for folder in all_candidate_folders}
    
    for candidate_folder in all_candidate_folders:
        parent_dict[candidate_folder.parent].append(candidate_folder)
    
    # Get the most recent folder for each parent
    last_folders = [sorted(folders)[-1] for folders in parent_dict.values()]
    
    # Check each folder for .slp files
    for folder in last_folders:
        slp_files = list(folder.glob("*.slp"))
        if len(slp_files) != 5:
            incomplete_dirs.append({
                'path': str(folder),
                'count': len(slp_files)
            })
    
    return incomplete_dirs

def run_single_inference(video, model):
    """
    Runs inference on a single video.
    """
    output_folder = video.replace(".avi.mp4", "predictions.slp")
    print(f"Running inference on: {video}")
    
    try:
        subprocess.run(["sleap-track", "-m", model, "-o", output_folder, video], check=True)
        print(f"Saved results to: {output_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {video}: {e}")

def run_inference_parallel(video_paths, model, num_workers=4):
    """
    Runs inference in parallel using ProcessPoolExecutor.
    """
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_single_inference, video, model) for video in video_paths]
        for future in futures:
            future.result()  # Ensures errors are caught properly

if __name__ == "__main__":
    from pathlib import Path
    GEN_VIDEO_PATH = r"N:\SNeuroBiology_shared\P07_PREY_HUNTING_YE\e01_video_recordings\M29" # r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting"
    assert Path(GEN_VIDEO_PATH).exists(), f"Path {GEN_VIDEO_PATH} does not exist."

    side_model = r"D:\SLEAP_models\SLEAP_side_models\models\250314_091459.single_instance.n=659"
    bottom_model = r"D:\SLEAP_models\SLEAP_bottom_model\models\250116_131653.single_instance.n=416"
    assert Path(side_model).exists(), f"Side model path {side_model} does not exist."
    assert Path(bottom_model).exists(), f"Bottom model path {bottom_model} does not exist."

    side_paths, bottom_paths = get_video_paths(GEN_VIDEO_PATH)
    print(f"Found {len(side_paths)} side videos and {len(bottom_paths)} bottom videos.")

    # Run inference in parallel
    # run_inference_parallel(side_paths, side_model, num_workers=3)
    run_inference_parallel(bottom_paths, bottom_model, num_workers=1)
    incomplete = check_sleap_files_complete(GEN_VIDEO_PATH)
    if incomplete:
        print("\nWarning: The following directories don't have exactly 5 .slp files:")
        for dir_info in incomplete:
            print(f"Directory: {dir_info['path']}")
            print(f"Found {dir_info['count']} .slp files instead of 5")
    else:
        print("\nAll directories have exactly 5 .slp files!")
    #print("Inference done!")
