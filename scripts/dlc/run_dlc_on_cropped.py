from pathlib import Path
from tqdm import tqdm
from pprint import pprint
import os
import sys
import yaml
from typing import Dict, List, Tuple, Optional


# Change root depending on whether we're mac or linux:
if sys.platform == "darwin":
    print("Running on Mac", sys.platform )
    root_data_path = Path("/Volumes/SNeurobiology_RAW")
else:
    print("Running on Linux", sys.platform )
    root_data_path = Path("/mnt/y")


# CROSS-PLATFORM SOLUTION:
# DLC config files contain hardcoded absolute paths in the 'project_path' field,
# making them OS-specific. To handle this, we update the original config files
# to have the correct project_path pointing to their actual location on the current OS.


def get_model_config() -> Dict[str, Dict]:
    """
    Configuration dictionary that maps path patterns to DLC model configurations.
    
    Returns:
        Dict mapping model names to their configuration including:
        - config_path: path to the DLC config file
        - shuffle: shuffle number for the model
        - path_patterns: list of strings that must be present in video path
        - exclusion_patterns: list of strings that must NOT be present in video path
    """
    models_path = root_data_path / "nas_mirror" / "DLC_final_models"
    
    config = {
        # "cricket": {
        #     "config_path": models_path / "cricket-bottom-vigji-2025-07-01" / "config.yaml",
        #     # "shuffle": 8,
        #     "path_patterns": ["central", "cricket"],
        #     "exclusion_patterns": []
        # },
        # "object": {
        #     "config_path": models_path / "object-bottom-vigji-2025-07-01" / "config.yaml", 
        #     #"shuffle": 3,
        #     "path_patterns": ["central", "object"],
        #     "exclusion_patterns": []
        # },
        # "roach": {
        #     "config_path": models_path / "roach-bottom-vigji-2025-07-01" / "config.yaml",
        #     #"shuffle": 1,  # assuming shuffle 1, adjust as needed
        #     "path_patterns": ["central", "roach"],
        #     "exclusion_patterns": []
        # },
        "mouse-bottom": {
            "config_path": models_path / "mouse-bottom-vigji-2025-07-01" / "config.yaml",
            #"shuffle": 2,
            "path_patterns": ["central"],
            "exclusion_patterns": []
        },
        "mouse-side": {
            "config_path": models_path / "mouse-side-vigji-2025-07-01" / "config.yaml",
            #"shuffle": 2,
            "path_patterns": ["mirror"],
            "exclusion_patterns": []
        }
    }
    
    # Validate that all config files exist and update their project_path
    for model_name, model_config in config.items():
        config_path = Path(model_config["config_path"])
        if not config_path.exists():
            print(f"Warning: Config file for {model_name} does not exist: {config_path}")
        else:
            # Update the config file to have correct project_path for current OS
            update_config_project_path(config_path)
    
    return config


def get_applicable_models(video_path: Path, model_config: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
    """
    Determine which models should be applied to a given video based on path patterns.
    
    Args:
        video_path: Path to the video file
        model_config: Model configuration dictionary from get_model_config()
        
    Returns:
        List of tuples (model_name, model_config_dict) for models that should be applied to this video
    """
    applicable_models = []
    video_path_str = str(video_path).lower()
    
    for model_name, config in model_config.items():
        # Check if all required patterns are present
        patterns_match = all(pattern.lower() in video_path_str for pattern in config["path_patterns"])
        
        # Check if any exclusion patterns are present
        exclusions_match = any(pattern.lower() in video_path_str for pattern in config["exclusion_patterns"])
        
        if patterns_match and not exclusions_match:
            applicable_models.append((model_name, config))
    
    return applicable_models


def update_config_project_path(config_path: Path) -> None:
    """
    Update the project_path in a DLC config file to match its actual location.
    
    Args:
        config_path: Path to the DLC config file to update
    """
    # The project_path should be the directory containing the config file
    correct_project_path = config_path.parent
    
    # Read the current config
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Check if update is needed
    current_project_path = config_data.get('project_path', '')
    if str(correct_project_path) != current_project_path:
        print(f"Updating {config_path.name}: project_path from '{current_project_path}' to '{correct_project_path}'")
        
        # Update the project_path
        config_data['project_path'] = str(correct_project_path)
        
        # Write back to the original file
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    else:
        print(f"Config {config_path.name} already has correct project_path")


def run_inference_on_video(
    video_path: Path, 
    model_name: str, 
    model_config_dict: Dict,
    retrace: bool = False,
    dryrun: bool = False
) -> bool:
    """
    Run DLC inference on a single video with a specific model.
    
    Args:
        video_path: Path to the video file
        model_name: Name of the model to use
        model_config_dict: Model configuration dictionary for this specific model
        retrace: If True, run inference even if output already exists
        dryrun: If True, only print what would be done
        
    Returns:
        True if inference was run (or would be run in dryrun), False if skipped
    """
    config_path = model_config_dict["config_path"]
    shuffle = model_config_dict.get("shuffle")  # Use get() in case shuffle is not defined
    
    # Check if inference has already been run
    existing_outputs = list(video_path.parent.glob(f"{video_path.stem}*{model_name}*.h5"))
    
    if existing_outputs and not retrace:
        if not dryrun:
            print(f"Already inferred {model_name} for {video_path}")
        return False
    
    if dryrun:
        shuffle_info = f" (shuffle={shuffle})" if shuffle is not None else ""
        print(f"Would run {model_name}{shuffle_info} inference on {video_path}")
        return True
    
    try:
        import deeplabcut
        print(f"Running {model_name} inference on {video_path}")
        
        # Build arguments for deeplabcut.analyze_videos
        analyze_args = {
            "config": str(config_path),
            "videos": [str(video_path)],
            "batchsize": 1
        }
        
        # Only add shuffle if it's defined
        if shuffle is not None:
            analyze_args["shuffle"] = shuffle
            
        deeplabcut.analyze_videos(**analyze_args)
        return True
        
    except Exception as e:
        print(f"Error running inference with {model_name} on {video_path}: {e}")
        return False


def process_videos(
    video_paths: List[Path],
    models_to_retrace: Optional[List[str]] = None,
    dryrun: bool = True
) -> None:
    """
    Process multiple videos with appropriate DLC models.
    
    Args:
        video_paths: List of video file paths to process
        models_to_retrace: List of model names to retrace even if output exists
        dryrun: If True, only show what would be processed without running inference
    """
    if models_to_retrace is None:
        models_to_retrace = []
        
    model_config = get_model_config()
    
    # Build list of (video, model_name, model_config_dict, retrace) tuples to process
    videos_to_process = []
    
    for video_path in video_paths:
        applicable_models = get_applicable_models(video_path, model_config)
        
        for model_name, model_config_dict in applicable_models:
            retrace = model_name in models_to_retrace
            
            # Check if we need to process this combination
            existing_outputs = list(video_path.parent.glob(f"{video_path.stem}*{model_name}*.h5"))
            
            if not existing_outputs or retrace:
                videos_to_process.append((video_path, model_name, model_config_dict, retrace))
    
    if dryrun:
        print("=== DRY RUN MODE ===")
        print(f"Found {len(videos_to_process)} video/model combinations to process:")
        for video_path, model_name, model_config_dict, retrace in videos_to_process:
            retrace_str = " (retrace)" if retrace else ""
            shuffle = model_config_dict.get("shuffle")
            shuffle_info = f" (shuffle={shuffle})" if shuffle is not None else ""
            print(f"  {model_name}{shuffle_info}{retrace_str}: {video_path}")
        return
    
    # Run inference
    print(f"Processing {len(videos_to_process)} video/model combinations...")
    
    progress_bar = tqdm(videos_to_process, desc="Processing videos")
    for video_path, model_name, model_config_dict, retrace in progress_bar:
        progress_bar.set_description(f"Processing {model_name}")
        run_inference_on_video(
            video_path, 
            model_name, 
            model_config_dict, 
            retrace=retrace, 
            dryrun=False
        )


if __name__ == "__main__":
    # Example usage - you can modify this section as needed
    
    # Define which models to retrace (run even if output exists)
    models_to_retrace = []  # e.g., ["cricket"]
    
    # Set dryrun mode
    dryrun_mode = False
    
    # Get videos to process - this logic should be made configurable
    # For now, using the original pattern as an example
    video_pattern = "M31*/*11/cricket/*/*v2*/*.mp4" # "M*/*/*/*/*/*central*.mp4"
    
    # You'll need to define the base path - this should be made configurable
    # base_path = Path("/your/data/path")  # Uncomment and set appropriate path
    # base_path = Path("/Users/vigji/Desktop/videos_test/test-cricket-roach-object")
    
    base_path = root_data_path / "nas_mirror" # "P07_PREY_HUNTING_YE" / "e01_ephys_recordings" 
    assert base_path.exists(), f"{base_path} does not exist!"
    all_videos = sorted(list(base_path.glob(video_pattern)))
    pprint(all_videos)
    # Example of how to use the functions:
    process_videos(
        video_paths=all_videos,
        models_to_retrace=models_to_retrace,
        dryrun=dryrun_mode
    )
    