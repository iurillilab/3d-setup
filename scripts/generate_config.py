#!/usr/bin/env python3

from pathlib import Path
import yaml
from datetime import datetime


def generate_config(scorer, project_path, video_paths, bodyparts):
    """
    Generate a DeepLabCut config.yaml file with specified parameters
    """
    # Get current date
    date = datetime.now().strftime("%b%d")
    
    # Create video_sets dict
    video_sets = {}
    for video_path in video_paths:
        video_sets[str(video_path)] = {"crop": "0, 1108, 0, 752"}
    
    # Create skeleton based on bodyparts structure
    skeleton = []
    
    # Group bodyparts by prefix
    grouped_bodyparts = {}
    for bp in bodyparts:
        prefix = bp.split('_')[0]
        if prefix not in grouped_bodyparts:
            grouped_bodyparts[prefix] = []
        grouped_bodyparts[prefix].append(bp)
    
    # Create skeleton for each group
    for prefix, parts in grouped_bodyparts.items():
        if len(parts) >= 3:  # Only create skeleton if at least 3 points
            if parts[0] == parts[-1]:  # Skip if already a closed loop
                skeleton.append(parts)
            else:
                # Create a closed loop for certain parts
                if prefix in ["nose", "eye", "ear", "iris"]:
                    skeleton.append(parts + [parts[0]])
                else:
                    skeleton.append(parts)
    
    # Create config dictionary
    config = {
        "Task": "mouseface",
        "scorer": scorer,
        "date": date,
        "multianimalproject": False,
        "identity": None,
        
        "project_path": str(project_path),
        
        "video_sets": video_sets,
        "bodyparts": bodyparts,
        
        "start": 0,
        "stop": 1,
        "numframes2pick": 20,
        
        "skeleton": skeleton,
        "skeleton_color": "black",
        "pcutoff": 0.6,
        "dotsize": 12,
        "alphavalue": 0.7,
        "colormap": "rainbow",
        
        "TrainingFraction": [0.95],
        "iteration": 0,
        "default_net_type": "resnet_50",
        "default_augmenter": "default",
        "snapshotindex": -1,
        "batch_size": 8,
        
        "cropping": False,
        "x1": 0,
        "x2": 640,
        "y1": 277,
        "y2": 624,
        
        "corner2move2": [50, 50],
        "move2corner": True
    }
    
    return config


def save_config(config, output_folder: Path):
    """Save the config dictionary as a YAML file with proper formatting"""
    path = output_folder / "config.yaml"
    
    # Convert config to string with proper formatting
    yaml_str = ""
    for key, value in config.items():
        if value is None:
            yaml_str += f"{key}:\n"
        else:
            value_str = yaml.dump({key: value}, default_flow_style=False)
            yaml_str += value_str
    
    # Write to file
    path.write_text(yaml_str)
    print(f"Config saved to {path}")



target_folder = Path("/Users/vigji/Desktop/dummy_project")
video_paths = [...]
keypoints = [...]
config = generate_config(
        scorer="movement", 
        project_path=target_folder,
        video_paths=video_paths,
        bodyparts=keypoints
    )

save_config(config, target_folder)