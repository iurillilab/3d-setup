#!/usr/bin/env python3
"""
3D Pose Comparison Tool
Compare original vs filtered poses with skeleton visualization and frame slider
"""

import argparse
from pathlib import Path
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yaml
import sys


def load_skeleton_from_yaml(yaml_path: str) -> list[tuple[str, str]]:
    """Load skeleton connections from YAML config file."""
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        
        if not config:
            raise ValueError("YAML file is empty or invalid")
        
        skeleton = config.get('skeleton', [])
        if not skeleton:
            raise ValueError("No 'skeleton' key found in YAML file")
        
        skeleton_tuples = []
        for i, connection in enumerate(skeleton):
            if not isinstance(connection, (list, tuple)) or len(connection) != 2:
                raise ValueError(f"Invalid skeleton connection at index {i}: {connection}")
            skeleton_tuples.append(tuple(connection))
        
        print(f"Loaded {len(skeleton_tuples)} skeleton connections")
        return skeleton_tuples
        
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading skeleton from YAML: {e}")


def create_pose_comparison_viewer(original_data, filtered_data, keypoints, skeleton, max_frames=1000):
    """
    Create 3D pose comparison viewer with slider for frame navigation
    """
    # Color palette for keypoints
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow', 'lime', 'navy']
    
    # Create frames for animation
    frames = []
    
    for frame_idx in range(min(max_frames, len(original_data.time))):
        frame_data = []
        
        # Plot keypoints for original data (left subplot)
        for i, kp in enumerate(keypoints):
            if kp in original_data.keypoints.values:
                pos = original_data.sel(keypoints=kp).position.values[frame_idx]
                # Handle array shapes
                if pos.ndim == 3 and pos.shape[2] == 1:
                    pos = pos.squeeze(axis=2)
                elif pos.ndim == 2:
                    pos = pos.squeeze(axis=1)
                
                if not np.isnan(pos).any():
                    frame_data.append(go.Scatter3d(
                        x=[pos[0]], y=[pos[1]], z=[pos[2]],
                        mode='markers',
                        name=f'{kp} (Original)',
                        marker=dict(size=8, color=colors[i % len(colors)], opacity=0.8),
                        showlegend=True,
                        scene='scene'  # Left subplot
                    ))
        
        # Plot keypoints for filtered data (right subplot)
        for i, kp in enumerate(keypoints):
            if kp in filtered_data.keypoints.values:
                pos = filtered_data.sel(keypoints=kp).position.values[frame_idx]
                # Handle array shapes
                if pos.ndim == 3 and pos.shape[2] == 1:
                    pos = pos.squeeze(axis=2)
                elif pos.ndim == 2:
                    pos = pos.squeeze(axis=1)
                
                if not np.isnan(pos).any():
                    frame_data.append(go.Scatter3d(
                        x=[pos[0]], y=[pos[1]], z=[pos[2]],
                        mode='markers',
                        name=f'{kp} (Filtered)',
                        marker=dict(size=8, color=colors[i % len(colors)], opacity=0.8),
                        showlegend=True,
                        scene='scene2'  # Right subplot
                    ))
        
        # Add skeleton connections for original data (left subplot)
        for conn in skeleton:
            kp1, kp2 = conn
            if (kp1 in keypoints and kp2 in keypoints and 
                kp1 in original_data.keypoints.values and kp2 in original_data.keypoints.values):
                
                pos1 = original_data.sel(keypoints=kp1).position.values[frame_idx]
                pos2 = original_data.sel(keypoints=kp2).position.values[frame_idx]
                
                # Handle array shapes
                if pos1.ndim == 3 and pos1.shape[2] == 1:
                    pos1 = pos1.squeeze(axis=2)
                elif pos1.ndim == 2:
                    pos1 = pos1.squeeze(axis=1)
                if pos2.ndim == 3 and pos2.shape[2] == 1:
                    pos2 = pos2.squeeze(axis=2)
                elif pos2.ndim == 2:
                    pos2 = pos2.squeeze(axis=1)
                
                if not np.isnan(pos1).any() and not np.isnan(pos2).any():
                    frame_data.append(go.Scatter3d(
                        x=[pos1[0], pos2[0]],
                        y=[pos1[1], pos2[1]],
                        z=[pos1[2], pos2[2]],
                        mode='lines',
                        line=dict(color='gray', width=4),
                        name=f'{kp1}-{kp2} (Original)',
                        showlegend=False,
                        opacity=0.8,
                        scene='scene'  # Left subplot
                    ))
        
        # Add skeleton connections for filtered data (right subplot)
        for conn in skeleton:
            kp1, kp2 = conn
            if (kp1 in keypoints and kp2 in keypoints and 
                kp1 in filtered_data.keypoints.values and kp2 in filtered_data.keypoints.values):
                
                pos1 = filtered_data.sel(keypoints=kp1).position.values[frame_idx]
                pos2 = filtered_data.sel(keypoints=kp2).position.values[frame_idx]
                
                # Handle array shapes
                if pos1.ndim == 3 and pos1.shape[2] == 1:
                    pos1 = pos1.squeeze(axis=2)
                elif pos1.ndim == 2:
                    pos1 = pos1.squeeze(axis=1)
                if pos2.ndim == 3 and pos2.shape[2] == 1:
                    pos2 = pos2.squeeze(axis=2)
                elif pos2.ndim == 2:
                    pos2 = pos2.squeeze(axis=1)
                
                if not np.isnan(pos1).any() and not np.isnan(pos2).any():
                    frame_data.append(go.Scatter3d(
                        x=[pos1[0], pos2[0]],
                        y=[pos1[1], pos2[1]],
                        z=[pos1[2], pos2[2]],
                        mode='lines',
                        line=dict(color='red', width=4),
                        name=f'{kp1}-{kp2} (Filtered)',
                        showlegend=False,
                        opacity=0.8,
                        scene='scene2'  # Right subplot
                    ))
        
        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=['Original Pose', 'Filtered Pose'],
        horizontal_spacing=0.1
    )
    
    # Add initial data for first frame
    if frames:
        for trace in frames[0].data:
            fig.add_trace(trace)
    
    # Add slider
    fig.update_layout(
        title="3D Pose Comparison: Original vs Filtered",
        height=700,
        showlegend=True,
        template="plotly_white",
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Frame:',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [[f.name], {'frame': {'duration': 300, 'redraw': True},
                                       'mode': 'immediate',
                                       'transition': {'duration': 300}}],
                    'label': f.name,
                    'method': 'animate'
                } for f in frames
            ]
        }]
    )
    
    # Update scenes
    fig.update_scenes(
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)", 
        zaxis_title="Z (mm)",
        aspectmode="data"
    )
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="3D Pose Comparison Tool")
    parser.add_argument("--original", required=True, help="Path to original data")
    parser.add_argument("--filtered", required=True, help="Path to filtered data")
    parser.add_argument("--config", required=True, help="Path to YAML config with skeleton")
    parser.add_argument("-k", "--keypoints", nargs="+", default=None, help="Keypoints to visualize")
    parser.add_argument("--max_frame", type=int, default=1000, help="Maximum frame range")
    
    args = parser.parse_args()
    
    # Validate input files
    original_path = Path(args.original)
    filtered_path = Path(args.filtered)
    config_path = Path(args.config)
    
    if not original_path.exists():
        print(f"ERROR: Original data file not found: {original_path}")
        return 1
    
    if not filtered_path.exists():
        print(f"ERROR: Filtered data file not found: {filtered_path}")
        return 1
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1
    
    try:
        print(f"Loading original data from: {args.original}")
        print(f"Loading filtered data from: {args.filtered}")
        print(f"Loading skeleton from: {args.config}")
        
        # Load data
        original = xr.open_dataset(args.original)
        filtered = xr.open_dataset(args.filtered)
        
        # Load skeleton
        skeleton = load_skeleton_from_yaml(args.config)
        
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        return 1
    
    # Select keypoints
    if args.keypoints:
        print(f"Requested keypoints: {args.keypoints}")
        # Check which keypoints are available in both datasets
        available_orig = [kp for kp in args.keypoints if kp in original.keypoints.values]
        available_filt = [kp for kp in args.keypoints if kp in filtered.keypoints.values]
        common_keypoints = [kp for kp in args.keypoints if kp in original.keypoints.values and kp in filtered.keypoints.values]
        
        print(f"Available in original: {available_orig}")
        print(f"Available in filtered: {available_filt}")
        print(f"Common keypoints: {common_keypoints}")
        
        if not common_keypoints:
            print("ERROR: No common keypoints found between original and filtered data!")
            return
            
        args.keypoints = common_keypoints
        original = original.sel(keypoints=common_keypoints)
        filtered = filtered.sel(keypoints=common_keypoints)
    else:
        # Use all keypoints available in both datasets
        common_keypoints = [kp for kp in original.keypoints.values if kp in filtered.keypoints.values]
        print(f"Using common keypoints: {common_keypoints}")
        args.keypoints = common_keypoints
        original = original.sel(keypoints=common_keypoints)
        filtered = filtered.sel(keypoints=common_keypoints)
    
    # Validate frame range
    max_frames_orig = len(original.time)
    max_frames_filt = len(filtered.time)
    max_frames = min(max_frames_orig, max_frames_filt, args.max_frame)
    
    print(f"Creating comparison viewer for {max_frames} frames")
    
    # Generate the pose comparison viewer
    fig = create_pose_comparison_viewer(original, filtered, args.keypoints, skeleton, max_frames)
    
    print("\n" + "="*70)
    print("3D POSE COMPARISON VIEWER OPENED IN BROWSER")
    print("="*70)
    print("INTERACTIVE FEATURES:")
    print("• Frame slider: Navigate through all frames")
    print("• Mouse wheel: Zoom in/out")
    print("• Click & drag: Rotate 3D view")
    print("• Double-click: Reset zoom")
    print("• Hover: See exact coordinates")
    print("• Legend: Click to show/hide keypoints")
    print("")
    print("SLIDER CONTROLS:")
    print("• Drag slider: Navigate frames")
    print("• Click on slider: Jump to specific frame")
    print("• Gray skeleton: Original data")
    print("• Red skeleton: Filtered data")
    print("="*70)
    
    try:
        fig.show()
    except Exception as e:
        print(f"ERROR: Failed to display plot: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
