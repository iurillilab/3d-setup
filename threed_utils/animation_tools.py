# %%
import plotly.graph_objects as go
import plotly.graph_objects as go
from IPython.display import display, clear_output
from pathlib import Path
import numpy as np
from tqdm import tqdm
import plotly.express as px
import pickle
import multicam_calibration as mcc
from mpl_toolkits.mplot3d import Axes3D
import xarray as xr
import argparse

#%%

def animator_3d_plotly_xarray(datasets, arena_xarray=None, labels=None, skeleton=None, speed=200, 
                            visible_datasets=None, save_path=None):
    """
    Create 3D animation from multiple xarray datasets containing pose data
    
    Args:
        datasets: List of xarray datasets or single dataset with dimensions:
                 (time, space, keypoints, individuals)
        arena_xarray: xarray dataset containing arena points in 3D space
        labels: List of labels for each dataset (optional)
        skeleton: List of tuples defining connections between points (optional)
        speed: Animation speed in milliseconds per frame
        visible_datasets: List of booleans to toggle visibility of each dataset (optional)
        save_path: Path to save the animation as HTML (optional)
    """
    # Convert single dataset to list
    if not isinstance(datasets, list):
        datasets = [datasets]
    
    # Set default labels if none provided
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(len(datasets))]
    
    # Set default visibility if none provided
    if visible_datasets is None:
        visible_datasets = [True] * len(datasets)
    
    # Define colors for different datasets
    dataset_colors = px.colors.qualitative.Set3
    skeleton_colors = ['rgba(100, 100, 100, 0.8)', 'rgba(150, 150, 150, 0.8)', 
                      'rgba(200, 200, 200, 0.8)']
    
    # Get number of keypoints in first dataset
    n_keypoints = datasets[0].position.shape[2]  # keypoints dimension
    
    # Get keypoint names and create index mapping
    keypoint_names = datasets[0].keypoints.values
    keypoint_dict = {name: idx for idx, name in enumerate(keypoint_names)}
    
    # Define default skeleton using keypoint names
    if skeleton is None:
        skeleton = [
            # Head triangle
            ('nose', 'lear'),
            ('lear', 'rear'),
            ('rear', 'nose'),
            
            # Back connections
            ('upperback', 'tailbase'),
            ('upperback', 'uppermid'),
            ('uppermid', 'upperforward'),
            
            # Back limbs
            ('blimbmid', 'rblimb'),
            ('blimbmid', 'lblimb'),
            
            # Front limbs
            ('flimbmid', 'lflimb'),
            ('flimbmid', 'rflimb'),
            
            # Upper forward connections
            ('upperforward', 'lear'),
            ('upperforward', 'rear'),
            
            # Mid connections
            ('uppermid', 'flimbmid'),
            ('uppermid', 'blimbmid'),
        ]
        
        # Convert named connections to indices
        skeleton = [(keypoint_dict[start], keypoint_dict[end]) for start, end in skeleton]
    
    # Get number of frames
    n_frames = datasets[0].sizes['time']
    
    animato_frames = []
    for frame_idx in range(n_frames):
        frame_data = []
        
        # Add arena if provided
        if arena_xarray is not None:
            # Extract arena points and reshape to (n_points, 3)
            arena_points = arena_xarray.position.isel(time=0, individuals=0).transpose('keypoints', 'space').values
            arena_mesh = go.Mesh3d(
                x=arena_points[:, 0],
                y=arena_points[:, 1],
                z=arena_points[:, 2],
                color='rgba(200, 200, 200, 0.3)',
                name='arena',
                alphahull=0,
                visible=True,
                showlegend=True
            )
            frame_data.append(arena_mesh)
        
        # Add data for each dataset
        for ds_idx, (ds, label, is_visible) in enumerate(zip(datasets, labels, visible_datasets)):
            if not is_visible:
                continue
                
            # Extract points for current frame and reshape to (keypoints, 3)
            points = ds.position.isel(time=frame_idx, individuals=0).transpose('keypoints', 'space').values
            
            # Add scatter plot for points
            scatter = go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color=dataset_colors[ds_idx],
                    opacity=0.8,
                ),
                name=label,
                text=[f'{label} {kp}' for kp in keypoint_names],  # Use actual keypoint names
                hoverinfo='text+x+y+z'
            )
            frame_data.append(scatter)
            
            # Add skeleton lines
            for start, finish in skeleton:
                line = go.Scatter3d(
                    x=[points[start, 0], points[finish, 0]],
                    y=[points[start, 1], points[finish, 1]],
                    z=[points[start, 2], points[finish, 2]],
                    mode='lines',
                    line=dict(
                        width=2,
                        color=skeleton_colors[ds_idx % len(skeleton_colors)]
                    ),
                    name=f'{label} skeleton',
                    showlegend=False
                )
                frame_data.append(line)
        
        animato_frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
    
    # Create figure with first frame
    fig = go.Figure(data=animato_frames[0].data)
    
    # Update layout with legend for arena toggle
    fig.update_layout(
        scene=dict(
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='data'
        ),
        showlegend=True,  # Show legend for arena toggle
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            x=0,
            y=0,
            xanchor='left',
            yanchor='top',
            pad=dict(t=0, r=10),
            buttons=[
                dict(
                    label='Play',
                    method='animate',
                    args=[None, {
                        'frame': {'duration': speed, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0},
                        'mode': 'immediate'
                    }]
                ),
                dict(
                    label='Pause',
                    method='animate',
                    args=[[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                )
            ]
        )],
        sliders=[dict(
            currentvalue=dict(
                font=dict(size=12),
                prefix='Frame: ',
                visible=True,
                xanchor='right'
            ),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.1,
            y=0,
            steps=[dict(
                args=[[f.name], dict(mode='immediate', frame=dict(duration=speed, redraw=True))],
                label=str(k),
                method='animate'
            ) for k, f in enumerate(animato_frames)]
        )]
    )
    
    fig.frames = animato_frames
    
    # Save if path provided
    if save_path:
        import os
        # Convert to absolute path if relative path is given
        save_path = os.path.abspath(save_path)
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"Animation saved to: {save_path}")
    
    return fig

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description='Create 3D animation from pose and arena data')
    parser.add_argument('--pose_files', type=str, nargs='+', required=True,
                      help='Paths to the pose xarray datasets (.nc or .h5 files). Multiple files can be provided.')
    parser.add_argument('--pose_labels', type=str, nargs='+',
                      help='Labels for each pose dataset. Must match number of pose files.')
    parser.add_argument('--arena_file', type=str, required=True,
                      help='Path to the arena xarray dataset (.nc or .h5 file)')
    parser.add_argument('--output', type=str, default='animation.html',
                      help='Path to save the output animation (default: animation.html)')
    parser.add_argument('--speed', type=int, default=150,
                      help='Animation speed in milliseconds per frame (default: 150)')
    
    args = parser.parse_args()
    time_slice = (0, 500)
    # Load the datasets

    try:
        pose_datasets = [xr.open_dataset(file) for file in args.pose_files]
        pose_datasets = [ds.sel(time=slice(*time_slice)) for ds in pose_datasets]
        arena_ds = xr.open_dataset(args.arena_file)
        
        # Create default labels if not provided
        if args.pose_labels is None:
            pose_labels = [f"Pose {i+1}" for i in range(len(pose_datasets))]
        else:
            # Verify number of labels matches number of files
            if len(args.pose_labels) != len(args.pose_files):
                raise ValueError("Number of labels must match number of pose files")
            pose_labels = args.pose_labels
        
        print("Dataset shapes:")
        for i, ds in enumerate(pose_datasets):
            print(f"{pose_labels[i]}: {ds.position.shape}")
        print(f"Arena: {arena_ds.position.shape}")
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        exit(1)
    
    # Create and show the animation
    fig = animator_3d_plotly_xarray(
        datasets=pose_datasets,
        arena_xarray=arena_ds,
        labels=pose_labels,
        visible_datasets=[True] * len(pose_datasets),
        speed=args.speed,
        save_path=args.output
    )
    fig.show()


# python animation_tools.py --pose_files  /Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/multicam_video_2024-07-22T10_19_22_cropped_20250325101012/multicam_video_2024-07-22T10_19_22_cropped_20250325101012_triangulated_points_20250327-124608.h5 \
#           --arena_file /Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/newarena.h5\
#              --output /Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/animation_try_filtered.html \
#             --speed 200 \
#             --pose_labels  "anipose_optimised extra"