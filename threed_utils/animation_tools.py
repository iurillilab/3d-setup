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

#%%
arena_path = '/Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/3d-setup/tests/assets/arena_tracked_points.pkl'
with open(arena_path, 'rb') as f:
    arena_points = pickle.load(f)

arena_2d =arena_points['points']['arena_coordinates']

intrinsics = arena_points['intrinsics']

extrinsics = arena_points['extrinsics']

# %%

def triangulate_keypoints(keypoints_2d, intrinsics, extrinsics):
    all_triang = []
    for i in tqdm(range(keypoints_2d.shape[2])):
        all_triang.append(
            mcc.triangulate(
                keypoints_2d[:, :, i, [1, 0]], extrinsics, intrinsics
            )
        )
    all_triang = np.array(all_triang)
    return all_triang
arena_3d = triangulate_keypoints(arena_2d, intrinsics, extrinsics)
arena_3d = arena_3d.squeeze()

# %%
import xarray as xr
triangulation_path = '/Users/thomasbush/Downloads/anipose_triangulated_ds.h5'
mcc_path = '/Users/thomasbush/Downloads/mcc_triangulated_ds.h5'
anipose_ds = xr.open_dataset(triangulation_path)
mcc_ds = xr.open_dataset(mcc_path)


#%%

#%%


def animator_3d_plotly(tracked_points_sample, arena_triangulation, skeleton=None):

    """"
    Function to create 3d animation from tracked points. 

    args:   
    - tracked_points: np.array: array of shape (n_keypoints, n_frames, 3) containing the tracked points
    - skeleton: list of tuples: list of tuples containing the connections between the points
    - arena_triangulation: np.array: array of shape (n_points, 3) containing the triangulated points of the arena
    
    returns:
    It generates a 3d animation of the tracked points inside the arena
    """
    animato_frames = []
    if skeleton is None:
        skeleton = [(0, 1), (0, 2), (5, 3), (5, 4), (8, 6), (8, 7), (10, 11), (11, 12), (12, 9), (1, 10), (2, 10), (10, 5), (8, 12)]


    for frame_idx in range(tracked_points_sample.shape[1]):

        scatter = go.Scatter3d(
            x=tracked_points_sample[:, frame_idx, 0], y=tracked_points_sample[:, frame_idx, 1], z=tracked_points_sample[:, frame_idx, 2],
            mode='markers', marker=dict(size=5, color='blue'), name='tracked points'
        )

        lines = []
        for start, finish in skeleton:
            lines.append(go.Scatter3d(
                x=[tracked_points_sample[start, frame_idx, 0], tracked_points_sample[finish, frame_idx, 0]],
                y=[tracked_points_sample[start, frame_idx, 1], tracked_points_sample[finish, frame_idx, 1]],
                z=[tracked_points_sample[start, frame_idx, 2], tracked_points_sample[finish, frame_idx, 2]],
                mode='lines', line=dict(width=2, color='gray'), name='skeleton'
            ))

        arena_mesh = go.Mesh3d(
        x = arena_triangulation.squeeze()[:, 0], y = arena_triangulation.squeeze()[:, 1], z = arena_triangulation.squeeze()[:, 2], color='lightpink', opacity=0.5, name='arena', alphahull=0
    )

        animato_frames.append(go.Frame(data=[scatter, arena_mesh] + lines, name = str(frame_idx)))

    fig = go.Figure(data=[animato_frames[0].data[0], animato_frames[0].data[1]] + list(animato_frames[0].data[2:]))


    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(label='Play',
                        method='animate',
                        args = [None, {'frame':{'duration':200, 'redraw':True},
                                        'fromcurrent':True, 'mode':'immediate'}]),
                                        dict(label='pause', 
                                            method='animate',
                                            args=[None, {'frame':{'duration':0, 'redraw':True},
                                                        'mode':'immediate'}])]
        )])

    fig.frames = animato_frames
    fig.show()

def animator_3d_plotly_v02(tracked_points_sample, arena_triangulation, skeleton=None, speed=200):
    """
    Enhanced 3D animation function
    
    Args:
        tracked_points_sample: np.array of shape (n_keypoints, n_frames, 3)
        arena_triangulation: np.array of arena points
        skeleton: list of tuples defining connections between points
        speed: Animation speed in milliseconds per frame
    """
    # Define better colors for visualization
    point_colors = px.colors.qualitative.Set3
    skeleton_color = 'rgba(100, 100, 100, 0.8)'
    
    if skeleton is None:
        skeleton = [(0, 1), (0, 2), (5, 3), (5, 4), (8, 6), (8, 7), 
                   (10, 11), (11, 12), (12, 9), (1, 10), (2, 10), (10, 5), (8, 12)]

    # Define point labels
    point_labels = [f'Point {i}' for i in range(tracked_points_sample.shape[0])]
    
    animato_frames = []
    for frame_idx in range(tracked_points_sample.shape[1]):
        scatter = go.Scatter3d(
            x=tracked_points_sample[:, frame_idx, 0], 
            y=tracked_points_sample[:, frame_idx, 1], 
            z=tracked_points_sample[:, frame_idx, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=point_colors[:tracked_points_sample.shape[0]],
                opacity=0.8
            ),
            name='tracked points',
            text=point_labels,
            hoverinfo='text+x+y+z'
        )

        lines = []
        for start, finish in skeleton:
            lines.append(go.Scatter3d(
                x=[tracked_points_sample[start, frame_idx, 0], tracked_points_sample[finish, frame_idx, 0]],
                y=[tracked_points_sample[start, frame_idx, 1], tracked_points_sample[finish, frame_idx, 1]],
                z=[tracked_points_sample[start, frame_idx, 2], tracked_points_sample[finish, frame_idx, 2]],
                mode='lines',
                line=dict(width=3, color=skeleton_color),
                name='skeleton',
                showlegend=False
            ))

        arena_mesh = go.Mesh3d(
            x=arena_triangulation.squeeze()[:, 0],
            y=arena_triangulation.squeeze()[:, 1],
            z=arena_triangulation.squeeze()[:, 2],
            color='rgba(200, 200, 200, 0.3)',
            name='arena',
            alphahull=0
        )

        animato_frames.append(go.Frame(data=[scatter, arena_mesh] + lines, name=str(frame_idx)))

    # Create figure with first frame
    fig = go.Figure(data=[animato_frames[0].data[0], animato_frames[0].data[1]] + list(animato_frames[0].data[2:]))

    # Improve layout with better controls
    fig.update_layout(
        scene=dict(
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='data'
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
    return fig

def animator_3d_plotly_xarray(datasets, arena_triangulation=None, labels=None, skeleton=None, speed=200, 
                            visible_datasets=None, save_path=None):
    """
    Create 3D animation from multiple xarray datasets containing pose data
    
    Args:
        datasets: List of xarray datasets or single dataset with dimensions:
                 (time, individuals, keypoints, space)
        arena_triangulation: np.array of arena points (optional)
        labels: List of labels for each dataset (optional)
        skeleton: List of tuples defining connections between points
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
    
    # Default skeleton definition (indices 0-12)
    if skeleton is None:
        skeleton = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                   (1, 8), (8, 9), (9, 10), (10, 11), (11, 12)]
    
    # Get number of frames (assume all datasets have same length)
    n_frames = datasets[0].dims['time']
    
    animato_frames = []
    for frame_idx in range(n_frames):
        frame_data = []
        
        # Add arena if provided
        if arena_triangulation is not None:
            arena_mesh = go.Mesh3d(
                x=arena_triangulation.squeeze()[:, 0],
                y=arena_triangulation.squeeze()[:, 1],
                z=arena_triangulation.squeeze()[:, 2],
                color='rgba(200, 200, 200, 0.3)',
                name='arena',
                alphahull=0,
                visible=True,  # Initial visibility state
                showlegend=True  # Show in legend for toggle
            )
            frame_data.append(arena_mesh)
        
        # Add data for each dataset
        for ds_idx, (ds, label, is_visible) in enumerate(zip(datasets, labels, visible_datasets)):
            if not is_visible:
                continue
                
            # Extract points for current frame
            points = ds.position.isel(time=frame_idx, individuals=0).values
            keypoint_labels = ds.keypoints.values
            
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
                text=[f'{label} {kp}' for kp in keypoint_labels],
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
    fig = animator_3d_plotly_xarray(
        datasets=[anipose_ds, mcc_ds],
        arena_triangulation=arena_3d,
        labels=['Anipose', 'MCC'],
        visible_datasets=[True, True],
        speed=150, 
        save_path='animation.html'
    )
    fig.show()
