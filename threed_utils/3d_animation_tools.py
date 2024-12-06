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

# %%

# with open(r'C:\Users\SNeurobiology\code\3d-setup\tests\assets\arena_tracked_points.pkl', 'rb') as f:
#     arena_triangulation = pickle.load(f)

with open('/Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/3d-setup/tests/assets/arena_tracked_points.pkl', 'rb') as f:
    arena_triangulation = pickle.load(f)


# Load the calibration data
adj_intrinsics = arena_triangulation['intrinsics']
adj_extrinsics = arena_triangulation['extrinsics']
calib_uvs = arena_triangulation['points']
#%%
calib_uvs['tracked_points'].shape


# %%
def moving_average_filter(data, window_size=5):
    """
    Apply a moving average filter to smooth data across all cameras.
    
    Args:
        data: np.array of shape (5, 200, 13, 2)
              5 cameras, 200 frames, 13 keypoints, 2 dimensions (x, y)
        window_size: Size of the moving window
    
    Returns:
        Smoothed data of the same shape
    """
    smoothed_data = np.empty_like(data)
    for camera_idx in range(data.shape[0]):  # Iterate over cameras
        for keypoint_idx in range(data.shape[2]):  # Iterate over keypoints
            for dim in range(data.shape[3]):  # Iterate over dimensions (x, y)
                smoothed_data[camera_idx, :, keypoint_idx, dim] = np.convolve(
                    data[camera_idx, :, keypoint_idx, dim], 
                    np.ones(window_size) / window_size, 
                    mode='same'
                )
    return smoothed_data

# Example usage
smoothed_2d_points = moving_average_filter(calib_uvs['tracked_points'])


#%%

# function to triangulate all the keypoints
# TODO: move the function inside the triangulation.py file
def triangulate_all_keypoints(
    calib_uvs, adj_extrinsics, adj_intrinsics, progress_bar=True
):
    all_triang = []
    progbar = tqdm if progress_bar else lambda x: x
    for i in progbar(range(calib_uvs.shape[2])):
        all_triang.append(
            mcc.triangulate(calib_uvs[:, :, i, :], adj_extrinsics, adj_intrinsics)
        )

    return np.array(all_triang)

# %%
# triangualte points

#TODO: save arena points with right format of calibration [..., [1, 0]]
points_3d = triangulate_all_keypoints(smoothed_2d_points, adj_extrinsics, adj_intrinsics)

arena_3d = triangulate_all_keypoints(calib_uvs['arena_coordinates'][...,[1, 0]], adj_extrinsics, adj_intrinsics)

# %%
# from pykalman import KalmanFilter

# def kalman_filter_3d(points_3d):
#     """
#     Apply a Kalman filter to smooth 3D points.
    
#     Args:
#         points_3d: np.array of shape (n_keypoints, n_frames, 3)
    
#     Returns:
#         Smoothed 3D points
#     """
#     n_keypoints, n_frames, _ = points_3d.shape
#     smoothed_points = np.zeros_like(points_3d)
    
#     for i in range(n_keypoints):
#         kf = KalmanFilter(initial_state_mean=points_3d[i, 0], n_dim_obs=3)
#         smoothed_state_means, _ = kf.smooth(points_3d[i])
#         smoothed_points[i] = smoothed_state_means
    
#     return smoothed_points

# smoothed_3d_points = kalman_filter_3d(points_3d)
# %%
def median_filter_3d(points_3d, kernel_size=5):
    """
    Apply median filter for outlier removal
    """
    from scipy.signal import medfilt
    filtered = np.zeros_like(points_3d)
    
    for i in range(points_3d.shape[0]):
        for j in range(3):
            filtered[i, :, j] = medfilt(points_3d[i, :, j], kernel_size)
    
    return filtered

median_filtered_3d_points = median_filter_3d(points_3d)
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

if __name__ == "__main__":
    fig = animator_3d_plotly_v02(median_filtered_3d_points , arena_3d)
    # fig.show()
    fig.write_html("update_3d_animation.html")
    print('hi')# %%
