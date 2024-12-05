
import plotly.graph_objects as go
import ipywidgets as widgets
import threading
import asyncio
import plotly.graph_objects as go
from IPython.display import display, clear_output
import time
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import plotly.express as px
import pickle

import multicam_calibration as mcc

from mpl_toolkits.mplot3d import Axes3D


with open(r'C:\Users\SNeurobiology\code\3d-setup\tests\assets\arena_tracked_points.pkl', 'rb') as f:
    arena_triangulation = pickle.load(f)


# Load the calibration data
adj_intrinsics = arena_triangulation['intrinsics']
adj_extrinsics = arena_triangulation['extrinsics']
calib_uvs = arena_triangulation['points']

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


# triangualte points

#TODO: save arena points with right format of calibration [..., [1, 0]]
points_3d = triangulate_all_keypoints(calib_uvs['tracked_points'], adj_extrinsics, adj_intrinsics)

arena_3d = triangulate_all_keypoints(calib_uvs['arena_coordinates'][...,[1, 0]], adj_extrinsics, adj_intrinsics)



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



if __name__ == "__main__":
    animator_3d_plotly(points_3d, arena_3d)
