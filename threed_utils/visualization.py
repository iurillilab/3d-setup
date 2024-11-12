import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path
import cv2

def test_helloworld():
    return "the usual"


def plot_triangulations(*points_triangulated, labels = None, idx=None):

    '''
    It plots the triangulated points in 3D.

    Parameters:
    points_triangulated: list of numpy arrays of shape (keypoints, frames, 3)
    labels: list of strings: elements that will be used as labels in the plot
    idx: int: index of the frame to plot
    '''

    if not idx:
        idx = 0
    if not None and len(labels) != len(points_triangulated):
        raise ValueError("Number of labels must match number of triangulations")
    
    points_triangulated = points_triangulated[:, idx, :]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.jet(np.linspace(0, 1, len(points_triangulated)))

    for i, arr in enumerate(points_triangulated):
        if arr.shape[1] != 3:
            raise ValueError("Triangulated points must have 3 dimensions")
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], c=colors[i], label=labels[i] if labels is not None else f'Triangulation {i+1}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()




def plot_back_projections(video_dir, back_projections, idx):
    '''
    It plots the back projections of the 3D points in the cameras.
    
    Parameters:
    video_dir: Path: directory where the videos are stored
    back_projections: dictionary of length n_cameras, each element: numpy array of shape (keypoints, frames, 2)
    idx: int: index of the frame to plot
    '''



    video_paths = [str(f) for f in video_dir.iterdir() if f.suffix == '.mp4' and 'overlay' not in f.stem]
    camera_sequence = [Path(video_path).stem.split('_')[-1].split('.avi')[0] for video_path in video_paths]
    frames = {}
    for frame, name in zip(video_paths, camera_sequence):
        cap = cv2.VideoCapture(str(frame))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        res, frame = cap.read()
        frames[name] = frame
    plt.figure()
    for i, view in enumerate(camera_sequence):
        plt.subplot(2, 3, i+1)
        plt.imshow(frames[view])
        plt.scatter(back_projections[view][:, 0], back_projections[view][:, 1], c='r', s=10)
        plt.title(view)
    plt.show()





