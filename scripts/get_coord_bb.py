import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import cv2
import napari
import numpy as np
from matplotlib import cm

# # dispaly user interface to select the directory, it expect a directory with the 5 videos cropped in mp4 format (it can be changed, but mp4 works better with sleaap)
# root = tk.Tk()
# root.withdraw()

# directory_path = filedialog.askdirectory(title='Select the directory where the videos are located')

# if directory_path:
#     path = Path(directory_path)
# else:
#     print("No directory selected")


def bounding_boxes_multiple_videos(directory_path):
    """Function that iterates through the cropped videos and select the bounding boxes for each video
    -args: path to the directory where the videos are located

    - it itereates through all the videos cropped and it runs the get_coordinates_bounding_boxes function
    """

    for video in path.glob("*.mp4"):
        print(video)
        get_coordinates_bounding_boxes(str(video))


def get_coordinates(frame, name: str, coordinates):
    viewer = napari.Viewer()
    cmap = cm.get_cmap("hsv", len(coordinates[name]))
    colors = [cmap(i) for i in range(len(coordinates[name]))]

    viewer.add_image(frame, name=name, contrast_limits=[0, 255])
    points_layer = viewer.add_points(
        coordinates[name],
        size=10,
        face_color=colors,
        name="points",
        edge_color="white",
        edge_width=0.5,
    )
    points_layer.editable = True
    napari.run()
    adjusted_points = points_layer.data
    return adjusted_points


def get_coordinates_bounding_boxes(video_path):
    """
    Function that allows you to select with napari the bounding boxes of the arena and get the coordinates for improve calibration

    -args:
        video_path: str, path to the video file
    -returns:
        list of tuples with the coordinates of the bounding boxes

    """

    cap = cv2.VideoCapture(video_path)
    frame_count = 100
    frame_interval = 100
    pooled_frames = []

    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()

        if not ret:
            print(f"Failed to read frame {i}")
            break
        pooled_frames.append(frame)

    frame_avg = np.mean(pooled_frames, axis=0)
    frame_avg = np.uint8(frame_avg)
    frame_avg = cv2.cvtColor(frame_avg, cv2.COLOR_BGR2GRAY)
    return get_coordinates(frame_avg)


def hflip(arr, value):
    points_flipped = np.zeros_like(arr)  # initialize the array
    points_flipped[:, 1] = (
        np.asanyarray(value)[2] - arr[:, 1]
    )  # the new x is the width - the old x
    points_flipped[:, 0] = arr[:, 0]  # the y remains the same
    return points_flipped, np.asanyarray(value)


def transpose1(arr, value):
    points_transposed = np.zeros_like(arr)  # initialize the array
    points_transposed[:, 1] = (
        np.asanyarray(value)[3] - arr[:, 0]
    )  # the new x is the height - the old y
    points_transposed[:, 0] = arr[:, 1]  # the y remains the same
    value_tranps = np.array(
        [value[0], value[1], value[3], value[2]]
    )  # rotate the rectangle to follow transformation
    return points_transposed, value_tranps


def transpose2(arr, value):
    points_transposed = np.zeros_like(arr)  # initialize the array
    points_transposed[:, 1] = arr[:, 0]  # new x is the old y
    points_transposed[:, 0] = value[2] - arr[:, 1]  # new y is the width - the old x
    value_tranps = np.array(
        [value[0], value[1], value[3], value[2]]
    )  # change width and height to reflect the transformation
    return points_transposed, value_tranps


def no_transformation(arr, value):
    return arr, value


def transformation(key, coordinates_cropped, value):
    transform_filters = {
        "mirror-top": [transpose2, transpose2, hflip],
        "mirror-bottom": [hflip],
        "mirror-left": [transpose2, hflip],
        "mirror-right": [transpose1, hflip],
        "central": [no_transformation],
    }
    for f in transform_filters[key]:
        coordinates_cropped, value = f(coordinates_cropped, value)
    return coordinates_cropped


def get_coordinates_arena_and_transform(rectangles, frame):
    """
    Function that allows you to adjust bounding boxes of the arena and get the coordinates for improve calibration

    -args: rectangles for cropping, frame to display the bounding boxes
    -returns: dictionary with the coordinates of the bounding boxes fo the arena transforrmed accordingly to the transformation applied to that point of view

    original coordinates are:
    """

    coordinates = np.load(
        r"C:\Users\SNeurobiology\code\3d-setup\notebooks\right_coords.pkl",
        allow_pickle=True,
    )

    coordinates_arena = {}

    for key, value in rectangles.items():
        coordinates_arena[key] = np.asanyarray(
            get_coordinates(frame, str(key), coordinates)
        )
        # cropping
        y, x, h, w = value  # try it first

        coordinates_arena[key][:, 1] = coordinates_arena[key][:, 1] - x
        coordinates_arena[key][:, 0] = coordinates_arena[key][:, 0] - y
        print(f"print cropped coordinates for {key} \n {coordinates_arena[key]}")

        # transform them
        coordinates_arena[key] = transformation(key, coordinates_arena[key], value)
    return coordinates_arena


# add something for saving them during the video croppping


if __name__ == "__main__":
    # bounding_boxes_multiple_videos(path)
    parms = {
        "central": (240.0, 250.0, 600.0, 620.0),
        "mirror-top": (20.0, 250.0, 600.0, 220.0),
        "mirror-bottom": (860.0, 250.0, 600.0, 220.0),
        "mirror-left": (240.0, 30.0, 220.0, 620.0),
        "mirror-right": (240.0, 850.0, 220.0, 620.0),
    }

    video_path = r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240725\Calibration\multicam_video_2024-07-25T12_02_01.avi"
    cap = cv2.VideoCapture(video_path)
    frame_count = 100
    frame_interval = 100
    pooled_frames = []

    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()

        if not ret:
            print(f"Failed to read frame {i}")
            break
        pooled_frames.append(frame)

    frame_avg = np.mean(pooled_frames, axis=0)
    frame_avg = np.uint8(frame_avg)
    frame_avg = cv2.cvtColor(frame_avg, cv2.COLOR_BGR2GRAY)

    final_coords = get_coordinates_arena_and_transform(parms, frame_avg)
