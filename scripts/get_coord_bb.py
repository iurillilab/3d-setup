from pathlib import Path
import napari
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


# dispaly user interface to select the directory, it expect a directory with the 5 videos cropped in mp4 format (it can be changed, but mp4 works better with sleaap)
root = tk.Tk()
root.withdraw()

directory_path = filedialog.askdirectory(title='Select the directory where the videos are located')

if directory_path:
    path = Path(directory_path)
else:
    print("No directory selected")


def bounding_boxes_multiple_videos(directory_path):
    '''Function that iterates through the cropped videos and select the bounding boxes for each video
    -args: path to the directory where the videos are located

    - it itereates through all the videos cropped and it runs the get_coordinates_bounding_boxes function
    '''

    for video in path.glob("*.mp4"):
        print(video)
        get_coordinates_bounding_boxes(str(video))


def get_coordinates_bounding_boxes(video_path):
    '''
    Function that allows you to select with napari the bounding boxes of the arena and get the coordinates for improve calibration

    -args:
        video_path: str, path to the video file
    -returns:
        list of tuples with the coordinates of the bounding boxes

    '''

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

    viewer = napari.Viewer()
    viewer.add_image(frame_avg, name="Average frame", contrast_limits=[0, 255])

    points_layer = viewer.add_points(np.empty((0, 2)), size=2, face_color='red')
    points_layer.mode = 'add'

    def get_points_coordinates():
        return points_layer.data

    napari.run()
    print(get_points_coordinates())
    return get_points_coordinates()
    cap.release()


if __name__ == "__main__":
    bounding_boxes_multiple_videos(path)