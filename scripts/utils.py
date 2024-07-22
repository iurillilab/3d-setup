import subprocess
from pathlib import Path
import cv2
import numpy as np
import napari
from concurrent.futures import ThreadPoolExecutor
import json


def crop_all_views(input_file, output_dir, cropping_specs_file, num_frames=None):

    with open(cropping_specs_file, 'r') as f:
        cropping_specs = json.load(f)

    output_dir.mkdir(exist_ok=True)
    
    # Use ThreadPoolExecutor to run the tasks in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for spec in cropping_specs:
            output_file = output_dir / spec['output_file']
            futures.append(executor.submit(apply_transformations, input_file, output_file, 
                                           spec['filters'], spec['ffmpeg_args'], num_frames))
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()


def apply_transformations(input_file, output_file, filters, ffmpeg_args, num_frames=None):
    ffmpeg_command = [
        'ffmpeg',
        '-i', str(input_file),  # Input file
        '-vf', filters,  # Video filters
    ]

    # Add additional FFmpeg arguments
    for key, value in ffmpeg_args.items():
        ffmpeg_command.extend([key, value])
    
    if num_frames is not None:
        ffmpeg_command.extend(['-vframes', str(num_frames)])  # Number of frames to process
    
    ffmpeg_command.append(str(output_file))  # Output file
    
    # Run the FFmpeg command
    subprocess.run(ffmpeg_command, check=True)


def read_first_frame(input_file):
    cap = cv2.VideoCapture(str(input_file))
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Failed to read the first frame of the video")
    cap.release()
    return frame

def _get_width_height(rect):
    width = rect[1][1] - rect[0][1]
    height = rect[2][0] - rect[1][0]
    return width, height

def _get_final_crop_params(napari_rectangles):
    final_rectangles_crops = {}
    for view_name, rect in napari_rectangles.items():
        x, y = rect[0]
        width, height = _get_width_height(rect)
        final_rectangles_crops[view_name] = (x, y, width, height)
    return final_rectangles_crops

def annotate_cropping_windows(avg_frame):
    napari_viewer = napari.Viewer()
    napari_viewer.add_image(avg_frame, name="Average frame", contrast_limits=[0, 255])

    corner_sw = (860, 250)
    corner_nw = (240, 250)
    corner_ne = (240, 850)
    corner_se = (860, 850)
    def_side = 220

    default_rectangles = {
        "central": [corner_nw, corner_ne, corner_se, corner_sw],
        "mirror-top": [(corner_nw[0] - def_side, corner_nw[1]), (corner_ne[0] - def_side, corner_ne[1]), corner_ne, corner_nw],
        "mirror-bottom": [corner_sw, corner_se, (corner_se[0] + def_side, corner_se[1]), (corner_sw[0] + def_side, corner_sw[1])],
        "mirror-left": [(corner_nw[0], corner_nw[1] - def_side), corner_nw, corner_sw, (corner_sw[0], corner_sw[1] - def_side)],
        "mirror-right": [corner_ne, (corner_ne[0], corner_ne[1] + def_side), (corner_se[0], corner_se[1] + def_side), corner_se],
    }

    default_colors = {
        "central": "red",
        "mirror-top": "blue",
        "mirror-bottom": "green",
        "mirror-left": "yellow",
        "mirror-right": "purple",
    }

    for view_name, rect in default_rectangles.items():
        napari_viewer.add_shapes(
            data=np.array([rect]),
            shape_type='rectangle',
            edge_color=default_colors[view_name],
            face_color='#ffffff00',
            edge_width=4,
            opacity=1,
            name=view_name
        )
        napari_viewer.layers[view_name].mode = "select"

    # napari.run()

    rectangles = {}
    for view_name in default_rectangles.keys():
        rectangles[view_name] = napari_viewer.layers[view_name].data[0].copy()
    
    final_rectangles_crops = _get_final_crop_params(rectangles)
    return final_rectangles_crops
