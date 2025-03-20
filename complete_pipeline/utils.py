import datetime
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import napari
import numpy as np
import toml


def crop_all_views(
    input_file,
    output_dir,
    cropping_specs_file,
    num_frames=None,
    verbose=False,
):
    """Crop all views of a video file using FFmpeg.

    Parameters
    ----------
    input_file : str or Path
        Path to the input video file.
    output_dir : str or Path
        Path to the output directory for the cropped videos.
    cropping_specs_file : str or Path
        Path to the JSON file containing the cropping specifications.
    num_frames : int, optional
        Number of frames to process, by default None means all frames
    """

    # assert json parameters:
    assert Path(
        cropping_specs_file
    ).exists(), f"File {cropping_specs_file} does not exist"
    assert (
        Path(cropping_specs_file).suffix == ".json"
    ), f"File {cropping_specs_file} is not a JSON file"

    assert Path(input_file).exists(), f"File {input_file} does not exist"

    with open(cropping_specs_file, "r") as f:
        cropping_specs = json.load(f)
        cropping_specs = cropping_specs[:-1]

    output_dir.mkdir(exist_ok=True)

    # Use ThreadPoolExecutor to run the tasks in parallel
    tnow = datetime.datetime.now()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for spec in cropping_specs:
            futures.append(
                executor.submit(
                    apply_transformations,
                    input_file,
                    output_dir,
                    spec["output_file_suffix"],
                    spec["filters"],
                    spec["ffmpeg_args"],
                    num_frames,
                    verbose,
                )
            )

        # Wait for all tasks to complete
        for future in futures:
            future.result()

    if verbose:
        print(f"Elapsed time: {datetime.datetime.now() - tnow}")

    return futures


def apply_transformations(
    input_file,
    output_dir,
    output_suffix,
    filters,
    ffmpeg_args,
    num_frames=None,
    verbose=True,
):
    """Apply transformations to a video file using FFmpeg.

    Parameters
    ----------
    input_file : str or Path
        Path to the input video file.
    output_file : str or Path
        Path to the output video file.
    filters : str
        FFmpeg video filters to apply.
    ffmpeg_args : list
        Additional FFmpeg arguments.
    num_frames : int, optional
        Number of frames to process, by default None means all frames
    """
    output_dir = Path(output_dir)

    ffmpeg_command = [
        "ffmpeg",
        "-i",
        str(input_file),  # Input file
        "-vf",
        filters,  # Video filters
    ]

    # Add additional FFmpeg arguments
    for key, value in ffmpeg_args.items():
        ffmpeg_command.extend([key, value])

    if num_frames is not None:
        ffmpeg_command.extend(
            ["-vframes", str(num_frames)]
        )  # Number of frames to process

    output_file = output_dir / f"{input_file.stem}_{output_suffix}.mp4"
    ffmpeg_command.append(
        str(output_file)
    )  # Output file

    # Run the FFmpeg command
    additional_kwargs = {}
    if not verbose:
        additional_kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    subprocess.run(ffmpeg_command, check=True, **additional_kwargs)

    return output_file


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


def annotate_cropping_windows(frame):
    """Annotate cropping windows on one frame using napari.

    Parameters
    ----------
    frame : np.ndarray
        A frame of the video.

    Returns
    -------
    final_rectangles_crops : dict
        A dictionary containing the final crop parameters for each view.
    """


    napari_viewer = napari.Viewer()
    napari_viewer.add_image(frame, name="Average frame", contrast_limits=[0, 255])

    # Central rectangle corner points
    img_height, img_width = frame.shape[:2]  # Get image dimensions

    # Central rectangle corner points
    corner_sw = (860, 250)
    corner_nw = (240, 250)
    corner_ne = (240, 850)
    corner_se = (860, 850)

    def_side = 220  # Default side length for mirror regions

    # Padding for video divisibility by 16
    padding_left_right = 2
    width_padding_tb = 4

    # **Step 1: Compute projection points on the image borders**
    # These are the central points for each side projected to the image boundary

    # Projected points on the top and bottom borders
    top_mid_left = (0, corner_nw[1])    # (y = 0, x of NW corner)
    top_mid_right = (0, corner_ne[1])   # (y = 0, x of NE corner)

    bottom_mid_left = (img_height - 1, corner_sw[1])  # (y = max, x of SW corner)
    bottom_mid_right = (img_height - 1, corner_se[1])  # (y = max, x of SE corner)

    # Projected points on the left and right borders
    left_mid_top = (corner_nw[0], 0)   # (y of NW corner, x = 0)
    left_mid_bottom = (corner_sw[0], 0)  # (y of SW corner, x = 0)

    right_mid_top = (corner_ne[0], img_width - 1)  # (y of NE corner, x = max)
    right_mid_bottom = (corner_se[0], img_width - 1)  # (y of SE corner, x = max)

    # **Step 2: Construct the rectangles so they extend toward the center**
    default_rectangles = {
        "central": [
            (corner_nw[0] - padding_left_right, corner_nw[1] - width_padding_tb),
            (corner_ne[0] - padding_left_right, corner_ne[1] + width_padding_tb),
            (corner_se[0] + padding_left_right, corner_se[1] + width_padding_tb),
            (corner_sw[0] + padding_left_right, corner_sw[1] - width_padding_tb),
        ],
        "mirror-top": [  # Corrected: Extends downward, keeping edges aligned
            (top_mid_left[0], top_mid_left[1]),  # Left projected point (top border)
            (top_mid_right[0], top_mid_right[1]),  # Right projected point (top border)
            (top_mid_right[0] + def_side, top_mid_right[1]),  # Extending downward
            (top_mid_left[0] + def_side, top_mid_left[1]),  # Extending downward
        ],
        "mirror-bottom": [  # Corrected: Extends upward, correctly aligned
            (corner_sw[0], corner_sw[1]),  # Aligned with SW (corrected)
            (corner_se[0], corner_se[1]),  # Aligned with SE (corrected)
            (bottom_mid_right[0], bottom_mid_right[1]),  # Right projected point (at bottom border)
            (bottom_mid_left[0], bottom_mid_left[1]),  # Left projected point (at bottom border)
        ],
        "mirror-left": [  # Extends rightward toward the central rectangle
            (left_mid_top[0], left_mid_top[1]),  # Top projected point (at left border)
            (corner_nw[0], corner_nw[1]),  # Aligned with NW (corrected)
            (corner_sw[0], corner_sw[1]),  # Aligned with SW (corrected)
            (left_mid_bottom[0], left_mid_bottom[1]),  # Bottom projected point (at left border)
        ],
        "mirror-right": [  # Extends leftward toward the central rectangle
            (corner_ne[0], corner_ne[1]),  # Aligned with NE (corrected)
            (right_mid_top[0], right_mid_top[1]),  # Top projected point (at right border)
            (right_mid_bottom[0], right_mid_bottom[1]),  # Bottom projected point (at right border)
            (corner_se[0], corner_se[1]),  # Aligned with SE (corrected)
        ],
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
            shape_type="rectangle",
            edge_color=default_colors[view_name],
            face_color="#ffffff00",
            edge_width=4,
            opacity=1,
            name=view_name,
        )
        napari_viewer.layers[view_name].mode = "select"

    napari.run()

    rectangles = {}
    for view_name in default_rectangles.keys():
        rectangles[view_name] = napari_viewer.layers[view_name].data[0].copy()

    final_rectangles_crops = _get_final_crop_params(rectangles)
    return final_rectangles_crops
