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
    Recently fixed to 1) remove extension from output file name, 2) avoid issue with changing frames numbers.

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
        "-copyts", "-vsync", "0",
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


def make_rectangle_divisible_by_16(rect):
    """
    Takes a rectangle as 4 corner points and adjusts bottom/right edges
    inward so width and height are divisible by 16.
    Assumes corners are ordered clockwise starting from top-left.
    """
    # Extract y and x values
    ys = [pt[0] for pt in rect]
    xs = [pt[1] for pt in rect]

    top = min(ys)
    bottom = max(ys)
    left = min(xs)
    right = max(xs)

    height = bottom - top
    width = right - left

    # Adjust height
    if height % 16 != 0:
        bottom -= height % 16
    # Adjust width
    if width % 16 != 0:
        right -= width % 16

    # Rebuild the adjusted rectangle (clockwise from top-left)
    return [
        (top, left),
        (top, right),
        (bottom, right),
        (bottom, left),
    ]


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
# TODO: make sure that each rect is dividible by 16

    napari_viewer = napari.Viewer()
    napari_viewer.add_image(frame, name="Average frame", contrast_limits=[0, 255])
    
    img_height, img_width = frame.shape[:2]

    corner_sw = (860, 250)
    corner_nw = (240, 250)
    corner_ne = (240, 850)
    corner_se = (860, 850)
    def_side = 220

    # //TODO make the padding automatic

    # padding is used to make videos divisible by 16
    padding_left_right = 2
    width_padding_tb = 4

    default_rectangles = {
        "central": [
            (corner_nw[0] - padding_left_right, corner_nw[1] - width_padding_tb),
            (corner_ne[0] - padding_left_right, corner_ne[1] + width_padding_tb),
            (corner_se[0] + padding_left_right, corner_se[1] + width_padding_tb),
            (corner_sw[0] + padding_left_right, corner_sw[1] - width_padding_tb),
        ],

        "mirror-top": [
            (
                0,
                corner_nw[1] - width_padding_tb,
            ),
            (
                0,
                corner_ne[1] + width_padding_tb,
            ),
            (def_side + width_padding_tb, corner_ne[1] + width_padding_tb),
            (def_side + width_padding_tb, corner_nw[1] - width_padding_tb),
        ],

         "mirror-bottom": [
            (img_height - def_side, corner_sw[1] - width_padding_tb),
            (img_height - def_side, corner_se[1] + width_padding_tb),
            (
                img_height,
                corner_se[1] + width_padding_tb,
            ),
            (
                img_height,
                corner_sw[1] - width_padding_tb,
            ),
        ],
        "mirror-left": [
            (
                corner_nw[0] - padding_left_right,
                0,
            ),
            (corner_nw[0] - padding_left_right, def_side),
            (corner_sw[0] + padding_left_right, def_side),
            (
                corner_sw[0] + padding_left_right,
                0,
            ),
        ],

        "mirror-right": [
            (corner_ne[0] - padding_left_right, img_width - def_side),
            (
                corner_ne[0] - padding_left_right,
                img_width,
            ),
            (
                corner_se[0] + padding_left_right,
                img_width,
            ),
            (corner_se[0] + padding_left_right, img_width - def_side),
        ],
    }

    # After constructing default_rectangles, but before adding to napari viewer:

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
