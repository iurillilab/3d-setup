import datetime
import json
from pathlib import Path

import napari
import numpy as np
from matplotlib import cm
from utils import annotate_cropping_windows, crop_all_views, read_first_frame

possible_para = [
    "ffmpeg",
    "-y",
    "-i",
    "-pix_fmt",
    "yuv420p",
    "-preset",
    "superfast",
    "-crf",
    "23",
]

ffmpeg_args = {
    "-c:v": "libx264",
    "-b:v": "30M",
    "-crf": "23",
    "-preset": "superfast",
    "-pix_fmt": "yuv420p",
    "-c:a": "copy",
}

transform_filters = {
    "mirror-top": "transpose=2,transpose=2,hflip",
    "mirror-bottom": "hflip",
    "mirror-left": "transpose=2,hflip",
    "mirror-right": "transpose=1,hflip",
    "central": "",  # No transformation
}


def get_coordinates(frame, name: str, coordinates, value):
    viewer = napari.Viewer()
    cmap = cm.get_cmap("hsv", len(coordinates[name]))
    colors = [cmap(i) for i in range(len(coordinates[name]))]
    y, x, widht, height = value
    top_left = [y, x]
    top_right = [y, x + widht]
    bottom_right = [y + height, x + widht]
    bottom_left = [y + height, x]
    rectangle = np.array([top_left, top_right, bottom_right, bottom_left])

    rect = viewer.add_shapes(
        rectangle,
        shape_type="polygon",
        edge_color="red", 
        face_color="white",)


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
        r"D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting\20240803\M4\140337\right_coords.pkl",
        allow_pickle=True,
    )

    coordinates_arena = {}

    for key, value in rectangles.items():
        coordinates_arena[key] = np.asanyarray(
            get_coordinates(frame, str(key), coordinates, value)
        )
        # cropping
        y, x, h, w = value  # try it first

        coordinates_arena[key][:, 1] = coordinates_arena[key][:, 1] - x
        coordinates_arena[key][:, 0] = coordinates_arena[key][:, 0] - y
        print(f"print cropped coordinates for {key} \n {coordinates_arena[key]}")

        # transform them
        coordinates_arena[key] = transformation(key, coordinates_arena[key], value)
    return coordinates_arena


def check_rectangles_fit_frame(rectangles, frame):
    exeding_rectangles = []
    for key, (y, x, width, height) in rectangles.items():
        if x + width > frame.shape[1] or y + height > frame.shape[0]:
            exeding_rectangles.append(key)
    if exeding_rectangles:
        raise ValueError(
            f"Warning: the following rectangles exceed the frame dimensions: {exeding_rectangles}"
        )
        #return True
    #return False

def get_right_rectangles(frame):
    while True:
        try:
            rectangles = annotate_cropping_windows(frame)
            check_rectangles_fit_frame(rectangles, frame)
            break
        except ValueError:
            print("Rectangles exceed the frame dimensions")
    return rectangles

    


def main(input_file):
    input_file = Path(input_file)

    tstamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    frame = read_first_frame(input_file)
    #rectangles = annotate_cropping_windows(frame)
    rectangles = get_right_rectangles(frame)
    #add warning if any of the rectangles exceed the frame dimensions


    # insert function to
    coordinates_transfrmed = get_coordinates_arena_and_transform(rectangles, frame)

    test_output_dir = input_file.parent / f"test-output_{tstamp}"

    cropping_specs = []
    for rect_name, rect in rectangles.items():
        x, y, width, height = rect
        filters = f"crop={width}:{height}:{y}:{x},format=gray"
        if rect_name in transform_filters and transform_filters[rect_name]:
            filters += "," + transform_filters[rect_name]

        cropping_specs.append(
            {
                "transform": rect_name,
                "output_file_suffix": f"{rect_name}.avi",  # try to substitue .mp4
                "filters": filters,
                "ffmpeg_args": ffmpeg_args,
            }
        )
    for key, value in coordinates_transfrmed.items():
        coordinates_transfrmed[key] = value.tolist()
    cropping_specs.append({"points_coordinate": coordinates_transfrmed})
    # Save cropping specs to a JSON file
    json_file = input_file.parent / f"{input_file.stem}_{tstamp}.json"
    with open(json_file, "w") as f:
        json.dump(cropping_specs, f, indent=4)

    print(f"Cropping parameters saved to {json_file}")

    # Test the cropping parameters on the first 100 frames
    test_output_dir = input_file.parent / f"test-output_{tstamp}"
    crop_all_views(input_file, test_output_dir, json_file, num_frames=100, verbose=True)

    napari_viewer = napari.Viewer()
    # add first frame of original video:
    napari_viewer.add_image(frame, name="Original video", contrast_limits=[0, 255])

    # read and display the first frame of each cropped video using napari, and placing it on the side
    # of the original video one above the other:
    offset = 0
    cropping_specs = cropping_specs[:-1]
    for spec in cropping_specs:
        output_file = next(test_output_dir.glob(f"*{spec['output_file_suffix']}*"))
        frame = read_first_frame(output_file)
        napari_viewer.add_image(
            frame, name=spec["output_file_suffix"], contrast_limits=[0, 255]
        )
        napari_viewer.layers[spec["output_file_suffix"]].translate = (offset, 1100)
        offset += frame.shape[0]

    napari.run()


if __name__ == "__main__":
    # nput_file = input("Enter the path to the input video file: ")
    import argparse

    parser = argparse.ArgumentParser(description="Define cropping parameters")
    parser.add_argument("file", type=str, help="File to define cropping parameters")

    args = parser.parse_args()

    main(Path(args.file))
