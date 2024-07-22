import json
from pathlib import Path
import datetime
from utils import crop_all_views, read_first_frame, annotate_cropping_windows
import napari

ffmpeg_args = {
        '-c:v': 'libx264',
        '-b:v': '30M',
        '-crf': '1',
        '-preset': 'veryfast',
        '-pix_fmt': 'gray',
        '-c:a': 'copy'
    }

transform_filters = {
        "mirror-top": "transpose=2,transpose=2,hflip",
        "mirror-bottom": "hflip",
        "mirror-left": "transpose=2,hflip",
        "mirror-right": "transpose=1,hflip",
        "central": ""  # No transformation
    }


def main(input_file):

    input_file = Path(input_file)

    tstamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    frame = read_first_frame(input_file)
    rectangles = annotate_cropping_windows(frame)

    test_output_dir = input_file.parent / f'test-output_{tstamp}'

    cropping_specs = []
    for rect_name, rect in rectangles.items():
        x, y, width, height = rect
        filters = f'crop={width}:{height}:{y}:{x},format=gray'
        if rect_name in transform_filters and transform_filters[rect_name]:
            filters += ',' + transform_filters[rect_name]
        
        cropping_specs.append({
            'transform': rect_name,
            'output_file': f'output-cropped-{rect_name}.avi',
            'filters': filters,
            'ffmpeg_args': ffmpeg_args
        })

    # Save cropping specs to a JSON file
    json_file = input_file.parent / f"{input_file.stem}_{tstamp}.json"
    with open(json_file, 'w') as f:
        json.dump(cropping_specs, f, indent=4)
    
    print(f"Cropping parameters saved to {json_file}")
    
    # Test the cropping parameters on the first 100 frames
    test_output_dir = input_file.parent / f'test-output_{tstamp}'
    crop_all_views(input_file, test_output_dir, json_file, num_frames=100, verbose=True)

    napari_viewer = napari.Viewer()
    # add first frame of original video:
    napari_viewer.add_image(frame, name="Original video", contrast_limits=[0, 255])

    # read and display the first frame of each cropped video using napari, and placing it on the side
    # of the original video one above the other:
    offset = 0
    for spec in cropping_specs:
        output_file = test_output_dir / spec['output_file']
        frame = read_first_frame(output_file)
        napari_viewer.add_image(frame, name=spec['output_file'], contrast_limits=[0, 255])
        napari_viewer.layers[spec['output_file']].translate = (offset, 1100)
        offset += frame.shape[0]

    napari.run()


if __name__ == "__main__":
    # nput_file = input("Enter the path to the input video file: ")
    import argparse

    parser = argparse.ArgumentParser(description="Define cropping parameters")
    parser.add_argument('file', type=str, help='File to define cropping parameters')

    args = parser.parse_args()
    
    main(Path(args.file))
