import subprocess
from concurrent.futures import ThreadPoolExecutor
import datetime


def crop_and_grayscale_video(input_file, output_file, crop_width, crop_height, crop_x, crop_y, transform, n_frames=None):
    transform_filters = {
        "mirror-top": "transpose=2,transpose=2,hflip",
        "mirror-bottom": "hflip",
        "mirror-left": "transpose=2,hflip",
        "mirror-right": "transpose=1,hflip",
        "central": ""  # No transformation
    }

    filters = f'crop={crop_width}:{crop_height}:{crop_x}:{crop_y}'
    
    if transform in transform_filters and transform_filters[transform]:
        filters += ',' + transform_filters[transform]

    # Construct the FFmpeg command
    ffmpeg_command = [
        'ffmpeg',
        '-i', str(input_file),  # Input file
        '-vf', filters,  # Crop filter
        # '-vframes', str(frames),  # Number of frames to process
        '-c:v', 'libx264',  # Re-encode video with H.264 codec
        '-b:v', '30M',  # Re-encode video with H.264 codec
        '-crf', '1',  # Constant Rate Factor for quality (lower means better quality)
        '-preset', 'fast',  # Preset for encoding speed vs. quality
        '-pix_fmt', 'gray',  # Use grayscale pixel format
        '-c:a', 'copy',  # Copy the original audio codec
        output_file  # Output file
    ]
    if n_frames:
        ffmpeg_command.insert(-1, '-vframes')
        ffmpeg_command.insert(-1, str(n_frames))
    
    # Run the FFmpeg command
    subprocess.run(ffmpeg_command, check=True)


def crop_movie(input_file, rectangles_dict):
    cropping_specs = []
    for rect_name, rect in rectangles_dict.items():
        x, y, width, height = rect
        cropping_specs.append({
            'transform': rect_name,
            'output_file': f'output-cropped-{rect_name}',
            'crop_width': width,
            'crop_height': height,
            'crop_x': y,
            'crop_y': x
        })
    now = datetime.datetime.now()

    # Ensure the output directory exists
    for spec in cropping_specs:
        print(spec)
        spec['output_file'] = input_file.parent / f"{spec['output_file']}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.avi"

    # Use ThreadPoolExecutor to run the tasks in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for spec in cropping_specs:
            futures.append(executor.submit(crop_and_grayscale_video, input_file, spec['output_file'], 
                                           spec['crop_width'], spec['crop_height'], spec['crop_x'], spec['crop_y'], spec['transform']))
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()

    return cropping_specs
