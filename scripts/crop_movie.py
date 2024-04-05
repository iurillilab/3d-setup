from tqdm import tqdm
import cv2
from pathlib import Path
import numpy as np


def crop_and_save_views(input_file, output_prefix, views_coords, contrast_lims):
    """
    Crop and save views from an AVI movie.

    Args:
    - input_file: Path to the input AVI movie.
    - output_prefix: Prefix for the output file names.
    - views_coords: A dictionary with the coordinates for each view:
        {
            'central': (x, y, width, height),
            'left': (x, y, width, height),
            'right': (x, y, width, height),
            'top': (x, y, width, height),
            'bottom': (x, y, width, height)
        }
    - contrast_lims: A dictionary with the contrast limits for each view
    """
    # Open the video file
    cap = cv2.VideoCapture(input_file)
    
    # Extract video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Use same codec as input video file:
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    outputs = {}
    for view in views_coords:
        x, y, w, h = views_coords[view]

        # If left or right, swap width and height:
        if view in ['left', 'right']:
            w, h = h, w

        outputs[view] = cv2.VideoWriter(f'{output_prefix}_{view}.avi', fourcc, fps, (w, h))
    
    # Add an output consisting of all views combined, center on the left and for others stacked one on top of the othe on the right:
    left_view = 'central'
    right_views = ['left', 'right', 'top', 'bottom']
    w_left, h_left = views_coords[left_view][2], views_coords[left_view][3]
    h_right, w_right = views_coords[right_views[0]][2], views_coords[right_views[0]][3]
    w_right_all = w_right
    h_right_all = h_right * len(right_views)
    w_all = w_left + w_right_all
    h_all = max(h_left, h_right_all)
    outputs['all'] = cv2.VideoWriter(f'{output_prefix}_all.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (w_all, h_all))

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        i += 1
        
        # combine views addind them to collage frame:
        collage = np.zeros((h_all, w_all, 3), dtype=np.uint8)

        # Crop each view and write to file
        for view, (x, y, w, h) in views_coords.items():
            cropped_view = frame[y:y+h, x:x+w]
            if contrast_lims[view] is not None:
                cropped_view = cv2.convertScaleAbs(cropped_view, alpha=1.0, beta=contrast_lims[view][0])
                cropped_view = cv2.convertScaleAbs(cropped_view, alpha=255.0/(contrast_lims[view][1] - contrast_lims[view][0]), beta=-contrast_lims[view][0]*255.0/(contrast_lims[view][1] - contrast_lims[view][0]))

            # print(view, h_right, w_left)
            # If central, add to left in collage:
            if view == 'central':
                collage[:h_left, :w_left] = cropped_view

            # If bottom, flip the image:
            if view == 'top':
                cropped_view = cv2.flip(cropped_view, 0)
                collage[:h_right, w_left:] = cropped_view

            # If left, rotate the image:
            if view == 'left':
                cropped_view = cv2.rotate(cropped_view, cv2.ROTATE_90_COUNTERCLOCKWISE)
                collage[h_right:h_right*2, w_left:] = cropped_view
            
            # If right, rotate the image:
            if view == 'right':
                cropped_view = cv2.rotate(cropped_view, cv2.ROTATE_90_CLOCKWISE)
                collage[h_right*2:h_right*3, w_left:] = cropped_view
            
            if view == 'bottom':
                collage[h_right*3:, w_left:] = cropped_view

            outputs[view].write(cropped_view)

        # Combine views and write to file
        if i > 1000 and i < 1200:
            outputs['all'].write(collage)
        
    # Release everything
    cap.release()
    for output in outputs.values():
        output.release()


# Example usage
input_folder = Path('/Users/vigji/Desktop')
pattern = 'Basler*20240301*.avi'

left_line = 420
right_line = 1020
top_line = 220
bottom_line = 830
width = right_line - left_line
height = bottom_line - top_line
cropped_height = 250
cropped_width = 600

contrast_lims = {"central": (0, 100), 
                 "left": (0, 30), 
                 "right": (0, 30), 
                 "top": (0, 30), 
                 "bottom": (0, 30)}


for input_file in tqdm(list(input_folder.glob(pattern))):
    output_dir = input_file.parent / input_file.stem
    output_dir.mkdir(exist_ok=True)
    output_prefix = str(output_dir / (input_file.stem + '_crop_'))

    # Open the video file
    cap = cv2.VideoCapture(str(input_file))

    # Extract video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    views_coords = {
        'central': (left_line, top_line, width, height, 
                    ),  # Example coordinates
        "left": (170, top_line, cropped_height, cropped_width),
        "right": (right_line, top_line, cropped_height,
                cropped_width),
        "top": (left_line, 0, cropped_width, cropped_height),
        "bottom": (left_line, bottom_line, cropped_width, cropped_height),
    }
    # print(views_coords)
    cap.release()

    crop_and_save_views(str(input_file), output_prefix, views_coords, contrast_lims)
