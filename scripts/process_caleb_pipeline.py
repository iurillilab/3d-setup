from multicam_calibration.detection import process_video, run_calibration_detection, detect_chessboard
from pathlib import Path


input_folder = Path(r'C:\Users\Administrator\Desktop\rig_test_videos')
all_crop_files = list(input_folder.glob('checkerboard*/*/*.avi'))
assert len(all_crop_files) > 0, "No video files found in the specified folder"

# for crop_file in all_crop_files:
#     process_video(crop_file, output_folder='/Users/vigji/Desktop/crops_output')

all_videos = [str(crop_file) for crop_file in all_crop_files]
# print(all_videos)
if __name__ == '__main__':
    run_calibration_detection(all_videos, detect_chessboard, 
                          detection_options=dict(board_shape=(5, 7)), n_workers=12)
    
