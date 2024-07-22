from multicam_calibration.detection import process_video, run_calibration_detection, detect_chessboard
from pathlib import Path

from numpy import mat


#input_folder = Path(r'/Users/vigji/Desktop/dest_dir/19042024/Calibration/Basler_acA1440-220um__40075240__20240419_100853427_cropped')
input_folder = next(Path(r"/Users/vigji/Desktop/dest_dir/19042024").glob("Calibration/Basler_acA1440-220um__*_cropped"))

all_crop_files = list(input_folder.glob('*.mp4'))
assert len(all_crop_files) > 0, "No video files found in the specified folder"

# for crop_file in all_crop_files:
#     process_video(crop_file, output_folder='/Users/vigji/Desktop/crops_output')

all_videos = [str(crop_file) for crop_file in all_crop_files]
# print(all_videos)
if __name__ == '__main__':
    run_calibration_detection(all_videos, detect_chessboard, 
                          detection_options=dict(board_shape=(5, 7),
                                                 match_score_min_diff=0.2,
                                                 match_score_min=0.7,
), 
                          n_workers=12)
    
