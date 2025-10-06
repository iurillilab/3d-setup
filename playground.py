print("Script is running...")

import cv2

def read_first_frame(input_file):
    print(f"Attempting to open video file: {input_file}")
    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise ValueError(f"Failed to open the video file {input_file}")
    
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Failed to read the first frame of the video")
    
    cap.release()
    return frame

if __name__ == "__main__":
    input_file = "/mnt/e/CHOMP/20250729/M32/101656/multicam_video_2025-07-29T10_20_43.avi"  # Use an actual file path here
    read_first_frame(input_file)