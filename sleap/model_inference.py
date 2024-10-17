import pathlib
import subprocess
import os
import numpy as np
from datetime import datetime
import re
from pathlib import Path



# function to iterate through all the mice videos and select the paths of all videos except the bottom one. 

def get_video_paths(main_path):
    '''
    Input: main_path: str: path to the main folder containing all the videos
    It selects the video output form scrip for cropping them, (.avi.avi), and it exlucdes the central one.
    Output: video_paths: list: list of paths to all the videos in the main folder
    '''
    video_paths = []
    p = pathlib.Path(main_path)


    for video in p.rglob('*'):
        if 'calibration' in [parent.name for parent in video.parents]:
            continue
        if (
            video.is_file()
            and video.name.endswith('avi.avi')
            and 'central' not in video.name
        ):
            video_paths.append(str(video))
    return video_paths

#function that takes video paths and encode them and convert them into mp4 format to run inference using ffmpeg
def encode_and_convert(video_paths):
    '''
    Input: video_paths: list: list of paths to all the videos in the main folder
    It encodes the video to mp4 format using ffmpeg and ir returns the output path of the encoded and coverted video. 
    '''
    encoded_paths = []
    for video in video_paths:
        
        try:
            output_path = str(Path(video).with_name(Path(video).stem.replace('.avi.avi', '')+ 'encoded.mp4'))
            #output_path = video.replace('.avi.avi', 'encoded.mp4')
            print(f'Encoding video: {video}')

            subprocess.run(['ffmpeg', '-y', '-i', video, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'superfast', '-crf', '23', output_path], check=True)
            encoded_paths.append(output_path)
            print(f'Encoded video: {output_path}')
        except subprocess.CalledProcessError as e:
          print(f"Error running comman {encoded_paths}, error: {e}")
    return encoded_paths


# function that iterates thorugh all the video paths and run model inference on them and save the results in a specific folder.
def run_inference(enc_vid):
    '''
    Input: video_paths: list: list of paths to all the videos in the main folder
    It runs the model inference on the encoded videos and saves the results in a specific folder. 
    '''
    for video in enc_vid:
        output_folder = video.replace('.avi.avi', 'predictions.slp')
        print(f'Running inference on video: {video}')
        try:

            subprocess.run(['sleap-track', '-m', model,  '-o', output_folder,  video], check=True)
            print(f'Inference results saved to: {output_folder}')
        except subprocess.CalledProcessError as e:
            print(f"Error running comman {output_folder}, error: {e}")


if __name__ == '__main__':
    GEN_VIDEO_PATH = r'D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting'
    model = r"D:\SLEAP_models\test\models\241007_120850.single_instance.n=500"


    video_paths = get_video_paths(GEN_VIDEO_PATH)
    encoded_paths = encode_and_convert(video_paths)
    run_inference(encoded_paths)
    print('Inference done!')