# 3d-setup
Processing of data from mirror setup for 3D pose estimation

## (In progress) overall description of the pipeline
The starting data correpond of single videos acquired over multiple days for (single) animals abituating to the arena or hunting. For each day of experiment, a calibration movie was acquired with the standard checkerboard.

The overall data processing pipeline can be divided in four steps:
 - **0. Prepare the movies**: movies are initially cropped by a Napari GUI to set the masks for the different views. As long as the camera does not move, movies can be cropped using the same masks. We can potentially streamline/automatize this in the future with some dumb or smart heuristics to detect arena corners. Crucial: calibration movies have to be cropped in the same way as the mouse movies using that calibration!
   - state:  ✅ working, to improve:
      - automatic edge detection
      - clarify overall data structure better, to comply to current lab guidelines
 - **1. Label the movies**: this is done using two trained SLEAP models, one for the bottom view and one for the side views.
   - state: ✅ working, to improve:
      - label dataset with insects; I would get to 1000 annotated frames total
      - compare different models
 - **2. Create calibration and triangulate points**: calibration videos are processed with OpenCV functions and used to triangulate points in 3D
   - state:🚧 kind of working, in progress:
      - check new calibration videos with more camera triplets detections and better plexiglass calibration object
      - streamline the process
 - **3. GIMBAL denoising** use [GIMBAL](https://github.com/calebweinreb/gimbal/tree/main) to denoise the triangulation,
   - state ❌ still untested



## 0 Crop the movies to prepare them for `sleap` or `DLC`

For cropping the movies, there are two scripts to run in sequence: `0a_define_cropping.py` and `0b_process_videos.py`.

Run them from the `lab-env` environment or any other environment with the required dependencies installed (`opencv-python`, `numpy`, `napari`)

To run the scripts, start by activating the environment and navigate to the `3d-setup/complete_pipeline` folder:
    
```bash
conda activate lab-env
cd code/3d-setup/complete_pipeline
```

### 0.0 Run `0a_define_cropping.py`
This script defines the cropping parameters for a given video file. It allows the user to interactively define the cropping windows using Napari, saves the cropping parameters to a JSON file, and tests the cropping on the first 100 frames.

When you run it you have to specify the movie from which the cropping will be defined in this way:

```bash
python 0a_define_cropping.py /path/to/movie.avi
```

When you define the windows, please try if possible to just drag them and not change their size!

After running, the script will open a viewer from which you can check the results of the cropping.

The important output of the script is a `json` file that is saved in the same folder as the movie file, containing the `ffmpeg` commands for the cropping.

### 0.1 Run `0b_process_videos.py`
This script processes all the AVI video files in a specified folder using the cropping parameters defined in the JSON file created by `0a_define_cropping.py`.

To run it you have to specify the folder with the movies and the json file with the cropping parameters in this way:

```bash
python 0b_process_videos.py /path/containing/avi/files /path/to/cropping_parameters.json
```

When you run it, all files in the data folder will be processed, so make sure you define the correct one!
After processing, the cropped movies will be saved in the same folder as the original movies, with the suffix `_cropped` added to the file name.

## Dataset and model locataions

### Source data:
4 days baseline + 7 days hunting, 8 mice across all days; every day has a calibration video

#### Original data
baseline + hunting in: N:\SNeuroBiology_shared\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting

For every experiment day:
 - 1 calibration folder with AVI with data
 - 8 mice folders with a .avi file each (baselines: 10minutes; hunting: variable, early termination if jumping, longer times if they started to hunt)


#### Temporary location for data processing
D:\P05_3DRIG_YE-LP\e01_mouse_hunting\v04_mice-hunting: 4 days baseline + 7 days hunting, subset, 8 mice, ready for SLEAP refining which has not happened yet; to be cropped

videos already used for side view no cricket model training are in D:\Yadu\sleap_models\vids_for_sleap

From now on, we keep in D:\P05_3DRIG_YE-LP a partial mirror of  N:\SNeuroBiology_shared\P05_3DRIG_YE-LP with all the videos we will be working on.

#### Models
 - side view, high quality videos: available, but still to be trained with insect
 - bottom view: still missing (only DLC model)
BUT we do not fint the existing model anymore

Model location:
Trained models should live in: D:\SLEAP_models and back up constantly in N:\SNeuroBiology_shared\SLEAP_models.


Processing happens as fast as `ffmpeg` and multiprocessing allow, but it can still be slow for large files. The script will print the progress to the console.


### Notes on new files organization:

Root directory: nas_mirror

nas_mirror
    -almost_final_models/
        - dlc models for cricket, object and roach + mouse bottom-view
    - calibration/ 
        - cropping_params.json
        - 20250509/
            -calibration session data
    - DLC_final_mdoels/
        - cricket-bottom-model
        - mouse-bottom-model
        - mouse-side-model
        - roach_bottom-model
        - object-bottom-model
    - used_calibration/
        empty
    -M29, M30, M31/
        - cricket/
            -.csv file: timestamps (for frame rate?)
            - .avi: original video
            - 133050/
                - multicam_video_2025-05-07T14_11_04_cropped-v2_20250701121021: 
                    - cropped videos (central, sides) .mp4
                    - Tracking:
                        - Central: full.pickle, meta.pickle, snapshot.h5-> differences?
                        - Sides: full.pickle, meta.pickle, snapshot.h5
                        - Triangulations: datetime.h5 (how it has been obtained?)
                - /multicam_video_2025-05-07T14_11_04_cropped_20250528154623
                    - just central tracking
        - object:
            - Follows the same structure as cricketb
    - valid_crops_with_full_tracking.txt: list of cropped videos with valid params (for cropping or framerate?)
    - multicam_video_2025-05-07T10_12_11_20250528-153946.json: ffmpeg arguments for the cropping


