# 3d-setup
Processing of data from mirror setup for 3D pose estimation


## Crop the movies to prepare them for `sleap` or `DLC`

For cropping the movies, there are two scripts to run in sequence: `0a_define_cropping.py` and `0b_process_videos.py`.

Run them from the `lab-env` environment or any other environment with the required dependencies installed (`opencv-python`, `numpy`, `napari`)

To run the scripts, start by activating the environment and navigate to the `3d-setup/complete_pipeline` folder:
    
```bash
conda activate lab-env
cd code/3d-setup/complete_pipeline
```

### 0. Run `0a_define_cropping.py`
This script defines the cropping parameters for a given video file. It allows the user to interactively define the cropping windows using Napari, saves the cropping parameters to a JSON file, and tests the cropping on the first 100 frames.

When you run it you have to specify the movie from which the cropping will be defined in this way:

```bash
python 0a_define_cropping.py /path/to/movie.avi
```

When you define the windows, please try if possible to just drag them and not change their size!

After running, the script will open a viewer from which you can check the results of the cropping.

The important output of the script is a `json` file that is saved in the same folder as the movie file, containing the `ffmpeg` commands for the cropping.

### 1. Run `0b_process_videos.py`
This script processes all the AVI video files in a specified folder using the cropping parameters defined in the JSON file created by `0a_define_cropping.py`.

To run it you have to specify the folder with the movies and the json file with the cropping parameters in this way:

```bash
python 0b_process_videos.py /path/containing/avi/files /path/to/cropping_parameters.json
```

When you run it, all files in the data folder will be processed, so make sure you define the correct one!
After processing, the cropped movies will be saved in the same folder as the original movies, with the suffix `_cropped` added to the file name.

Processing happens as fast as `ffmpeg` and multiprocessing allow, but it can still be slow for large files. The script will print the progress to the console.
