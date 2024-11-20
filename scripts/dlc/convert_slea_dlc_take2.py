# %%
import argparse
import io
import json
import yaml
import time
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd

import PIL.Image as Image
import cv2
from sleap.io.dataset import load_file
from matplotlib import pyplot as plt


# def extract_frames_from_pkg_slp(file_path, base_output_dir, project_name='converted_project', scorer='me'):
slp_file = Path("/Users/vigji/Downloads/labels.v001.pkg.slp")
dlc_dir = Path("/Users/vigji/Desktop/converted_from_sleap/test7")
project_name = "test4"
scorer = "me"
# %%
slp_labels = load_file(str(slp_file))
# dir(slp_labels
frame = slp_labels.labeled_frames[0]
dir(frame)
frame.frame_idx, frame.n_user_instances, frame.n_predicted_instances

# %%
selected_frames = []
for frame in slp_labels.labeled_frames:
    if frame.n_user_instances > 0:
        selected_frames.append(frame)

# %%
frame = selected_frames[-1]

# get the image:
img = np.array(frame.image)

# get the user defined instance:
instance = frame.user_instances[0]
points = instance.points
x_arr = np.array([p.x for p in points])
y_arr = np.array([p.y for p in points])
visible_arr = np.array([p.visible for p in points])
x_arr[~visible_arr] = np.nan
y_arr[~visible_arr] = np.nan
bodyparts = [node.name for node in instance.skeleton.nodes]

# plot bodyparts with names on the frame:
plt.figure()
plt.imshow(img)
plt.scatter(x_arr, y_arr, c='red', s=4)
for i, bp in enumerate(bodyparts):
    plt.text(x_arr[i], y_arr[i], bp, color='red', fontsize=6)

# %%

# %%
plt.figure()
plt.imshow(img)
coords = frame.numpy()[0, :, :]
plt.scatter(coords[:, 0], coords[:, 1], c='red', s=4)
# print(len(img), img[0].shape)
# instances = frame.instances
# img.shape, instances.shape
# %%
# %%
base_output_dir = dlc_dir
file_path = slp_file

#create directories
base_output_dir = Path(base_output_dir)
base_output_dir.mkdir(parents=True, exist_ok=True)
(base_output_dir / "labeled-data").mkdir(exist_ok=True)
(base_output_dir / "videos").mkdir(exist_ok=True)

config = {'Task': project_name,
            'scorer': scorer,
            'date': time.strftime("%Y-%m-%d"),
            'identity': None,
            'project_path': base_output_dir,
            'engine': 'pytorch',
            'video_sets': {},
            'start': 0,
            'stop': 1,
            'numframes2pick': 20,
            'skeleton_color': 'black',
            'pcutoff': 0.6,
            'dotsize': 12,
            'alphavalue': 0.6,
            'colormap': 'rainbow',
            'TrainingFraction': [0.95],
            'iteration': 0,
            'default_net_type': 'resnet_50',
            'default_augmenter': 'default',
            'snapshotindex': -1,
            'detector_snapshotindex': -1,
            'batch_size': 8,
            'detector_batch_size': 1,
            'cropping': False,
            'x1': 0,
            'x2': 640,
            'y1': 277,
            'y2': 624,
            'corner2move2': [50, 50],
            'move2corner': True,
            'SuperAnimalConversionTables': None
            }
    
# %%
#parse .slp file
with h5py.File(file_path, 'r') as hdf_file:
    # Identify video names:
    video_names = {}
    for video_group_name in hdf_file.keys():
        if video_group_name.startswith('video') and video_group_name[5:].isdigit():
            print(video_group_name)
            source_video_path = f'{video_group_name}/source_video'

            if source_video_path in hdf_file:
                source_video_json = hdf_file[source_video_path].attrs['json']
                source_video_dict = json.loads(source_video_json)
                video_filename = source_video_dict['backend']['filename']
                video_names[video_group_name] = video_filename
                

# %%
with h5py.File(file_path, 'r') as hdf_file:
    # Extract and save images for each video
    for video_group, video_filename in list(video_names.items())[:1]:
        # video_group, video_filename = list(video_names.items())[0]
        print("processing ", video_filename)
        data_frames = []
        scorer_row, bodyparts_row, coords_row = None, None, None
        output_dir = base_output_dir / "labeled-data" / Path(video_filename).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # extract labeled frames and save them in a separate directory for each video
        if video_group in hdf_file and 'video' in hdf_file[video_group]:
            video_data = hdf_file[f'{video_group}/video'][:]
            frame_numbers = hdf_file[f'{video_group}/frame_numbers'][:]
            frame_names = []

            if len(frame_numbers) > 200:
                print('Too many frames, skipping video {}, {}'.format(video_group, frame_numbers))
                print(f"No labeled data found for video {video_filename}")
                # Empty folder content and unlink folder
                for file in output_dir.glob('*'):
                    file.unlink()
                output_dir.rmdir()
                continue

            for i, (img_bytes, frame_number) in tqdm(list(enumerate(zip(video_data, frame_numbers)))):
                if len(frame_numbers) > 200:
                    print('Too many frames, skipping video {}, {}'.format(video_group, frame_numbers))
                    print(f"No labeled data found for video {video_filename}")
                    # Empty folder content and unlink folder
                    for file in output_dir.glob('*'):
                        file.unlink()
                    output_dir.rmdir()
                    continue
                img = Image.open(io.BytesIO(np.array(img_bytes, dtype=np.uint8)))
                img = np.array(img)
                if i==0:
                    video_path = base_output_dir / 'videos' / Path(video_names[video_group]).name
                    config['video_sets'][str(video_path)] = {'crop': f'0, {img.shape[1]}, 0, {img.shape[0]}'}
                frame_name = f"img{str(frame_number).zfill(8)}.png"
                cv2.imwrite(str(output_dir / frame_name), img)
                frame_names.append(frame_name)
                # print(f"Saved frame {frame_number} as {frame_name}")
         
# %%     
        # extract coordinates and save them in a separate directory for each video
        if video_group in hdf_file and 'frames' in hdf_file:
            frames_dataset = hdf_file['frames']
            frame_references = {
                frame['frame_id']: frame['frame_idx']
                for frame in frames_dataset
                if frame['video'] == int(video_group.replace('video', ''))
            }

            if len(frame_numbers) > 200:
                print('Too many frames, skipping video {}, {}'.format(video_group, frame_numbers))
                print(f"No labeled data found for video {video_filename}")
                # Empty folder content and unlink folder
                for file in output_dir.glob('*'):
                    file.unlink()
                output_dir.rmdir()
                continue


            # Extract instances and points
            points_dataset = hdf_file['points']
            instances_dataset = hdf_file['instances']
            # print(instances_dataset)
            
            data = []
            for frame_id in tqdm(frame_references.keys()):
                for i in range(len(instances_dataset)):
                    try:
                        if frame_id==instances_dataset[i]['frame_id']:
                            point_id_start = instances_dataset[i+1]['point_id_start']
                            point_id_end = instances_dataset[i+1]['point_id_end']
                            break
                    except IndexError:
                        continue

                points = points_dataset[point_id_start:point_id_end]

                keypoints_flat = []
                for kp in points:
                    x, y, vis = kp['x'], kp['y'], kp['visible']
                    if np.isnan(x) or np.isnan(y) or vis==False:
                        x, y = None, None
                    keypoints_flat.extend([x, y])

                frame_idx = frame_references[frame_id]
                if len(keypoints_flat) > 0:
                    data.append([frame_idx] + keypoints_flat)

            if len(data) == 0:
                print(f"No labeled data found for video {video_filename}")
                # Empty folder content and unlink folder
                for file in output_dir.glob('*'):
                    file.unlink()
                output_dir.rmdir()
                continue
                    
                
            # parse data
            print("parsing data")
            metadata_json = hdf_file['metadata'].attrs['json']
            metadata_dict = json.loads(metadata_json)
            nodes = metadata_dict['nodes']
            links = metadata_dict['skeletons'][0]['links']
            
            keypoints = [node['name'] for node in nodes]
            skeleton = [[keypoints[l['source']], keypoints[l['target']]] for l in links]
            config['skeleton'] = skeleton
            
            keypoints_ids = [n['id'] for n in metadata_dict['skeletons'][0]['nodes']]
            keypoints_ordered = [keypoints[idx] for idx in keypoints_ids]
            config['bodyparts'] = keypoints_ordered

            columns = [
                'frame'
            ] + [
                f'{kp}_x' for kp in keypoints_ordered
            ] + [
                f'{kp}_y' for kp in keypoints_ordered
            ]
            # print(columns, '\n', [len(d) for d in data])
            scorer_row = ['scorer'] + [f'{scorer}'] * (len(columns) - 1)
            bodyparts_row = ['bodyparts'] + [f'{kp}' for kp in keypoints_ordered for _ in (0, 1)]
            coords_row = ['coords'] + ['x', 'y'] * len(keypoints_ordered)

            print("creating dataframe")
            labels_df = pd.DataFrame(data, columns=columns)
            video_base_name = Path(video_filename).stem
            labels_df['frame'] = labels_df['frame'].apply(
                lambda x: str(Path("labeled-data") / video_base_name / f"img{str(int(x)).zfill(8)}.png")
            )
            labels_df = labels_df.groupby('frame', as_index=False).first()
            data_frames.append(labels_df)
                
            # Combine all data frames into a single DataFrame
            combined_df = pd.concat(data_frames, ignore_index=True)
                
            header_df = pd.DataFrame(
                [scorer_row, bodyparts_row, coords_row],
                columns=combined_df.columns
            )
            final_df = pd.concat([header_df, combined_df], ignore_index=True)
            final_df.columns = [None] * len(final_df.columns)  # Set header to None
            
            # Save concatenated labels
            final_df.to_csv(output_dir / f"CollectedData_{scorer}.csv", index=False, header=None)
print("All folders written")
with open(base_output_dir / 'config.yaml', 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)

# parser = argparse.ArgumentParser()
# parser.add_argument("--slp_file", type=str)
# parser.add_argument("--dlc_dir", type=str)
# args = parser.parse_args()
# slp_file = Path(args.slp_file)
# dlc_dir = Path(args.dlc_dir)
 
# print(f"Converting SLEAP project to DLC project located at {dlc_dir}")
 
# # Check provided SLEAP path exists
# if not slp_file.exists():
#     raise NotADirectoryError(f"did not find the file {slp_file}")
 
# # Extract and save labeled data from SLEAP project
# extract_frames_from_pkg_slp(slp_file, dlc_dir)

'''