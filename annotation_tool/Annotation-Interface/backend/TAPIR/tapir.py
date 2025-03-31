# Parts of code are adapted from from TAPIR repo
import haiku as hk
import jax
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import tree
import json
from Tapnet import tapir_model
from Tapnet import tapnet_model
from Tapnet.utils import transforms
from Tapnet.utils import viz_utils
import cv2
from typing import Optional, List, Tuple
import os
import argparse



def load_video(path):
    # Initialize a VideoCapture object
    cap = cv2.VideoCapture(path)

    # Initialize a list to hold the frames
    frames = []

    # Loop over the frames from the video stream
    while cap.isOpened():
        # Read the next frame, read one frame per second
        ret, frame = cap.read()
        if ret:
            # Convert the frame from BGR to RGB format
            
            # Append the frame to the list
            frames.append(frame)
            # if len(frames) == 400:
            # break
        else:
            break

    # Release the VideoCapture object
    cap.release()

    # Convert the list of frames to a numpy array
    video = np.stack(frames)

    return video


# Utility Functions 

def preprocess_frames(frames):
  """Preprocess frames to model inputs.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

  Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
  """
  frames = frames.astype(np.float32)
  frames = frames / 255 * 2 - 1
  return frames



def inference(frames, query_points, model_apply, params, state):
  """Inference on one video.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8
    query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]

  Returns:
    tracks: [num_points, 3], [-1, 1], [t, y, x]
    visibles: [num_points, num_frames], bool
  """
  # Preprocess video to match model inputs format
  frames = preprocess_frames(frames)
  num_frames, height, width = frames.shape[0:3]
  query_points = query_points.astype(np.float32)
  frames, query_points = frames[None], query_points[None]  # Add batch dimension

  # Model inference
  rng = jax.random.PRNGKey(42)
  outputs, _ = model_apply(params, state, rng, frames, query_points)
  outputs = tree.map_structure(lambda x: np.array(x[0]), outputs)
  # tracks, occlusions, expected_dist = outputs['tracks'], outputs['occlusion'], outputs['expected_dist']

  # # Binarize occlusions
  # visibles = postprocess_occlusions(occlusions, expected_dist)
  # return tracks, visibles
  return outputs


def convert_select_points_to_query_points(frame, points):
  """Convert select points to query points.

  Args:
    points: [num_points, 2], in [x, y]
  Returns:
    query_points: [num_points, 3], in [t, y, x]
  """
  points = np.stack(points)
  query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
  query_points[:, 0] = frame
  query_points[:, 1] = points[:, 1]
  query_points[:, 2] = points[:, 0]
  return query_points


# Build Model
def build_model(frames, query_points):
  """Compute point tracks and occlusions given frames and query points."""
  model = tapir_model.TAPIR(bilinear_interp_with_depthwise_conv=False)
  outputs = model(
      video=frames,
      is_training=False,
      query_points=query_points,
      query_chunk_size=64,
  )
  return outputs

# Postprocess for TAPIR
def postprocess_occlusions(occlusions, expected_dist, threshold = 0.5):
  """Postprocess occlusions to boolean visible flag.

  Args:
    occlusions: [num_points, num_frames], [-inf, inf], np.float32
    expected_dist: [num_points, num_frames], [-inf, inf], np.float32

  Returns:
    visibles: [num_points, num_frames], bool
  """
  visibles = (1 - jax.nn.sigmoid(occlusions)) * (1 - jax.nn.sigmoid(expected_dist)) > threshold
  return visibles

def get_visibles(occlusions, expected_dist, threshold = 0.5):

  return (1 - jax.nn.sigmoid(occlusions)) * (1 - jax.nn.sigmoid(expected_dist))


def ndarray_to_list(data):
    if isinstance(data, dict):
        return {key: ndarray_to_list(value) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [ndarray_to_list(item) for item in data]
    else:
        return data

import time
if __name__ == "__main__":
    # Load Checkpoint
    time1 = time.time()

    HOME = os.getcwd() + "/TAPIR"
    CHECKPOINT_PATH = os.path.join(HOME, "tapnet/checkpoints/tapir_checkpoint.npy")
    ckpt_state = np.load(CHECKPOINT_PATH, allow_pickle=True).item()
    params, state = ckpt_state['params'], ckpt_state['state']
    time2 = time.time()
    print("Time to load checkpoint: ", time2 - time1)

    # Use argparse to get the input

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--json_path', type=str, help='Path to the json file', default='./points_tracking_data.json' )
    parser.add_argument('--output_type', type=str, help='Type of the output', default='image' )

    args = parser.parse_args()

    # Load the json file
    with open(args.json_path, 'r') as f:
        data = json.load(f)
    
    # Load parameters
    path = data['input_video_path']
    start_frame = int(data['start_frame'])
    end_frame = int(data['end_frame'])

    # If img path is /Bench/applaro/step_0/KPs0ik2FcsY/frame_0.jpg, then video path is /Bench/applaro/step_0/KPs0ik2FcsY/KPs0ik2FcsY_clip.mp4
    input_video_path = '/'.join(data['input_video_path'].split('/')[0:-1])+ '/' + data['input_video_path'].split('/')[-2] + '_clip.mp4'

    output_video_dir = data['output_video_dir']
    output_data_dir = data['output_data_dir']
    select_points = data['select_points']
    select_frame = int(data['select_frame'])

    time1 = time.time()

    print("Load data time elapsed: ", time1 - time2)

    colormap = viz_utils.get_colors(100)

    # Preprocess the data
    filename = path.split('/')[-1].split('.')[0]
    select_points = np.array(eval(select_points))

    time2 = time.time()

    print("Preprocess data time elapsed: ", time2 - time1)

    # Build Model
    model = hk.transform_with_state(build_model)
    model_apply = jax.jit(model.apply)
    time1 = time.time()

    print("Build model time elapsed: ", time1 - time2)


    # Load video

    video = load_video(input_video_path)
    height = video[0].shape[0]
    width = video[0].shape[1]

    time2 = time.time()

    print("Load video time elapsed: ", time2 - time1)

    resize_height = 256  
    resize_width = 256 

    # The selected_points was normalized to [0, 1] in the frontend, so we need to convert it back to the original size
    select_points[:, 0] = select_points[:, 0] * width
    select_points[:, 1] = select_points[:, 1] * height


    frames = media.resize_video(video[start_frame:end_frame], (resize_height, resize_width))
    query_points = convert_select_points_to_query_points(select_frame - start_frame, select_points)
    query_points = transforms.convert_grid_coordinates(
        query_points, (1, height, width), (1, resize_height, resize_width), coordinate_format='tyx')

    outputs = inference(frames, query_points, model_apply, params, state)
    # outputs = inference(frames, query_points, model_apply, params_tapir, state_tapir)

    time1 = time.time()
    print("Inference time elapsed: ", time1 - time2)

    converted_data = ndarray_to_list(outputs)
    # Normalize tracks to [0, 1]
    # converted_data['tracks'][:, :, 1] = converted_data['tracks'][:, :, 1] / resize_height
    # converted_data['tracks'][:, :, 2] = converted_data['tracks'][:, :, 2] / resize_width


    # Create the output directory if it does not exist
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir) 
    with open(output_data_dir + '/' + filename + '.json', 'w') as f:
        json.dump(converted_data, f, indent=4)
    print("Data saved to: ", output_data_dir + '/' + filename + '.json')

    time2 = time.time()
    print("Save data time elapsed: ", time2 - time1)

    tracks, occlusions, expected_dist = outputs['tracks'], outputs['occlusion'], outputs['expected_dist']
    tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))

    # Binarize occlusions
    # visibles = postprocess_occlusions(occlusions, expected_dist)

    # Get visibles
    visibles = get_visibles(occlusions, expected_dist)
    video_viz = viz_utils.paint_point_track(video[start_frame:end_frame], tracks, visibles, colormap)

    # If directory does not exist, create it
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)

    media.write_video(output_video_dir + '/' + filename + '_viz.mp4', video_viz, fps=10)
    print("video_viz saved to: ", output_video_dir + '/' + filename + '_viz.mp4')

    time1 = time.time()
    print("Save video time elapsed: ", time1 - time2)
    
    # Save the output video and data path to the json file
    data['output_video_path'] = output_video_dir + '/' + filename + '_viz.mp4'
    data['output_data_path'] = output_data_dir + '/' + filename + '.json'

    # Save the json file
    with open(args.json_path, 'w') as f:
        json.dump(data, f, indent=4)


    # python TAPIR/tapir.py --json_path ./points_tracking_data.json