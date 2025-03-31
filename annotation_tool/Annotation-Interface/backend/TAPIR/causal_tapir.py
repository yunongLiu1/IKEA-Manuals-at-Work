# Parts of code are adapted from from TAPIR repo
import haiku as hk
import jax
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import tree
import json
from tapnet import tapir_model
from tapnet import tapnet_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils
import cv2
from typing import Optional, List, Tuple
import os
import argparse

# Causal TAPIR
import functools
import jax.numpy as jnp
from tqdm import tqdm
import time
import base64

from flask import Flask, request, jsonify , send_file
from flask_cors import CORS
import json
from jax import profiler, disable_jit

app = Flask(__name__)
CORS(app) # This will enable CORS for all routes

user_data = {}

resize_height = 256  
resize_width = 256

# Utility Functions 

def convert_select_points_to_query_points(frame, points, padding=100):
    """Convert select points to query points.

    Args:
      frame: the frame number
      points: [num_points, 2], in [x, y]
      padding: the length to which the array will be padded
    Returns:
      query_points: [padding, 3], in [t, y, x]
    """
    points = np.array(points)
    num_points = points.shape[0]

    # Initialize an array of zeros with shape [padding, 3]
    query_points = np.zeros(shape=(padding, 3), dtype=np.float32)

    # Fill in the points
    query_points[:num_points, 0] = frame
    query_points[:num_points, 1] = points[:, 1]
    query_points[:num_points, 2] = points[:, 0]

    return query_points


# Build Model
# @title Build Model {form-width: "25%"}

# Internally, the tapir model has three stages of processing: computing
# image features (get_feature_grids), extracting features for each query point
# (get_query_features), and estimating trajectories given query features and
# the feature grids where we want to track (estimate_trajectories).  For
# tracking online, we need extract query features on the first frame only, and
# then call estimate_trajectories on one frame at a time.

def build_online_model_init(frames, query_points):
  """Initialize query features for the query points."""
  model = tapir_model.TAPIR(use_causal_conv=True, bilinear_interp_with_depthwise_conv=False)

  feature_grids = model.get_feature_grids(frames, is_training=False)
  query_features = model.get_query_features(
      frames,
      is_training=False,
      query_points=query_points,
      feature_grids=feature_grids,
  )
  return query_features


def build_online_model_predict(frames, query_features, causal_context):
  """Compute point tracks and occlusions given frames and query points."""
  # dummy = jnp.array([0])  # Create a dummy JAX array
  # jax.device_get(dummy)  # Synchronize device
  time1 = time.time()

  model = tapir_model.TAPIR(use_causal_conv=True, bilinear_interp_with_depthwise_conv=False)

  # dummy = jnp.array([0])  # Create a dummy JAX array
  # jax.device_get(dummy)  # Synchronize device
  time2 = time.time()
  print("########model = tapir_model.TAPIR time elapsed: ", time2 - time1)

  # dummy = jnp.array([0])  # Create a dummy JAX array
  # jax.device_get(dummy)  # Synchronize device
  time1 = time.time()

  feature_grids = model.get_feature_grids(frames, is_training=False)
  # dummy = jnp.array([0])  # Create a dummy JAX array
  # jax.device_get(dummy)  # Synchronize device
  time2 = time.time()
  print("#######feature_grids = model.get_feature_grids time elapsed: ", time2 - time1)

  # dummy = jnp.array([0])  # Create a dummy JAX array
  # jax.device_get(dummy)  # Synchronize device
  time1 = time.time()

  trajectories = model.estimate_trajectories(
      frames.shape[-3:-1],
      is_training=False,
      feature_grids=feature_grids,
      query_features=query_features,
      query_points_in_video=None,
      query_chunk_size=64,
      causal_context=causal_context,
      get_causal_context=True,
  )
  # dummy = jnp.array([0])  # Create a dummy JAX array
  # jax.device_get(dummy)  # Synchronize device
  time2 = time.time()
  print("########trajectories = model.estimate_trajectories time elapsed: ", time2 - time1)


  # dummy = jnp.array([0])  # Create a dummy JAX array
  # jax.device_get(dummy)  # Synchronize device
  time1 = time.time()

  causal_context = trajectories['causal_context']
  del trajectories['causal_context']
  return_info = {k: v[-1] for k, v in trajectories.items()}, causal_context

  # dummy = jnp.array([0])  # Create a dummy JAX array
  # jax.device_get(dummy)  # Synchronize device
  time2 = time.time()
  print("########Get return info time elapsed: ", time2 - time1)

  return return_info


# @title Utility Functions {form-width: "25%"}

def preprocess_frame(frame):
  """Preprocess frames to model inputs.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

  Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
  """
  frame = frame.astype(np.float32)
  frame = frame / 255 * 2 - 1
  return frame


def postprocess_occlusions(occlusions, expected_dist):
  """Postprocess occlusions to boolean visible flag.

  Args:
    occlusions: [num_points, num_frames], [-inf, inf], np.float32

  Returns:
    visibles: [num_points, num_frames], bool
  """
  pred_occ = jax.nn.sigmoid(occlusions)
  pred_occ = 1 - (1 - pred_occ) * (1 - jax.nn.sigmoid(expected_dist))
  visibles = pred_occ < 0.5  # threshold
  return visibles



def construct_initial_causal_state(num_points, num_resolutions):
  value_shapes = {
      "tapir/~/pips_mlp_mixer/block_1_causal_1": (1, num_points, 2, 512),
      "tapir/~/pips_mlp_mixer/block_1_causal_2": (1, num_points, 2, 2048),
      "tapir/~/pips_mlp_mixer/block_2_causal_1": (1, num_points, 2, 512),
      "tapir/~/pips_mlp_mixer/block_2_causal_2": (1, num_points, 2, 2048),
      "tapir/~/pips_mlp_mixer/block_3_causal_1": (1, num_points, 2, 512),
      "tapir/~/pips_mlp_mixer/block_3_causal_2": (1, num_points, 2, 2048),
      "tapir/~/pips_mlp_mixer/block_4_causal_1": (1, num_points, 2, 512),
      "tapir/~/pips_mlp_mixer/block_4_causal_2": (1, num_points, 2, 2048),
      "tapir/~/pips_mlp_mixer/block_5_causal_1": (1, num_points, 2, 512),
      "tapir/~/pips_mlp_mixer/block_5_causal_2": (1, num_points, 2, 2048),
      "tapir/~/pips_mlp_mixer/block_6_causal_1": (1, num_points, 2, 512),
      "tapir/~/pips_mlp_mixer/block_6_causal_2": (1, num_points, 2, 2048),
      "tapir/~/pips_mlp_mixer/block_7_causal_1": (1, num_points, 2, 512),
      "tapir/~/pips_mlp_mixer/block_7_causal_2": (1, num_points, 2, 2048),
      "tapir/~/pips_mlp_mixer/block_8_causal_1": (1, num_points, 2, 512),
      "tapir/~/pips_mlp_mixer/block_8_causal_2": (1, num_points, 2, 2048),
      "tapir/~/pips_mlp_mixer/block_9_causal_1": (1, num_points, 2, 512),
      "tapir/~/pips_mlp_mixer/block_9_causal_2": (1, num_points, 2, 2048),
      "tapir/~/pips_mlp_mixer/block_10_causal_1": (1, num_points, 2, 512),
      "tapir/~/pips_mlp_mixer/block_10_causal_2": (1, num_points, 2, 2048),
      "tapir/~/pips_mlp_mixer/block_11_causal_1": (1, num_points, 2, 512),
      "tapir/~/pips_mlp_mixer/block_11_causal_2": (1, num_points, 2, 2048),
      "tapir/~/pips_mlp_mixer/block_causal_1": (1, num_points, 2, 512),
      "tapir/~/pips_mlp_mixer/block_causal_2": (1, num_points, 2, 2048),
  }
  fake_ret = {
      k: jnp.zeros(v, dtype=jnp.float32) for k, v in value_shapes.items()
  }
  return [fake_ret] * num_resolutions * 4
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


# Load Checkpoint
# dummy = jnp.array([0])  # Create a dummy JAX array
# jax.device_get(dummy)  # Synchronize device
time1 = time.time()

HOME = os.getcwd() + "/TAPIR"
CHECKPOINT_PATH = os.path.join(HOME, "tapnet/checkpoints/causal_tapir_checkpoint.npy")
ckpt_state = np.load(CHECKPOINT_PATH, allow_pickle=True).item()
params, state = ckpt_state['params'], ckpt_state['state']

# dummy = jnp.array([0])  # Create a dummy JAX array
# jax.device_get(dummy)  # Synchronize device
time2 = time.time()
print("####Time to load checkpoint: ", time2 - time1)

# dummy = jnp.array([0])  # Create a dummy JAX array
# jax.device_get(dummy)  # Synchronize device
time1 = time.time()

online_init = hk.transform_with_state(build_online_model_init)
online_init_apply = jax.jit(online_init.apply)

online_predict = hk.transform_with_state(build_online_model_predict)
online_predict_apply = jax.jit(online_predict.apply)

rng = jax.random.PRNGKey(42)
online_init_apply = functools.partial(
    online_init_apply, params=params, state=state, rng=rng
)
online_predict_apply = functools.partial(
    online_predict_apply, params=params, state=state, rng=rng
)

# dummy = jnp.array([0])  # Create a dummy JAX array
# jax.device_get(dummy)  # Synchronize device
time2 = time.time()

print("#######Online init time elapsed: ", time2 - time1)


############# Warmup for jax.jit compilation ##########

# dummy = jnp.array([0])  # Create a dummy JAX array
# jax.device_get(dummy)  # Synchronize device
time1 = time.time()

dummy_select_points = np.zeros((4, 2))
# Frame shape (256, 256, 3)
dummy_frames = np.zeros((1, 256, 256, 3))
dummy_width = 256
dummy_height = 256
dummy_resize_height = 256
dummy_resize_width = 256
# The selected_points was normalized to [0, 1] in the frontend, so we need to convert it back to the original size
dummy_select_points[:, 0] = dummy_select_points[:, 0] * dummy_width
dummy_select_points[:, 1] = dummy_select_points[:, 1] * dummy_height
dummy_query_points = convert_select_points_to_query_points(0, dummy_select_points)
dummy_query_points = transforms.convert_grid_coordinates(
  dummy_query_points, (1, dummy_height, dummy_width), (1, dummy_resize_height, dummy_resize_width), coordinate_format='tyx')

dummy_query_features, _ = online_init_apply(frames=preprocess_frame(dummy_frames[None, None, 0]), query_points=dummy_query_points[None])
dummy_causal_state = construct_initial_causal_state(dummy_query_points.shape[0], len(dummy_query_features.resolutions) - 1)
(dummy_prediction, dummy_causal_state), _ = online_predict_apply(
    frames=preprocess_frame(dummy_frames[None, None, 0]),
    query_features=dummy_query_features,
    causal_context=dummy_causal_state,
  )

# dummy = jnp.array([0])  # Create a dummy JAX array
# jax.device_get(dummy)  # Synchronize device
time2 = time.time()
print("#######Warm up time elapsed: ", time2 - time1)

############# End of Warmup for jax.jit compilation ##########

# @app.route('/init-video', methods=['POST'])
# def init_video():




#     return jsonify({'messages': ['Video initialized and all reset.']})




# Redo this if the query points are changed
@app.route('/init-points', methods=['POST'])
def init_points():
    print("Initilize video")
    # # Initilize a global variable to store the video
    # global height
    # global width
    # global frames

    datas = request.json
    user = datas['user']
    path = datas['videoPath']
    print("init video path: ", path)
    # video = media.read_video(path)
    
    img_data = datas['image'].split(",")[1]  # Remove "data:image/png;base64," prefix
    img_bytes = base64.b64decode(img_data)
    image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[0:2]

    # Save data to user_data
    user_data[user] = {}
    user_data[user]['videoPath'] = path
    user_data[user]['height'] = height
    user_data[user]['width'] = width

    print(" Video Initilization Done!")

    print("initilize points")


    # Sync jax device
    # dummy = jnp.array([0])  # Create a dummy JAX array
    # jax.device_get(dummy)  # Synchronize device
    start_time = time.time()
    datas = request.json
  

    # print("datas: ", datas)
    select_points = []
    for data in datas["2d-coordinates"]:
      point = [data['x'], data['y']]
      select_points.append(point)
    select_points = np.array(select_points)
    
    num_points = select_points.shape[0]
    user_data[datas['user']]['num_points'] = num_points


    # The selected_points was normalized to [0, 1] in the frontend, so we need to convert it back to the original size
    select_points[:, 0] = select_points[:, 0] * width
    select_points[:, 1] = select_points[:, 1] * height
    query_points = convert_select_points_to_query_points(0, select_points)
    query_points = transforms.convert_grid_coordinates(
      query_points, (1, height, width), (1, resize_height, resize_width), coordinate_format='tyx')


    #Sync jax device
    # dummy = jnp.array([0])  # Create a dummy JAX array
    # jax.device_get(dummy)  # Synchronize device
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"######Convert query_points: {elapsed_time} seconds")

    frame = image
    # Resieze frame to 256 * 256
    frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
    
    # dummy = jnp.array([0])  # Create a dummy JAX array
    # jax.device_get(dummy)  # Synchronize device
    start_time = time.time()
    query_features, _ = online_init_apply(frames=preprocess_frame(frame[None, None]), query_points=query_points[None])
    causal_state = construct_initial_causal_state(query_points.shape[0], len(query_features.resolutions) - 1)
    # torch.cuda.sychronize()
    # dummy = jnp.array([0])  # Create a dummy JAX array
    # jax.device_get(dummy)  # Synchronize device
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"#######Points initialization time: {elapsed_time} seconds")
    
    # Save data to user_data
    user_data[datas['user']]['query_points'] = query_points
    user_data[datas['user']]['query_features'] = query_features
    user_data[datas['user']]['causal_state'] = causal_state
    user_data[datas['user']]['num_points'] = num_points

    print("query_points: ", query_points)
    print(" Points initialized! ")
    return jsonify({'messages': ['Points initialized.']})


def get_confidences(occlusions, expected_dist):
  """Postprocess occlusions to get confidences.

  Args:
    occlusions: [num_points, num_frames], [-inf, inf], np.float32

  Returns:
    visibles: [num_points, num_frames], bool
  """
  pred_occ = jax.nn.sigmoid(occlusions)
  pred_occ = 1 - (1 - pred_occ) * (1 - jax.nn.sigmoid(expected_dist))
  confidence = 1- pred_occ
  return confidence


@app.route('/get-predictions', methods=['POST'])
def get_predictions():
    # global causal_state
    # global num_points
    datas = request.json
    user = datas['user']
    category = datas['Category']
    name = datas['SubCategory']
    step_id = datas["Object"]
    video = datas['video']
    video = video.split('/')[-1].split('.')[0]
    print("video: ", video) 
    projection = datas['Projection']
    mode = 'Projection' if projection else 'Segmentation'
    fps = datas['fps']

    query_features = user_data[user]['query_features']
    causal_state = user_data[user]['causal_state']
    num_points = user_data[user]['num_points']
    

    image = datas['image'].split(",")[1]  # Remove "data:image/png;base64," prefix
    img_bytes = base64.b64decode(image)
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)

    # Synchronize jax device
    # dummy = jnp.array([0])  # Create a dummy JAX array
    # jax.device_get(dummy)  # Synchronize device 
    start_time = time.time()

    (prediction, causal_state), _ = online_predict_apply(
    frames=preprocess_frame(frame[None, None]),
    query_features=query_features,
    causal_context=causal_state,
    )
    user_data[user]['causal_state'] = causal_state


    # Synchronize jax device 
    # dummy = jnp.array([0])  # Create a dummy JAX array
    # jax.device_get(dummy)  # Synchronize device
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"#######Time for this prediction online_predict_apply: {elapsed_time} seconds")

    print('num_points',num_points)
    tracks = prediction['tracks']

    print('tracks.shape',tracks.shape)
    points_coordinates = tracks[0,:num_points,0,:]
    print('points_coordinates.shape',points_coordinates.shape)


    #Sync jax device
    # dummy = jnp.array([0])  # Create a dummy JAX array
    # jax.device_get(dummy)  # Synchronize device

    start_time = time.time()

    print("prediction['occlusion'].shape", prediction['occlusion'].shape)
    print("prediction['expected_dist'].shape", prediction['expected_dist'].shape)
    confidences = get_confidences(prediction['occlusion'][:,:num_points,:],prediction['expected_dist'][:,:num_points,:])
    confidences = confidences.flatten()
    print("confidences", confidences)
    print("confidences shape", confidences.shape)

    #Sync jax device
    # dummy = jnp.array([0])  # Create a dummy JAX array
    # jax.device_get(dummy)  # Synchronize device
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"#######Time for this prediction get_confidences: {elapsed_time} seconds")

    output_data = {}
    output_data['pointsCoordinates'] = points_coordinates.tolist()
    output_data['confidences'] = confidences.tolist()
    output_data['prediction_frame_time'] = datas["currentFrameTime"]
    output_data['num_points'] = num_points
    output_data['video'] = video
    output_data['category'] = category
    output_data['name'] = name
    output_data['step_id'] = step_id
    output_data['user'] = user
    output_data['occlusion'] = prediction['occlusion'][:,:num_points,:].tolist()
    output_data['expected_dist'] = prediction['expected_dist'][:,:num_points,:].tolist()
    output_data['image'] = datas['image']
    

    # Save the prediction to local file
    output_data_dir = os.getcwd() + "/dynamic_dataset/points-tracking/" + category + "/" + name + "/" + step_id + '/'+mode
    # Create the output directory if it does not exist
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    output_path = output_data_dir +'/'+user+'_' + video + str(int(float(datas["currentFrameTime"])*fps)) + '.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print("Saved points prediction to :",output_path)


    return jsonify({'messages': ['Predictions returned.'], "pointsCoordinates": points_coordinates.tolist(), "confidences": confidences.tolist()})

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=9000, threaded=True) #Enable threaded=True to allow multiple connections.
