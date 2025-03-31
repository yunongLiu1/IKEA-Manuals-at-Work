import numpy as np
import cv2
import os
import open3d as o3d
import json
import numpy as np
from PIL import Image
from pyvirtualdisplay import Display

import argparse
import base64
import cv2
import json
import numpy as np
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
from utils.utils import render_part, render_parts, get_cam_pose_from_look_at, is_the_same_part
import copy

display = Display().start()

def pose_estimation(k, int_mat, aux_ext_mat=None):
  '''
  k is a dict containing:
    keypoints_2d:
    keypoints_3d:
    image_width:
    image_height:
  '''

  keypoints_2d = np.array(k['keypoints_2d'], dtype=np.float32)
  keypoints_3d = np.array(k['keypoints_3d'], dtype=np.double)
  if not (keypoints_2d.shape[1] == 2 and keypoints_3d.shape[1] == 3 and keypoints_2d.shape[0] == keypoints_3d.shape[0]):
    return False, f"Keypoints shape mismatch. {keypoints_2d.shape=} {keypoints_3d.shape=}", {}

  n = keypoints_2d.shape[0]
  if n <= 3:
    return False, "Not enough keypoints.", {}

  cx = k['image_width'] / 2
  cy = k['image_height'] / 2
  f = np.array(int_mat)[0, 0]
  camera_matrix = np.array([
      [f, 0, cx],
      [0, f, cy],
      [0, 0, 1]
    ], dtype=np.float32)
  dist_coeffs = np.zeros((4,1))
  # success, rotation_vector, translation_vector = cv2.solvePnP(keypoints_3d, keypoints_2d, camera_matrix, dist_coeffs, flags=0)
  # success, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(keypoints_3d, keypoints_2d, camera_matrix, dist_coeffs, flags=0)
  if aux_ext_mat is not None:
    rotation_vector_init = cv2.Rodrigues(aux_ext_mat[:3, :3])[0]
    translation_vector_init = np.array(aux_ext_mat[:3, [3]])
    success, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(keypoints_3d, keypoints_2d, camera_matrix, dist_coeffs, 
    # success, rotation_vector, translation_vector = cv2.solvePnP(keypoints_3d, keypoints_2d, camera_matrix, dist_coeffs, 
    rotation_vector_init, translation_vector_init,
    useExtrinsicGuess=True,
    flags=0)
  else:
    success, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(keypoints_3d, keypoints_2d, camera_matrix, dist_coeffs, flags=0)
  rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
  ext_matrix = np.zeros((4, 4))
  ext_matrix[:3, :3] = rotation_matrix
  ext_matrix[:3, 3] = translation_vector[:, 0]
  ext_matrix[3, 3] = 1

  reprojected = cv2.projectPoints(keypoints_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)[0]
  reprojected = reprojected[:, 0, :]
  reproj_err = ((keypoints_2d - reprojected) ** 2).sum(axis=-1).mean()
  
  result = {}
  if success:
    result['reprojected'] = reprojected.tolist()
    result['extrinsic'] = ext_matrix.tolist()
    result['reprojection_error'] = reproj_err
    result['intrinsic'] = camera_matrix.tolist()
    
  return success, "Success." if result != {} else "SolvePnP failed.", result


# if __name__ == '__main__':

###### Data Preparation ######
parser = argparse.ArgumentParser(description='Pose estimation and rendering')
parser.add_argument('--json', type=str, default='./user_data/yunong_pose_estimation_data.json', help='path to the json file')

args = parser.parse_args()
json_path = args.json

# Read the data from json file
with open(json_path) as f:
    data = json.load(f)
    points_2d = data['points_2d']
    points_3d = data['points_3d']
    part_idxs = data['part_idxs']
    # video_path = data['video_path']
    video_path = data['video_path']
    output_path = data['output_path']
    obj_path = data['obj_path']
    currentFrameTime = float(data['currentFrameTime'])
    img_data = data['image'].split(",")[1]  # Remove "data:image/png;base64," prefix
    json_path_for_curr_user = data['json_path_for_curr_user']

img_bytes = base64.b64decode(img_data)
image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


width = img.shape[1]
height = img.shape[0]
# print("Image width: ", width)
# print("Image height: ", height)

for i in range(len(points_2d)):
    points_2d[i][0] = points_2d[i][0] * width
    points_2d[i][1] = points_2d[i][1] * height

# # Convert to only 3 digits after the decimal point
points_2d = [[round(x, 3), round(y, 3)] for (x, y) in points_2d]
points_3d = [[round(x, 3), round(y, 3), round(z, 3)] for (x, y, z) in points_3d]
# print(points_2d)
# print(points_3d)

if data['aux_ext_mat'] is not None and data['aux_ext_mat'] !=[]:
  # if data['aux_ext_mat'][0] != [[]]:
  #   aux_ext_mat = None
  # else:
  #   aux_ext_mat = np.array(data['aux_ext_mat'][0])
  aux_ext_mat = np.array(data['aux_ext_mat'])
  # print("aux_ext_mat.shape: ", aux_ext_mat.shape)
else:
  aux_ext_mat = None

# aux_ext_mat = np.array(data['aux_ext_mat'])
# print("aux_ext_mat.shape: ", aux_ext_mat.shape)
 
# part_idxs = [int(i) for i in part_idxs.split(',')]

## Get video intrinsic
json_data = json.load(open(json_path_for_curr_user))
for furniture in json_data:
  # print(f'Finding furniture {furniture["name"]} in {furniture["category"]}...')
  if furniture['name'] == data['name'] and furniture['category'] == data['category']:
    # print('Found furniture')
    # print(f'Finding step {data["step_id"]}...')
    for step in furniture['steps']:
      # print(f' Current step: {step["step_id"]}')
      if int(step['step_id']) == int(data['step_id']):
        # print('Found step', step['step_id'])
        for video in step['video']:
          # print(f'Finding video {video["video_id"]}...')
          if video['video_id'].split("/watch?v=")[-1] == data['video_id']:
            # print('Found video', video['video_id'])
            video_intrinsic = video['video_intrinsic']
            print('video_intrinsic', video_intrinsic)
            
k = {
    'keypoints_2d': points_2d,
    'keypoints_3d': points_3d,
    'image_width': width,
    'image_height': height
    }

success, msg, result = pose_estimation(k, video_intrinsic)
print('msg: ',msg)
print('result',result)

ext_mat = np.array(result['extrinsic'])
int_mat = np.array(result['intrinsic'])
assert np.allclose(int_mat, video_intrinsic), "Intrinsic matrix mismatch."

print("Start rendering...")
render_part(obj_path, part_idxs, ext_mat, int_mat, width, height, output_path)
print("Done.")

# Write back the result and success to json file
data['success'] = success
data['extrinsic'] = ext_mat.tolist()
data['intrinsic'] = int_mat.tolist()
with open(json_path, 'w') as f:
    json.dump(data, f, indent=4)
result_ext = np.array(result['extrinsic'])

########## Get the furniture information ##########
T_cam_to_part_list = []
part_mesh_list = []
part_mesh_id_list = []
int_mat_list = []
video_intrinsic = None
json_data = json.load(open(json_path_for_curr_user))
for furniture in json_data:
  # print(f'Finding furniture {furniture["name"]} in {furniture["category"]}...')
  if furniture['name'] == data['name'] and furniture['category'] == data['category']:
    # print('Found furniture')
    # print(f'Finding step {data["step_id"]}...')
    for step in furniture['steps']:
      # print(f' Current step: {step["step_id"]}')
      if int(step['step_id']) == int(data['step_id']):
        # print('Found step', step['step_id'])
        for video in step['video']:
          # print(f'Finding video {video["video_id"]}...')
          if video['video_id'].split("/watch?v=")[-1] == data['video_id']:
            # print('Found video', video['video_id'])
            video_intrinsic = video['video_intrinsic']
            # print('video_intrinsic', video_intrinsic)
            for frame in video['frames']:
              if int(frame['frame_id']) == int(data['frame_id']):
                # Get part meshes:
                for i, part_ids in enumerate(frame['parts']):
                  # print('Starting part', part_ids)
                  ext_mat = np.array(frame['extrinsics'][frame['parts'].index(part_ids)])
                  int_mat = np.array(frame['intrinsics'][frame['parts'].index(part_ids)])
                  

                  if is_the_same_part(part_ids, ''.join(part_idxs.split(','))):
                    ext_mat = result_ext
                  
                  if ext_mat.shape != (4, 4) or int_mat.shape != (3, 3):
                      # print('Invalid extrinsic or intrinsic matrix. Skip')
                      continue
                  assert np.allclose(int_mat, video_intrinsic), "Intrinsic matrix mismatch."

                  T_cam_to_part_list.append(ext_mat)
                  int_mat_list.append(int_mat)

                  part_ids_ls = part_ids.split(',')

                  part_meshes = []
                  vertices = []
                  triangle = []
                  vertices_count = 0
                  for part_id in part_ids_ls:
                      # print(f'Loading part {part_id}')
                      mesh_filename = os.path.join(obj_path, str(part_id).zfill(2) + '.obj')
                      mesh = o3d.io.read_triangle_mesh(mesh_filename)
                      part_meshes.append(mesh)

                      vertices.append(np.asarray(mesh.vertices))
                      triangle.append(np.asarray(mesh.triangles) + vertices_count)
                      vertices_count += len(mesh.vertices)
                  
                  # print('Loading done')
                  # merge subassemblies
                  vertices = np.concatenate(vertices, axis=0)
                  triangles = np.concatenate(triangle, axis=0)
                  merged_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices),
                                                          triangles=o3d.utility.Vector3iVector(triangles))

                  # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
                  # o3d.visualization.draw_geometries([merged_mesh, coordinate_frame])

                  part_mesh_list.append(merged_mesh)
                  part_mesh_id_list.append(part_ids)

#### Pose the obj to the right location ####
if len(part_mesh_list) >= 2:
  for p in range(len(part_mesh_list)):
      # put the second part in the frame of the first part
      # print(f'Putting part {p} in the frame of part 0')
      T_p1_to_p2 = np.linalg.inv(T_cam_to_part_list[0]) @ T_cam_to_part_list[p]
      part_mesh_list[p].transform(T_p1_to_p2)
                
if video_intrinsic is None:
  print("#### Warning: Cannot find the video intrinsic.")
  exit(0)


try:
    img_path = output_path.split('.')[0] + f'_overlay.jpg'
    # print(f'image_path: {img_path}')

    # Read and process the image
    pose_estimation_image = cv2.imread(output_path)
    pose_estimation_image = cv2.cvtColor(pose_estimation_image, cv2.COLOR_BGR2RGB)
    overlay_img = cv2.addWeighted(img, 0.5, pose_estimation_image, 0.5, 0)
    # Save overlay image to img_path
    plt.imshow(overlay_img)

    # Ensure the directory for img_path exists
    img_directory = os.path.dirname(img_path)
    if not os.path.exists(img_directory):
        os.makedirs(img_directory)

    # Save the image
    plt.savefig(img_path)
    # print(f'Save image to {img_path}')


    ######## Save different view images ########
    data['different_view_images'] = []

    for a in range(6):
            
      eye = [0, 0, 0]
      eye[a//2] = 5 if a%2 == 0 else -5
      if a//2 == 0:
          axis = 'x'
      elif a//2 == 1:
          axis = 'y'
      else:
          axis = 'z'
      angle = '90' if a%2 == 0 else '270'
      up = [0, 0, 0]
      # Make sure up is not parallel to the camera direction
      up[((a+2)//2)%3] = 1
      # print(eye, up)

      diff_view_output_folder = img_directory + f'/pose-estimation_check_{axis}{angle}/'
      diff_view_output_folder = os.path.join(diff_view_output_folder,furniture["category"], furniture["name"],'step_'+str(step["step_id"]), video["video_id"].split("/watch?v=")[-1]) 
      # if not os.path.exists(diff_view_output_folder):
      #     os.makedirs(diff_view_output_folder)
      
      diff_view_output_path = img_directory + '/'+ data['video_id'] +'_'+ f'frame_{data["frame_id"]}_pose_estimation_{axis}{angle}.png'
      new_cam_pose = get_cam_pose_from_look_at(at=[0, 0, 0], eye=eye, up=up)
      T_cam_to_p1 = np.linalg.inv(new_cam_pose)
      # print(new_cam_pose)
      part_mesh_list_in_cam = []
      for part_mesh in part_mesh_list:
          part_mesh_in_cam = copy.deepcopy(part_mesh)
          part_mesh_in_cam.transform(T_cam_to_p1)
          part_mesh_list_in_cam.append(part_mesh_in_cam)
      # o3d.visualization.draw_geometries(part_mesh_list_in_cam)
      img = render_parts(width, height, ext_mat,int_mat_list[0], part_mesh_list_in_cam, part_mesh_id_list)

      plt.imshow(img)
      plt.axis('off')
      plt.savefig(diff_view_output_path)
      # print(f'different view image saved to {diff_view_output_path}')
      data['different_view_images'].append(diff_view_output_path)

      with open(json_path, 'w') as f:
          json.dump(data, f, indent=4)
    
except Exception as e:
    print(f'Failed to save image to {img_path}, error: {e}')


# print("Done.")
# success = display.stop()
# print(success)
