import copy
import os
import cv2
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
import numpy as np


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import open3d as o3d
import trimesh
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import json

import os
import open3d as o3d
import json
from PIL import Image
import copy

from pyvirtualdisplay import Display
# Display().start()
import argparse

# Plot different parts in different colors
colors = [  [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0],[1.0, 0.0, 1.0, 1.0],[0.0, 1.0, 1.0, 1.0],
[1.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0], [0.0, 0.5, 0.0, 1.0], [0.0, 0.0, 0.5, 1.0]]

def is_the_same_part(part, part_str):
    '''
    part is a list of part
    part_str is a string of part
    '''
    part_str_copy = part_str
    for p in part.split(','):
        if p in part_str:
            # Remove the part from part_str
            part_str = part_str.replace(p, '')
        else:
            # print(f'part_str {part_str_copy} does not match part {part}')
            return False
    if len(part_str) == 0:
        # print(f'part_str {part_str_copy} matches part {part}')
        return True
    else:
        return False


def render_part(
    obj_path,
    part_ids,
    ext_mat,
    int_mat,
    img_width,
    img_height,
    save_path, debug=True):
    
    #Create a file to store all debug information


    # Create a pinhole camera intrinsic
    if debug:
        print("Creating pinhole camera intrinsic...")
    pinhole = o3d.camera.PinholeCameraIntrinsic(
    int(img_width), int(img_height), int_mat[0, 0], int_mat[1, 1], int_mat[0, 2], int_mat[1, 2])

    # Create a perspective camera
    if debug:
        print("Creating perspective camera...")
        
    render = o3d.visualization.rendering.OffscreenRenderer(int(img_width), int(img_height))

    # Load the object
    if debug:
        print("Loading object...")
    render.scene.set_background([255.0, 255.0, 255.0, 255.0])  # RGBA
    
    # Load the object
    if debug:
        print("Loading object...")
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultLit"



    idx = 0
    print('part_ids: ',part_ids)
    part_ids = part_ids.split(',')
    color_id = [int(j) for j in part_ids]
    color_id = min(color_id)
    print(f'render part color id: {color_id} for part {part_ids}')
    for part in part_ids:
        print('current part', part)
        idx += 1
        curr_obj_path = os.path.join( obj_path,str(part).zfill(2) + '.obj')
        print('curr_obj_path', curr_obj_path)

        mesh = o3d.io.read_triangle_mesh(curr_obj_path)
        # mtl.base_color = colors[idx % len(colors)]
        mtl.base_color = colors[int(color_id) % len(colors)]
        render.scene.add_geometry("Part" + str(part), mesh, mtl)



    render.setup_camera(pinhole, ext_mat)


    try:
        # if no response for 10 seconds, skip

        img_o3d = render.render_to_image()
        print("Image rendered.")
    except Exception as e:
        print(f"Exception occurred during rendering: {e}")



    # If directory does not exist, create it
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    # Overwrite the image if it exists
    if os.path.exists(save_path):
        os.remove(save_path)
    # Save the image
    Image.fromarray(np.array(img_o3d)).save(save_path)


    print("Rendered image saved to " + save_path)
    print("Rendered image: ", np.array(img_o3d))



def render_parts(img_width, img_height, ext_mat, int_mat, part_meshes, part_mesh_id_list, visualize=False, show_orign = False):
    """
    this function assumes that all parts are already in the camera frame
    """

    pinhole = o3d.camera.PinholeCameraIntrinsic(int(img_width), int(img_height), int_mat[0, 0], int_mat[1, 1], int_mat[0, 2], int_mat[1, 2])

    # Create a perspective camera
    render = o3d.visualization.rendering.OffscreenRenderer(int(img_width), int(img_height))

    # Load the object
    render.scene.set_background([255.0, 255.0, 255.0, 255.0])  # RGBA

    # Load the object
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultLit"

    for pi, part_mesh in enumerate(part_meshes):
        color_id = [int(j) for j in part_mesh_id_list[pi].split(',')]
        color_id = min(color_id)
        print(f'Color id: {color_id} for part {part_mesh_id_list[pi]}')
        mtl.base_color = colors[color_id % len(colors)]
        try:
            render.scene.add_geometry("Part" + str(pi), part_mesh, mtl)
        except:
            o3d.io.write_triangle_mesh("debug_part.obj", part_mesh)
            raise ValueError("Cannot add part to scene")
            

    
    if show_orign:
        # add origin
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        coordinate_frame.transform(ext_mat)
        render.scene.add_geometry("origin", coordinate_frame, o3d.visualization.rendering.MaterialRecord())

    render.setup_camera(pinhole, np.eye(4))

    img_o3d = render.render_to_image()

    if visualize:
        plt.imshow(np.asarray(img_o3d))
        plt.show()

    
    
    return np.asarray(img_o3d)


def get_cam_pose_from_look_at(at, eye, up=(0,0,1)):
  # https://stackoverflow.com/questions/349050/calculating-a-lookat-matrix
  at = np.array(at)
  eye = np.array(eye)
  up = np.array(up)

  # Compute rotation matrix
  zaxis = (at - eye) / np.linalg.norm(at - eye)
  xaxis = np.cross(zaxis, up) / np.linalg.norm(np.cross(zaxis, up))
  yaxis = np.cross(zaxis, xaxis)
  cam_pose = np.eye(4)
  cam_pose[:3, 0] = xaxis
  cam_pose[:3, 1] = yaxis
  cam_pose[:3, 2] = zaxis

  cam_pose[:3, 3] = eye

  # note: this returns the camera pose in the world frame

  return cam_pose

