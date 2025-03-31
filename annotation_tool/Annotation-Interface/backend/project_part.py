import numpy as np
import cv2
import os
import open3d as o3d
import json
import numpy as np
from PIL import Image
from pyvirtualdisplay import Display
Display().start()
import argparse


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

    # Plot different parts in different colors
    colors = [  [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0],[1.0, 0.0, 1.0, 1.0],[0.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0], [0.0, 0.5, 0.0, 1.0], [0.0, 0.0, 0.5, 1.0]]


    idx = 0
    for part in part_ids:

        idx += 1
        curr_obj_path = os.path.join( obj_path,str(part).zfill(2) + '.obj')
        print('curr_obj_path', curr_obj_path)

        mesh = o3d.io.read_triangle_mesh(curr_obj_path)
        mtl.base_color = colors[idx]
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
    print("Rendered image: ", img_o3d)

def pose_estimation_and_render_parts(obj_path, ext_mat, int_mat, part_idxs, image_path, output_path):
    img = cv2.imread(image_path)
    width = img.shape[1]
    height = img.shape[0]

    print("Start rendering...")
    render_part(obj_path, part_idxs, ext_mat, int_mat, width, height, output_path)
    print("Done.")

    
   

if __name__ == '__main__':

    # points_2d = []
    # points_3d = []
    # part_idxs = ['0','1','3','2']
    # image_path = 'img_manually.png'
    # output_path = 'output.png'

    # Use argparse to get the arguments
    parser = argparse.ArgumentParser(description='Pose estimation and rendering')
    parser.add_argument('--json', type=str, default='./pose_estimation_data.json', help='path to the json file')

    args = parser.parse_args()
    json_path = args.json

    
    # Read the data from json file
    with open(json_path) as f:
        data = json.load(f)
        ext_mat = np.array(data['extrinsic'])
        int_mat = np.array(data['intrinsic'])
        part_idxs = data['part_idxs']
        image_path = data['image_path']
        output_path = data['output_path']
        obj_path = data['obj_path']
    print("Data loaded from " + json_path)
    print("Extrinsic matrix: ", ext_mat)
    
    # Print image width and height
    img = cv2.imread(image_path)
    width = img.shape[1]
    height = img.shape[0]



    pose_estimation_and_render_parts(obj_path,ext_mat, int_mat, part_idxs, image_path, output_path)

    # Write back the result and success to json file
    data['extrinsic'] = ext_mat.tolist()
    data['intrinsic'] = int_mat.tolist()
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    print("Done.")