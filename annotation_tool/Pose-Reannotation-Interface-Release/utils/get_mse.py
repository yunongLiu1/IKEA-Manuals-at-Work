# camera_parameters_file = "camera_parameters.json"
from flask import Flask, jsonify, request
import trimesh
import numpy as np
import json
import os
from flask import Flask
from flask_cors import CORS
from flask import request
import locale
import base64
from io import BytesIO
from PIL import Image
import cv2
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt
import open3d as o3d
from pyvirtualdisplay import Display
import copy

Display().start()





def load_video(category, name, video_id):
    video_dir = '/root/Extend-IKEA-Manual-Annotation-Interface/backend/dynamic_dataset/video'
    video_path = os.path.join(video_dir, category, name, video_id, f"{video_id}.mp4")
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f'Could not open video {video_path}')
    
    return video

def get_frame_image(category, name, frame_id, video_id):
    video = load_video(category, name, video_id)
    frame = load_frame(json_data, category, name, frame_id, video_id)
    frame_time = frame['frame_time']
    video.set(cv2.CAP_PROP_POS_MSEC, frame_time*1000)
    ret, frame_img = video.read()
    if not ret:
        raise RuntimeError(f'Could not read frame {frame_id} from video {video_path}')
    
    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
    return frame_img

def load_frame(data, category, name, frame_id, video_id):

    for furniture in data:
        if furniture['category'] == category and furniture['name'] == name:
            for step in furniture['steps']:
                for video in step['video']:
                    if video['video_id'].split('watch?v=')[-1] == video_id:
                        for frame in video['frames']:
                            if int(frame['frame_id']) == int(frame_id):
                                return frame
    
    print('Frame not found')
    return None


def render_part(
    obj_path,
    part_ids,
    ext_mat,
    int_mat,
    img_width,
    img_height,
    save_path, debug=False):
    colors = [  [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0],[1.0, 0.0, 1.0, 1.0],[0.0, 1.0, 1.0, 1.0],
[1.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0], [0.0, 0.5, 0.0, 1.0], [0.0, 0.0, 0.5, 1.0]]

    print('ext_mat: ', ext_mat)
    print('int_mat: ', int_mat)
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
        img_o3d = render.render_to_image()
        print("Image rendered.")
    except Exception as e:
        print(f"Exception occurred during rendering: {e}")
        return None
    return img_o3d




### Compute mse between rendered image and mask
def get_mse(img1, img2):
    if img1.shape != img2.shape:
        print(f'Image shapes do not match, img1: {img1.shape}, img2: {img2.shape}')
        return None
    return np.sum((img1 - img2) ** 2) / np.sum(img2)

def mse_to_score(mse):
    return np.exp(-mse)

def get_mse_score(category, name, frame_id, video_id, json_path):
    obj_folder = '/root/Extend-IKEA-Manual-Annotation-Interface/backend/dynamic_dataset/parts'

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    frame = load_frame(json_data, category, name, frame_id, video_id)
    masks = frame['mask']

    masks_decoded = []

    for mask in masks:
        try:
            decoded_mask = mask_utils.decode(mask)
            masks_decoded.append(decoded_mask)
            print('Decode mask success, shape:', decoded_mask.shape)
            img_height, img_width = decoded_mask.shape[:2]
        except:
            # Handle decoding error
            print('Decoding error, using dummy mask')
            frame_img = get_frame_image(category, name, frame_id, video_id)
            img_height, img_width = frame_img.shape[:2]
            dummy_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            masks_decoded.append(dummy_mask)



    pose_imgs = []
    obj_path = obj_folder + '/' + category + '/' + name
    for pi, part_idxs in enumerate(frame['parts']):
        pose_estimation_path =f'./test_{part_idxs}.png'

        ext_mat = frame['extrinsics'][pi]
        ext_mat = np.array(ext_mat)
        
        int_mat = frame['intrinsics'][pi]
        int_mat = np.array(int_mat)
        if ext_mat.shape != (4, 4) or int_mat.shape != (3, 3):
            print('Invalid extrinsic or intrinsic matrix. Skip')
            pose_imgs.append(None)
            continue

        img = render_part(obj_path, part_idxs, ext_mat, int_mat, img_width, img_height, pose_estimation_path)
        pose_imgs.append(img)

    mse_scores = []
    mses = []
    
    for i, img in enumerate(pose_imgs):
        if img is None:
            continue
        img = np.array(img)
        # we only care binary mask
        img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0
        img = img > 0.5
        print(img.max(), img.min())
        
        plt.imshow(img)
        plt.savefig(f'/root/Pose-Reannotation-Interface/rendered_results/{i}_rendered.png')
        print('save rendered image to', f'/root/Pose-Reannotation-Interface/rendered_results/{i}_rendered.png')
        mask = 1-masks_decoded[i]
        print(mask.max(), mask.min())

        plt.imshow(mask)
        plt.savefig(f'/root/Pose-Reannotation-Interface/rendered_results/{i}_mask.png')
        print('save mask to', f'/root/Pose-Reannotation-Interface/rendered_results/{i}_mask.png')

        mse = get_mse(img, mask)
        mse_score = mse_to_score(mse)
        mse_scores.append(mse_score)
        mses.append(mse)
        print(f'MSE score: {mse_score}')

        
        plt.imshow((img - mask)*255)
        plt.savefig(f'/root/Pose-Reannotation-Interface/rendered_results/{i}.png')
        print('save image to', f'/root/Pose-Reannotation-Interface/rendered_results/{i}.png')

        print(f'MSE between rendered image and mask for part {i}: {mse}')

    return { 'mse_scores': mse_scores, 'mses': mses, 'pose_imgs': pose_imgs, 'masks': masks_decoded, 'frame': frame, 'img_width': img_width, 'img_height': img_height}






if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='Bench')
    parser.add_argument('--name', type=str, default='applaro')
    parser.add_argument('--frame_id', type=str, default='1161')
    parser.add_argument('--video_id', type=str, default='KPs0ik2FcsY')
    parser.add_argument('--data_path', type=str, default='./data.json')
    args = parser.parse_args()

    category = args.category
    name = args.name
    frame_id = args.frame_id
    video_id = args.video_id
    json_path = args.data_path
    print('category: ', category, 'name: ', name, 'frame_id: ', frame_id, 'video_id: ', video_id, 'json_path: ', json_path)


    # category = 'Bench'
    # name = 'applaro'
    # frame_id = '1161'
    # video_id = 'KPs0ik2FcsY'
    
    # output_folder = "output"
    # json_path = './data_truncated.json'

    result = get_mse_score(category, name, frame_id, video_id, json_path)

    print('result: ', result)
   