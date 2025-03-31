from flask import Flask, request, jsonify , send_file
from flask_cors import CORS
# from pose_estimation import pose_estimation_and_render_parts
import json
import subprocess
import math
from segment_anything import sam_model_registry
import os
import locale
import torch
import cv2
import numpy as np
from pycocotools import mask as mask_utils
# Use matplotlib.use('Agg') to avoid the error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.data_utils import get_data, get_data, get_video, is_same_parts, contain_this_parts, parts_overlap
#  Import ExtendedSamPredictor from ./SAM/sam.py
from SAM.sam import SamPredictor, show_mask, show_points, get_inner_points, remove_mask_overlap
from users import users
import threading
import base64
import time
import requests
import argparse
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
frontend_host = os.getenv("FRONTEND_HOST", "localhost")
port = int(os.getenv("BACKEND_PORT", "8000"))
print(f"Starting server on port {port}")

lock = threading.Lock()
verification_file_lock = [0 for _ in range(120)]

## Global Variables
PREV_DATA_JSON_PATH = './annotator_data/data.json' # Change to your path for **full** data
VIDEO_BASE_PATH = './dynamic_dataset/video/'
# Set the environment variable to explicitly set the locale to UTF-8.
# To avoid: “Fatal Python error: config_get_locale_encoding: failed to get the locale encoding: nl_langinfo(CODESET) failed”。
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LANGUAGE"] = "en_US.UTF-8"
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# # TAPIR Model (for points tracking, disabled)
# CHECKPOINT_PATH_TAPIR = os.path.join(os.getcwd() ,"TAPIR/tapnet","checkpoints/tapir_checkpoint.npy")
# ckpt_state_tapir = np.load(CHECKPOINT_PATH_TAPIR, allow_pickle=True).item()
# params_tapir, state_tapir = ckpt_state_tapir['params'], ckpt_state_tapir['state']
# TAPIR_host = os.getenv("TAPIR_HOST", "localhost")
# TAPIR_port = int(os.getenv("TAPIR_PORT", "9000"))

## Load model here to avoid loading model every time
# Segment Anything Model
CHECKPOINT_PATH_SAM = os.path.join(os.getcwd() + '/SAM', "weights", "sam_vit_h_4b8939.pth")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH_SAM).to(device=DEVICE)

app = Flask(__name__)
CORS(app) # This will enable CORS for all routes


user_data = {}
############## Health Check #################
@app.route('/health', methods=['GET'])
def health_check():
    print("Start Health Checking.....")
    cmd_to_execute = 'python ./pose_estimation.py'
    conda_cmd = f'conda run -n ikea-video-backend {cmd_to_execute}'

    try:
        # Run the subprocess with a timeout of 3 minutes (180 seconds)
        subprocess.run(conda_cmd, shell=True, capture_output=True, text=True, timeout=30)
        print('Healthy!')
        return jsonify({"status": "healthy"}), 200
    except subprocess.TimeoutExpired:
        # If the script takes longer than 3 minutes, return an unhealthy status
        print('Take longer than 3 mins, unhealthy!')
        return jsonify({"status": "unhealthy"}), 503

############## Progress Bar #################
@app.route('/get-progress-bar', methods=['POST'])
def get_progress_bar():

    datas = request.json
    #print(datas)
    category = datas['category']
    name = datas['name']
    step_id = int(datas['step_id'].split('_')[-1])
    video_id = datas["videoPath"].split('/')[-1].split('_clip')[0]
    Projection = datas['Projection']

    user = datas['user']
    data_json_path = users[user]['data_json_path']
    data = get_data(data_json_path)

    currentModelFilePaths = datas["currentModelFilePaths"]
    
    objs = []
    for currentModelFilePath in currentModelFilePaths:
        obj = currentModelFilePath.split('/')[-1].split('.')[0]
        # Convert back from 00, 01 to 0, 1
        obj = str(int(obj))
        objs.append(obj)
    
    parts = ','.join(objs)

    total_annotation_needed_for_curr_part_curr_step = 0
    num_of_annotated_for_curr_part_curr_step = 0
    current_part = ''
    for furnitrue in data:
        if furnitrue['category'] == category and furnitrue['name'] == name:
            for step in furnitrue['steps']:
                if step['step_id'] == step_id:
                    for video in step['video']:
                        if video['video_id'].split('watch?v=')[-1] == video_id:
                            for frame in video["frames"]:
                                # #print(f'current_part: {current_part}, frame["parts"]: {frame["parts"]}, current_part in frame["parts"]: {current_part in frame["parts"]}')
                                for part in frame['parts']:
                                    # #print(f'part: {part}, parts: {parts}')
                                    # #print(f'is_same_parts(part, parts): {is_same_parts(part, parts)}')
                                    if is_same_parts(part, parts):
                                        current_part = part
                                        # #print(f'current_part: {current_part}')
                                        break
    for furnitrue in data:
        if furnitrue['category'] == category and furnitrue['name'] == name:
            for step in furnitrue['steps']:
                if step['step_id'] == step_id:
                    for video in step['video']:
                        if video['video_id'].split('watch?v=')[-1] == video_id:
                            for frame in video["frames"]:
                                # #print(f'current_part: {current_part}, frame["parts"]: {frame["parts"]}, current_part in frame["parts"]: {current_part in frame["parts"]}')
                                if current_part in frame['parts']:
                                    total_annotation_needed_for_curr_part_curr_step += 1
                                    if Projection:
                                        if frame['intrinsics'][frame['parts'].index(current_part)] != [] and frame['extrinsics'][frame['parts'].index(current_part)] != []:
                                            num_of_annotated_for_curr_part_curr_step += 1
                                    else:
                                        if frame['mask'][frame['parts'].index(current_part)] != {} and frame['mask'][frame['parts'].index(current_part)] != []:
                                            num_of_annotated_for_curr_part_curr_step += 1
    #print(f"Total annotation needed for current part {current_part} in current step {step_id} is {total_annotation_needed_for_curr_part_curr_step}")
    #print(f"Total annotation done for current part {current_part} in current step {step_id} is {num_of_annotated_for_curr_part_curr_step}")

                                            
    #  Step Level for video (How far they can leave current step for current video)

    total_annotation_needed_for_curr_step_curr_video = 0
    num_of_annotated_for_curr_step_curr_video = 0

    for furnitrue in data:
        if furnitrue['category'] == category and furnitrue['name'] == name:
            for step in furnitrue['steps']:
                if step['step_id'] == step_id:
                    for video in step['video']:
                        if video['video_id'].split('watch?v=')[-1] == video_id:
                            for frame in video["frames"]:
                                for part in frame['parts']:
                                    total_annotation_needed_for_curr_step_curr_video += 1
                                    if Projection:
                                        if frame['intrinsics'][frame['parts'].index(part)] != [] and frame['extrinsics'][frame['parts'].index(part)] != []:
                                            num_of_annotated_for_curr_step_curr_video += 1
                                    else:
                                        if frame['mask'][frame['parts'].index(part)] != {} and frame['mask'][frame['parts'].index(part)] != []:
                                            num_of_annotated_for_curr_step_curr_video += 1
                            

    #print(f"Total annotation needed for current step {step_id} in current video {video_id} is {total_annotation_needed_for_curr_step_curr_video}")
    #print(f"Total annotation done for current step {step_id} in current video {video_id} is {num_of_annotated_for_curr_step_curr_video}")

    # Step Level for furniture (How far they can leave current step)
    total_annotation_needed_for_curr_step = 0
    num_of_annotated_for_curr_step = 0
    for furnitrue in data:
        if furnitrue['category'] == category and furnitrue['name'] == name:
            for step in furnitrue['steps']:
                if step['step_id'] == step_id:
                    for video in step['video']:
                        for frame in video["frames"]:
                            for part in frame['parts']:
                                total_annotation_needed_for_curr_step += 1
                                if Projection:
                                    if frame['intrinsics'][frame['parts'].index(part)] != [] and frame['extrinsics'][frame['parts'].index(part)] != []:
                                        num_of_annotated_for_curr_step += 1
                                else:
                                    if frame['mask'][frame['parts'].index(part)] != {} and frame['mask'][frame['parts'].index(part)] != []:
                                        num_of_annotated_for_curr_step += 1
    #print(f"Total annotation needed for current step {step_id} in current furniture {name} is {total_annotation_needed_for_curr_step}")
    #print(f"Total annotation done for current step {step_id} in current furniture {name} is {num_of_annotated_for_curr_step}")

    # Video level for all steps (How far they can leave current video)
    total_annotation_needed_for_curr_video = 0
    num_of_annotated_for_curr_video = 0

    for furnitrue in data:
        if furnitrue['category'] == category and furnitrue['name'] == name:
            for step in furnitrue['steps']:
                for video in step['video']:
                    if video['video_id'].split('watch?v=')[-1] == video_id:
                        for frame in video["frames"]:
                            for part in frame['parts']:
                                total_annotation_needed_for_curr_video += 1
                                if Projection:
                                    if frame['intrinsics'][frame['parts'].index(part)] != [] and frame['extrinsics'][frame['parts'].index(part)] != []:
                                        num_of_annotated_for_curr_video += 1
                                else:
                                    if frame['mask'][frame['parts'].index(part)] != {} and frame['mask'][frame['parts'].index(part)] != []:
                                        num_of_annotated_for_curr_video += 1
    #print(f"Total annotation needed for current video {video_id} in current furniture {name} is {total_annotation_needed_for_curr_video}")
    #print(f"Total annotation done for current video {video_id} in current furniture {name} is {num_of_annotated_for_curr_video}")

    # Furniture level for all steps (How far they can leave current furniture)
    total_annotation_needed_for_curr_furniture = 0
    num_of_annotated_for_curr_furniture = 0

    for furnitrue in data:
        if furnitrue['category'] == category and furnitrue['name'] == name:
            for step in furnitrue['steps']:
                for video in step['video']:
                    for frame in video["frames"]:
                        for part in frame['parts']:
                            total_annotation_needed_for_curr_furniture += 1
                            if Projection:
                                if frame['intrinsics'][frame['parts'].index(part)] != [] and frame['extrinsics'][frame['parts'].index(part)] != []:
                                    num_of_annotated_for_curr_furniture += 1
                            else:
                                if frame['mask'][frame['parts'].index(part)] != {} and frame['mask'][frame['parts'].index(part)] != []:
                                    num_of_annotated_for_curr_furniture += 1
    #print(f"Total annotation needed for current furniture {name} is {total_annotation_needed_for_curr_furniture}")
    #print(f"Total annotation done for current furniture {name} is {num_of_annotated_for_curr_furniture}")

    # Total annotation needed for all furniture
    total_annotation_needed_for_all_furniture = 0
    num_of_annotated_for_all_furniture = 0

    for furnitrue in data:
        for step in furnitrue['steps']:
            for video in step['video']:
                for frame in video["frames"]:
                    for part in frame['parts']:
                        total_annotation_needed_for_all_furniture += 1
                        # #print(f'part: {part}, parts: {parts}, frame["parts"]: {frame["parts"]}')
                        if Projection:
                            if frame['intrinsics'][frame['parts'].index(part)] != [] and frame['extrinsics'][frame['parts'].index(part)] != []:
                                num_of_annotated_for_all_furniture += 1
                        else:
                            if frame['mask'][frame['parts'].index(part)] != {} and frame['mask'][frame['parts'].index(part)] != []:
                                num_of_annotated_for_all_furniture += 1
    #print(f"Total annotation needed for all furniture is {total_annotation_needed_for_all_furniture}")
    #print(f"Total annotation done for all furniture is {num_of_annotated_for_all_furniture}")

    return jsonify({
        "total_annotation_needed_for_curr_part_curr_step": total_annotation_needed_for_curr_part_curr_step, 
        "num_of_annotated_for_curr_part_curr_step": num_of_annotated_for_curr_part_curr_step, 
        "total_annotation_needed_for_curr_step_curr_video": total_annotation_needed_for_curr_step_curr_video, 
        "num_of_annotated_for_curr_step_curr_video": num_of_annotated_for_curr_step_curr_video, 
        "total_annotation_needed_for_curr_step": total_annotation_needed_for_curr_step, 
        "num_of_annotated_for_curr_step": num_of_annotated_for_curr_step, 
        "total_annotation_needed_for_curr_video": total_annotation_needed_for_curr_video, 
        "num_of_annotated_for_curr_video": num_of_annotated_for_curr_video, 
        "total_annotation_needed_for_curr_furniture": total_annotation_needed_for_curr_furniture, 
        "num_of_annotated_for_curr_furniture": num_of_annotated_for_curr_furniture, 
        "total_annotation_needed_for_all_furniture": total_annotation_needed_for_all_furniture, 
        "num_of_annotated_for_all_furniture": num_of_annotated_for_all_furniture}), 200



########## Prev Mask Image ########
@app.route('/get-prev-mask', methods=['POST'])
def get_prev_mask():
    print('request.json')
    user = request.json['user']
    Category = request.json['category']
    SubCategory = request.json['subCategory']
    Step = request.json['step']
    Video = request.json['video']
    Part = request.json['part']
    Frame = request.json['frame']
    mask, videoPath, time = get_prev_mask_helper(user, Category, SubCategory, Step, Video, Part, Frame)

    success = False
    if videoPath and os.path.exists(videoPath):
        curr_video = cv2.VideoCapture(videoPath)
        if not curr_video.isOpened():
            print(f'###### Warning: could not open video {videoPath}')
            success = False
        try:
            curr_video.set(cv2.CAP_PROP_POS_MSEC, time*1000)
            success, image = curr_video.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print('success', success)
            assert success
        except:
            print(f'###### Warning: could not read frame from video {videoPath}')


    #Save it to base64
    mask_img = Image.fromarray(mask.astype(np.uint8))
    buffered = BytesIO()
    mask_img.save(buffered, format='PNG')
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Resize the mask array to match the dimensions of the image array
    mask_resized = np.stack((mask,)*3, axis=-1)

    try:
        frame_img_overlay = cv2.addWeighted(image, 0.5, mask_resized, 0.5, 0)


        buffered = BytesIO()
        Image.fromarray(frame_img_overlay).save(buffered, format='PNG')
        base64_image_overlay = base64.b64encode(buffered.getvalue()).decode('utf-8')
    except:
        base64_image_overlay = base64_image
        print('Could not overlay mask on image')
        print('mask:', mask_resized)
        print('image:', image)
        print('mask shape:', mask_resized.shape)
        print('image shape:', image.shape)
        print('mask dtype:', mask_resized.dtype)
        print('image dtype:', image.dtype)

    reason = 'Wrong Part'

    return jsonify({"image": base64_image, "image_overlay": base64_image_overlay, 'reason': reason}), 200

def get_prev_mask_helper(user, Category, SubCategory, Step, Video, Part, Frame):
    data_json_path = users[user]['data_json_path']
    prev_data_json_path = PREV_DATA_JSON_PATH
    
    video_id = Video.split('/')[-2]
    part_ids = [part.split('/')[-1].split('.')[0] for part in Part]
    part_ids = [int(part_id) for part_id in part_ids]
    part_ids = [str(part_id) for part_id in part_ids]
    part_ids = ','.join(part_ids)
    step_id = Step.split('_')[-1]
    video_base_path = VIDEO_BASE_PATH

    mask = np.zeros((1080, 1920))
    prev_data = get_data(prev_data_json_path)
    print(video_id, part_ids, step_id, Frame)
    print('prev_data_json_path', prev_data_json_path)   
    for furniture in prev_data:
        if furniture['category'] == Category and furniture['name'] == SubCategory:
            for step in furniture['steps']:
                if int(step['step_id']) == int(step_id):
                    for video in step['video']:
                        if video['video_id'].split('watch?v=')[-1] == video_id:
                            videoPath = video_base_path + furniture['category'] + '/' + furniture['name'] + '/' + video['video_id'].split('/watch?v=')[-1] + '/' + video['video_id'].split('/watch?v=')[-1] + '.mp4'
                            for frame in video['frames']:
                                if int(frame['frame_id']) == int(Frame):
                                    
                                    for part in frame['parts']:
                                        found_mask = False
                                        if contain_this_parts(part, part_ids):
                                            mask = frame['mask'][frame['parts'].index(part)]
                                            try:
                                                mask = mask_utils.decode(mask) *255
                                            except:
                                                mask = np.zeros((1080, 1920))
                                            print('Found mask')
                                            found_mask = True
                                            
                                            
                                            time = frame['frame_time']
                                            break
                                    if not found_mask:
                                        overlaps = []
                                        for part in frame['parts']:
                                            overlap = parts_overlap(part, part_ids)
                                            overlaps.append(overlap)
                                        if sum(overlaps) == 0:
                                            mask = np.zeros((1080, 1920))
                                        else:
                                            mask = frame['mask'][overlaps.index(max(overlaps))]
                                            try:
                                                mask = mask_utils.decode(mask) *255
                                            except:
                                                mask = np.zeros((1080, 1920))
    return mask, videoPath, time
                                        


############Change Command (Change a bunch of Status)########
def decomposate_image_path(image_path):
    image_path_refine = image_path.replace('//','/')
    # print('decomposing: ',image_path)
    frame = image_path_refine.split('/')[-1].split('frame_')[-1].split('_')[0]
    category = image_path_refine.split('/')[3]
    name = image_path_refine.split('/')[4]
    step = image_path_refine.split('/')[5].split('_')[-1]
    video_id = image_path_refine.split('/')[6]
    part_id =   image_path_refine.split('/')[-1].split('_part_')[-1].split('_')[0]
    # print(f' category: {category}, name: {name}, step: {step}, video_id: {video_id}')
    return category, name, step, video_id, frame, part_id

@app.route('/change_command/', methods=['POST'])
def change_command():
    datas = request.json
    category = datas['FurnitureCategory']
    name = datas['FurnitureName']
    video_id = datas['VideoID']
    part_id = datas['PartID']
    frame_from = datas['FrameFrom']
    frame_to = datas['FrameTo']
    images = datas['images']
    classification = datas['Classification']
    number = int(datas['Number'])
    frame_ls = []
    comments = 'auto'
    try:
        frame_from = int(frame_from)
        frame_to = int(frame_to)
    except:
        return jsonify({"error": "Please input valid frame number"}), 400

    key = category + '_' + name + '_' + video_id + '_' + part_id
    
    if 'pose-estimation_check/' in images[0]:
        base_folder = os.path.join(os.getcwd(), '/'.join(images[0].split('/pose-estimation_check/')[:-1]))
    elif 'mask_check/' in images[0]:
        base_folder = os.path.join(os.getcwd(), '/'.join(images[0].split('/mask_check/')[:-1]))
    else:
        base_folder = os.path.join(os.getcwd(), '/'.join(images[0].split('/img/')[:-1]))
    print(f'base_folder: {base_folder}')
    images = [image.split('quality_check')[-1] for image in images]

    for image in images:
        if '/img/' in image:
            continue
        if 'blank.png' in image:
            continue
        if 'part' not in image:
            continue
        category, name, step, video_id, frame, part_id = decomposate_image_path(image)
        curr_key = category + '_' + name + '_' + video_id + '_' + part_id
        if curr_key == key:
            
            if int(frame) >= frame_from and int(frame) <= frame_to:
                if frame not in frame_ls:
                    frame_ls.append(frame)
                print(f'current key {curr_key} for image {image}')
                print(f'requested key {key}')

                # Update check_data.json
                check_data_path = os.path.join(base_folder, 'check_data.json')
                if image.split('/')[-1] == 'blank.png':
                    continue
                # Update check_data.json
                if int(number) != -1:
                    while verification_file_lock[number] != 0:
                        wait_time = 0.1
                        print(f'waiting for {number} for {wait_time} seconds')
                        time.sleep(wait_time)
                        
                    verification_file_lock[number] = 1
                    data = json.load(open(check_data_path, 'r'))
                    verification_file_lock[number] = 0
                
                if image.split('/quality_check')[-1] in data:
                    data[image.split('/quality_check')[-1]]['status'] = classification
                    data[image.split('/quality_check')[-1]]['comments'] = comments
                else:
                    data[image.split('/quality_check')[-1]] = {}
                    data[image.split('/quality_check')[-1]]['status'] = classification
                    data[image.split('/quality_check')[-1]]['comments'] = comments
                
                if int(number) != -1:
                    while verification_file_lock[number] != 0:
                        wait_time = 0.1
                        print(f'waiting for {number} for {wait_time} seconds')
                        time.sleep(wait_time)
                        
                    verification_file_lock[number] = 1
                    data = json.dump(data, open(check_data_path, 'w'), indent=4)
                    verification_file_lock[number] = 0
                
        print('finish changing status')
    
    if len(frame_ls) == 0:
        return jsonify({"message": "No annotation found"}), 200
    else:
        message = f'Change {len(frame_ls)} frames from {frame_from} to {frame_to} to {classification}'
        return jsonify({"message": message}), 200
            

############# Change Status ################
@app.route('/change_status/', methods=['POST'])
def change_status():
    datas = request.json
    print(datas)
    classification = datas['Classification']
    print(f'classification: {classification}')
    images = datas['Images']
    print(f'images: {images}')
    number = int(datas['Number'])
    print(f'number: {number}')
    try:
        comments = datas['Command']
    except:
        comments = ''
    # images = images.split(',')
    print(f'comments: {comments}')
    if 'pose-estimation_check/' in images[0]:
        base_folder = os.path.join(os.getcwd(), '/'.join(images[0].split('/pose-estimation_check/')[:-1]))
    elif 'mask_check/' in images[0]:
        base_folder = os.path.join(os.getcwd(), '/'.join(images[0].split('/mask_check/')[:-1]))
    else:
        base_folder = os.path.join(os.getcwd(), '/'.join(images[0].split('/img/')[:-1]))
    print(f'base_folder: {base_folder}')

    
    check_data_path = os.path.join(base_folder, 'check_data.json')
    for image in images:
        if image.split('/')[-1] == 'blank.png':
            continue
        # Update check_data.json
        if int(number) != -1:
            while verification_file_lock[number] != 0:
                wait_time = 0.1
                print(f'waiting for {number} for {wait_time} seconds')
                time.sleep(wait_time)
                
            verification_file_lock[number] = 1
            data = json.load(open(check_data_path, 'r'))
            verification_file_lock[number] = 0
        
        if image.split('/quality_check')[-1] in data:
            data[image.split('/quality_check')[-1]]['status'] = classification
            data[image.split('/quality_check')[-1]]['comments'] = comments
        else:
            data[image.split('/quality_check')[-1]] = {}
            data[image.split('/quality_check')[-1]]['status'] = classification
            data[image.split('/quality_check')[-1]]['comments'] = comments
        
        if int(number) != -1:
            while verification_file_lock[number] != 0:
                wait_time = 0.1
                print(f'waiting for {number} for {wait_time} seconds')
                time.sleep(wait_time)
                
            verification_file_lock[number] = 1
            data = json.dump(data, open(check_data_path, 'w'), indent=4)
            verification_file_lock[number] = 0
        
    print('finish changing status')
    # 
    return jsonify({"success": True}), 200

    


#############Image Gallery################

def get_key_and_video_id(img):
    # print(f'img: {img}')
    offset = 0

    if 'pose-estimation_check/' in img:
        img = img.split('pose-estimation_check/')[-1]
    elif 'pose-estimation_check_new_color/' in img:
        img = img.split('pose-estimation_check_new_color/')[-1]
    elif 'mask_check/' in img:
        img = img.split('mask_check/')[-1]
    else :
        img = img.split('img/')[-1]
    
    # print(img)
    furniture_category= img.split('/')[offset]
    furniture_name = img.split('/')[offset+1]
    step_id = img.split('/')[offset+2]
    step_id = step_id.split('_')[-1]
    video_id = img.split('/')[offset+3]
    frame_id = img.split('/')[offset+4].split('frame_')[-1].split('_')[0].split('.')[0]
    
    part_id = img.split('/')[offset+4].split('_part_')[-1].split('_')[0].split('.')[0]
    key = furniture_category + '_' + furniture_name

    # print(f'key {key}, video_id {video_id}, frame_id {frame_id}, part_id {part_id}')
    

    return key, video_id, int(frame_id), part_id

def get_prefix(img):
    key, video_id, frame_id, part_id = get_key_and_video_id(img)
    return key + '_' + video_id + '_' + str(frame_id) + '_' + part_id

def get_key_and_video_id_2(img):
    offset = 0

    if 'pose-estimation_check/' in img:
        img = img.split('pose-estimation_check/')[-1]
    elif 'pose-estimation_check_new_color/' in img:
        img = img.split('pose-estimation_check_new_color/')[-1]
    elif 'mask_check/' in img:
        img = img.split('mask_check/')[-1]
    else :
        img = img.split('img/')[-1]
    
    furniture_category= img.split('/')[offset]
    furniture_name = img.split('/')[offset+1]
    step_id = img.split('/')[offset+2]
    # step_id = step_id.split('_')[-1]
    video_id = img.split('/')[offset+3]
    frame_id = img.split('/')[offset+4].split('frame_')[-1].split('_')[0].split('.')[0]
    if '_part_' in img:
        part_id = img.split('/')[offset+4].split('_part_')[-1].split('_')[0].split('.')[0]
    else:
        part_id = None

    return furniture_category, furniture_name, video_id, int(frame_id), part_id, step_id



@app.route('/image_viz/<path:filename>')
def get_image_viz(filename):
    video_path = os.path.join(filename)

    if os.path.exists(video_path):
        # Add a timestamp to the image URL
        response = send_file(video_path, mimetype='image/jpg')
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    else:
        return 'File not found', 404


################### Return Data Stats ####################
@app.route('/data-stat')
def get_data_stat():
    cmd_to_execute = 'python ./utils/get_data_stat.py'
    conda_cmd = f'conda run -n ikea-video-backend {cmd_to_execute}'
    # Run the subprocess with a timeout of 3 minutes (180 seconds)
    result = subprocess.run(conda_cmd, shell=True, capture_output=True, text=True, timeout=180)
    output_string = result.stdout
    print(output_string)
    print(f'result {result}')

    return jsonify({"data": output_string})



#################### Login ####################
@app.route('/login', methods=['POST'])
def login():
    # Extract the username and password from the request
    datas = request.json
    username = datas['username']
    password = datas['password']
    
    # Check if the username and password are correct
    if username in users and users[username]['password'] == password:
        # The username and password are correct, so return a success message and the data paths
        user_data[username] = {}
        user_data[username]['current_video_time'] = 0
        user_data[username]['save_mask_done'] = True
        user_data[username]['save_project_done'] = True
        user_data[username]['segmentation_done'] = True
        user_data[username]['pose_estimation_done'] = True
        return jsonify({
            "message": "Login successful",
            "data_json_path": users[username]['data_json_path']
        }), 200

    else:
        # The username or password are incorrect, so return an error message
        return jsonify({"message": "Invalid username or password"}), 401


#################### Image Rendering ####################
@app.route('/image/<path:filename>')
def get_image(filename):
    video_path = os.path.join(filename)

    if os.path.exists(video_path):
        # Add a timestamp to the image URL
        response = send_file(video_path, mimetype='image/jpg')
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    else:
        return 'File not found', 404
###################Video Rendering####################
@app.route('/video/<path:filename>')
def load_video(filename):
    video_path = os.path.join(filename)

    if os.path.exists(video_path):
        # Add a timestamp to the image URL
        response = send_file(video_path, mimetype='video/mp4')
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    else:
        return 'File not found', 404



#################### Pose Estimation ####################
@app.route('/create-pose-estimation-data', methods=['POST'])
def create_pose_estimation_data():
    print('################ Creating pose estimation data! ##############')
    datas = request.json
    user = datas['user']
    user_data[user]['pose_estimation_done'] = False
    fps = user_data[user]['fps']

    # make coord 2d and 3d from dict to list:
    points_2d = []
    points_3d = []
    for data in datas["3d-coordinates"]:
        point_3d = (data['x'], data['y'],data['z'])
        points_3d.append(point_3d)

    for data in datas["2d-coordinates"]:
        point_2d = (data['x'], data['y'])
        points_2d.append(point_2d)
    
    category = datas["Category"]
    name = datas["SubCategory"]
    step_id = datas["Object"]
    currentModelFilePaths = datas["currentModelFilePaths"]

    current_video = user_data[user]['selected_video']


    #e.g. parts = "0,1,2"
    parts = [currentModelFilePath.split('/')[-1].split('.')[0] for currentModelFilePath in currentModelFilePaths]
    if len(parts) == 0:
        print('currentModelFilePaths',currentModelFilePaths)
        user_data[user]['pose_estimation_done'] = True
        return jsonify({"error": "No part selected. Please select at least one part for segmentation."}), 500
    # change 00 -> 0, 01 -> 1, 02 -> 2
    parts = [str(int(part)) for part in parts]
    prefix = 'http://'+frontend_host+':' + str(port) + '/video/'
    video_path = os.path.join(os.getcwd() ,current_video.split(prefix)[-1])
    output_path  = os.path.join(os.getcwd(), 'dynamic_dataset/pose-estimation' , category, name, step_id)
    obj_path = os.path.join(os.getcwd(), 'dynamic_dataset/parts' , category, name)


    print("video_path: ", video_path)
    print("output_path: ", output_path)
    ##### Check if the parts overlap with previous frame #####
    data = {}
    data['previous_frame_data'] = {}
    data['same_parts_exist'] = False
    data['aux_ext_mat'] = None

    data_json_path = users[user]['data_json_path']
    json_data = get_data(data_json_path)

    currentFrameTime = user_data[user]['current_video_time']
    frame_id = int(currentFrameTime*fps)

    for prev_data in json_data:
        if prev_data["category"] == category and prev_data["name"] == name:
            step_id = current_video.split('/')[-3].split('_')[-1]
            video_id = current_video.split('/')[-1].split('_clip')[0]
            # frame_id = current_video.split('/')[-1].split('_')[-1].split('.')[0]

            for step in prev_data['steps']:
                if step['step_id'] == int(step_id):
                    for video in step['video']:
                        if video['video_id'].split('watch?v=')[-1] == video_id:
                            for j,frame in enumerate(video['frames']):
                                # Get extrinsic matrix from previous frame if it exists

                                if int(frame['frame_id']) == int(frame_id):
                                    data['previous_frame_data']=frame
                                    try:
                                        for i in range(len(video['frames'][j-1]['parts'])):
                                            if is_same_parts(','.join(parts), video['frames'][j-1]['parts'][i]) and video['frames'][j-1]['extrinsics'][i] != []:
                                                    data['aux_ext_mat'] = video['frames'][j-1]['extrinsics'][i]
                                    except:
                                        data['aux_ext_mat'] = None
                                    for i in range(len(frame['parts'])):
                                        if is_same_parts(','.join(parts), frame['parts'][i]) and frame['extrinsics'][i] != []:
                                            data['same_parts_exist'] = True
                                            break
                                if data['same_parts_exist']:
                                    break
                            if data['same_parts_exist']:
                                break
                    if data['same_parts_exist']:
                        break
            if data['same_parts_exist']:
                break
    print('Finish data loading')
    part_ls = []
    if len(data['previous_frame_data']) > 0:
        for part in data['previous_frame_data']['parts']:
            part_ls.append(part.split(','))

        # if parts has overlap with previous frame, raise error
        for part in parts:
            if part in part_ls:
                user_data[user]['pose_estimation_done'] = True
                return jsonify({"success": False, "msg": "Part overlap with previous frame." ,"imagePath": video_path})
    
    # if parts has no overlap with previous frame, and part_ls is not empty, we need to show the previous frame
    data['show_previous_frame'] = True if len(part_ls) > 0 else False
    
    # Load image in json
    data['image'] = user_data[user]['current_working_image']

    # save points_2d, points_3d, part_idxs, video_path, output_path to json file
    
    data["points_2d"] = points_2d
    data["points_3d"] = points_3d
    data["part_idxs"] = ','.join(parts)
    data["video_path"] = video_path
    data["output_path"] = os.path.join(output_path, user + '_' +  current_video.split('/')[-1].split('.')[0]+str(int(currentFrameTime*fps))+'_parts'+ ''.join(parts) + '.jpg')

    data["obj_path"] = obj_path
    data["category"] = category
    data["name"] = name
    data["step_id"] = step_id
    data["frame_id"] = frame_id
    data["video_id"] = current_video.split('/')[-1].split('_clip')[0]
    data["currentFrameTime"] = currentFrameTime

    user = datas['user']
    pose_estimation_data_dir = os.path.join(os.getcwd(), 'dynamic_dataset/pose-estimation/pose-estimation-data' , category, name, step_id)
    pose_estimation_data_path = pose_estimation_data_dir + '/' + user + '_' +  current_video.split('/')[-1].split('.')[0]+str(int(currentFrameTime*fps))+'_parts'+ ''.join(parts) + '.json'
    user_data[user]['pose_estimation_data_path'] = pose_estimation_data_path
    # if file not exist, creat the file:
    if not os.path.exists(pose_estimation_data_dir):
        os.makedirs(pose_estimation_data_dir)
    with open(pose_estimation_data_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)
    print('pose_estimation_data_path', pose_estimation_data_path)
    return jsonify({"success": True, "msg": "Pose estimation data created."}), 200

@app.route('/pose-estimation', methods=['POST'])
def pose_estimation():
    datas = request.json
    user = datas['user']
    user_data[user]['pose_estimation_done'] = False
    fps = user_data[user]['fps']

    # make coord 2d and 3d from dict to list:
    points_2d = []
    points_3d = []
    for data in datas["3d-coordinates"]:
        point_3d = (data['x'], data['y'],data['z'])
        points_3d.append(point_3d)

    for data in datas["2d-coordinates"]:
        point_2d = (data['x'], data['y'])
        points_2d.append(point_2d)
    
    category = datas["Category"]
    name = datas["SubCategory"]
    step_id = datas["Object"]
    currentModelFilePaths = datas["currentModelFilePaths"]

    current_video = user_data[user]['selected_video']


    #e.g. parts = "0,1,2"
    parts = [currentModelFilePath.split('/')[-1].split('.')[0] for currentModelFilePath in currentModelFilePaths]
    if len(parts) == 0:
        #print('currentModelFilePaths',currentModelFilePaths)
        user_data[user]['pose_estimation_done'] = True
        return jsonify({"error": "No part selected. Please select at least one part for segmentation."}), 500
    # change 00 -> 0, 01 -> 1, 02 -> 2
    parts = [str(int(part)) for part in parts]
    prefix = 'http://'+frontend_host+':' + str(port) + '/video/'
    video_path = os.path.join(os.getcwd() ,current_video.split(prefix)[-1])
    output_path  = os.path.join(os.getcwd(), 'dynamic_dataset/pose-estimation' , category, name, step_id)
    obj_path = os.path.join(os.getcwd(), 'dynamic_dataset/parts' , category, name)


    #print("video_path: ", video_path)
    #print("output_path: ", output_path)
    ##### Check if the parts overlap with previous frame #####
    data = {}
    data['previous_frame_data'] = {}
    data['same_parts_exist'] = False
    data['aux_ext_mat'] = None

    data_json_path = users[user]['data_json_path']
    json_data = get_data(data_json_path)

    currentFrameTime = user_data[user]['current_video_time']
    frame_id = int(currentFrameTime*fps)

    step_id = current_video.split('/')[-3].split('_')[-1]
    video_id = current_video.split('/')[-1].split('_clip')[0]
    for prev_data in json_data:
        if prev_data["category"] == category and prev_data["name"] == name:
            step_id = current_video.split('/')[-3].split('_')[-1]
            video_id = current_video.split('/')[-1].split('_clip')[0]
            # frame_id = current_video.split('/')[-1].split('_')[-1].split('.')[0]

            for step in prev_data['steps']:
                if step['step_id'] == int(step_id):
                    for video in step['video']:
                        if video['video_id'].split('watch?v=')[-1] == video_id:
                            for j,frame in enumerate(video['frames']):
                                # Get extrinsic matrix from previous frame if it exists

                                if int(frame['frame_id']) == int(frame_id):
                                    data['previous_frame_data']=frame
                                    try:
                                        for i in range(len(video['frames'][j-1]['parts'])):
                                            if is_same_parts(','.join(parts), video['frames'][j-1]['parts'][i]) and video['frames'][j-1]['extrinsics'][i] != []:
                                                    data['aux_ext_mat'] = video['frames'][j-1]['extrinsics'][i]
                                    except:
                                        data['aux_ext_mat'] = None
                                    for i in range(len(frame['parts'])):
                                        if is_same_parts(','.join(parts), frame['parts'][i]) and frame['extrinsics'][i] != []:
                                            data['same_parts_exist'] = True
                                            break
                                if data['same_parts_exist']:
                                    break
                            if data['same_parts_exist']:
                                break
                    if data['same_parts_exist']:
                        break
            if data['same_parts_exist']:
                break

    part_ls = []
    if len(data['previous_frame_data']) > 0:
        for part in data['previous_frame_data']['parts']:
            part_ls.append(part.split(','))

        # if parts has overlap with previous frame, raise error
        for part in parts:
            if part in part_ls:
                user_data[user]['pose_estimation_done'] = True
                return jsonify({"success": False, "msg": "Part overlap with previous frame." ,"imagePath": video_path})
    
    # if parts has no overlap with previous frame, and part_ls is not empty, we need to show the previous frame
    data['show_previous_frame'] = True if len(part_ls) > 0 else False
    
    # Load image in json
    data['image'] = user_data[user]['current_working_image']

    # save points_2d, points_3d, part_idxs, video_path, output_path to json file
    
    data["points_2d"] = points_2d
    data["points_3d"] = points_3d
    data["part_idxs"] = ','.join(parts)
    data["video_path"] = video_path
    data["output_path"] = os.path.join(output_path, user + '_' +  current_video.split('/')[-1].split('.')[0]+str(int(currentFrameTime*fps))+'_parts'+ ''.join(parts) + '_temp.jpg')

    data["obj_path"] = obj_path
    data["category"] = category
    data["name"] = name
    data["step_id"] = step_id
    data["frame_id"] = frame_id
    data["video_id"] = current_video.split('/')[-1].split('_clip')[0]
    data["currentFrameTime"] = currentFrameTime
    data['json_path_for_curr_user'] = users[user]['data_json_path']

    user = datas['user']

    pose_estimation_data_dir = os.path.join(os.getcwd(), 'dynamic_dataset/pose-estimation/pose-estimation-data' , category, name, step_id)
    pose_estimation_data_path = pose_estimation_data_dir + '/' + user + '_' +  current_video.split('/')[-1].split('.')[0]+str(int(currentFrameTime*fps))+'_parts'+ ''.join(parts) + '.json'
    user_data[user]['pose_estimation_data_path'] = pose_estimation_data_path
    # if file not exist, creat the file:
    if not os.path.exists(pose_estimation_data_dir):
        os.makedirs(pose_estimation_data_dir)
    with open(pose_estimation_data_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)

    #print("Start pose estimation...")
    cmd_to_execute = 'python ' + './pose_estimation.py' + ' --json ' + pose_estimation_data_path
    # Construct the Conda command
    conda_cmd = f'conda run -n ikea-video-backend {cmd_to_execute}'

    ret = subprocess.run(conda_cmd, shell=True, capture_output=True, text=True)

    # ret = subprocess.run(cmd_to_execute, shell=True, capture_output=True, text=True)
    #print(" Pose estimation finished")

    return_code = ret.returncode
    if "SolvePnP failed" in ret.stdout:
        #print ("SolvePnP failed!!")
        user_data[user]['pose_estimation_done'] = True
        return jsonify({"error": "SolvePnP failed."}), 500 #return the original image path
    elif "Not enough keypoints" in ret.stdout:
        #print ("Not enough keypoints!!")
        user_data[user]['pose_estimation_done'] = True
        return jsonify({"error": "Not enough keypoints. Please select at least 4 keypoints."}), 500
    elif "Keypoints shape mismatch" in ret.stdout:
        #print ("Keypoints shape mismatch!!")
        user_data[user]['pose_estimation_done'] = True
        return jsonify({"error": "Keypoints shape mismatch."}), 500


    print('ret', ret)
    print('return_code', return_code)
    output_paths = []
    for a in range(6):
        # print(data["output_path"])
        img_directory = os.path.dirname(data["output_path"])
        # print(f'image directory: {img_directory}')
        if a//2 == 0:
          axis = 'x'
        elif a//2 == 1:
            axis = 'y'
        else:
            axis = 'z'
        angle = '90' if a%2 == 0 else '270'
        
        diff_view_output_path = img_directory + '/'+ data["video_id"] + '_'+ f'frame_{frame["frame_id"]}_pose_estimation_{axis}{angle}.png'
        # if os.path.exists(diff_view_output_path):
        output_paths.append(diff_view_output_path)
            
    output_paths.append(data["output_path"])


    for i in range(len(output_paths)):
        output_paths[i] = 'http://'+frontend_host+':' + str(port) +'/image/' + os.path.relpath(output_paths[i], os.getcwd())
        #print("output_path: ", output_path)

    
    # output_path = 'http://'+frontend_host+':' + str(port) +'/image/' + os.path.relpath(data["output_path"], os.getcwd())
    print("output_paths: ", output_paths)

    # Read ext_mat and int_mat from pose_estimation json file
    # with open(pose_estimation_json_path) as f:
    user = datas['user']
    pose_estimation_data_path = user_data[user]['pose_estimation_data_path']
    with open(pose_estimation_data_path) as f:
        pose_estimation_data = json.load(f)
    try:
        ext_mat = pose_estimation_data["extrinsic"]
        int_mat = pose_estimation_data["intrinsic"]
    except:
        user_data[user]['pose_estimation_done'] = True
        return jsonify({"error": ["Pose estimation failed."]}), 500

    user_data[user]['pose_estimation_done'] = True
    user_data[user]['output_path'] = output_paths[-1]
    return jsonify({"imagePath": output_paths, "messages": ["Pose estimation finished!"]}), 200

# #################### Causal TAPIR ####################
# @app.route('/get-predictions', methods=['POST'])
# def get_predictions():
#     datas = request.json
#     user = datas['user']
#     category = datas['Category']
#     name = datas['SubCategory']
#     step_id = datas["Object"]
#     video = datas['video']
#     video = video.split('/')[-1].split('.')[0]
#     projection = datas['Projection']
#     coordinates3d = datas['3d-coordinates']

#     # Get the current frame time
#     currentFrameTime = user_data[user]['current_video_time']
#     frame_id = int(currentFrameTime*user_data[user]['fps'])

#     # Get the current frame image
#     current_working_image = user_data[user]['current_working_image']
#     fps = user_data[user]['fps']

#     url = 'http://'+TAPIR_host+':' + str(TAPIR_port) + '/get-predictions'

#     data = {
#         'user' : user,
#         'Category': category,
#         'SubCategory': name,
#         'Object': step_id,
#         'video': video,
#         'Projection': projection,
#         'image': current_working_image,
#         'currentFrameTime': currentFrameTime,
#         'fps': fps,
#         '3d-coordinates': coordinates3d
#     }

#     # Send a POST request to TAPIR
#     response = requests.post(url, json=data)
#     return response.json()



#################### Segmentation ####################
def segmentation_sam(segmentation_data_path, debug=True, low_quality_mask = None):
    with lock:   
        print('Start Segmentation')
        with open(segmentation_data_path, 'r') as f:
            data = json.load(f)


        video_path = data['video_path']
        output_path = data['output_path']
        currentFrameTime = float(data['currentFrameTime'])
        img_data = data['image'].split(",")[1]  # Remove "data:image/png;base64," prefix
        img_bytes = base64.b64decode(img_data)
        image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_image = image
        h, w = image.shape[:2]
        input_point = np.array(eval(data['positive_points']) + eval(data['negative_points']))
        input_label = np.array([1] * len(eval(data['positive_points'])) + [0] * len(eval(data['negative_points'])))

        if len(eval(data['positive_points'])) == 0 and len(eval(data['negative_points'])) == 0:
            print('No input points')
            binary_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:

            

    
            # image_bgr = cv2.imread(video_path)
            # image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            # original_image = image

        

            sam.to(device=DEVICE)
            # image_embedding = np.load(image_embedding_path)

            # Create a new predictor
            # new_predictor = ExtendedSamPredictor(sam)
            new_predictor = SamPredictor(sam)
            new_predictor.set_image(image)

            # Get the original size
            original_size = original_image.shape[:2]  # Height and width of the original image

            # Apply the transform to get the input size
            input_image = new_predictor.transform.apply_image(original_image)
            input_size = input_image.shape[:2]  # Input size after transformation

            # Load the saved embedding and sizes
            # new_predictor.load_image_embedding(image_embedding, original_size, input_size)


            # The points coordinates are in the range [0, 1], and should be rescaled to the image size
            
            input_point[:, 0] *= w
            input_point[:, 1] *= h


            masks, scores, logits = new_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
                return_logits = True
            )
            
            mask_threshold = 0.5
        
            logits = logits[np.newaxis, :, :]
            resized_mask_torch = sam.postprocess_masks(torch.from_numpy(logits), input_size, original_size)
            binary_mask = (resized_mask_torch[0][0].cpu().numpy() > mask_threshold).astype(np.uint8)

        # # Remove overlapping with previous masks
        # if data['previous_mask_exist']:
        #     binary_mask = remove_mask_overlap(previous_masks, binary_mask)
        # Extract brush stroke data
        positive_brush_strokes = data['positive_brush_strokes']
        negative_brush_strokes = data['negative_brush_strokes']

        if low_quality_mask is not None:
            low_quality_mask = cv2.resize(low_quality_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            print('low_quality_mask', low_quality_mask)
            low_quality_mask = (low_quality_mask > 0.5).astype(np.uint8)
            binary_mask = np.maximum(binary_mask, low_quality_mask)
            

        # Apply brush strokes to the mask
        for stroke in positive_brush_strokes:
            for i in range(len(stroke) - 1):
                start_point = (int(stroke[i]['x'] * w), int(stroke[i]['y'] * h))
                end_point = (int(stroke[i+1]['x'] * w), int(stroke[i+1]['y'] * h))
                brush_size = int(stroke[i]['brushSize']*1.2)  # Convert to integer
                cv2.line(binary_mask, start_point, end_point, 1, brush_size)

        for stroke in negative_brush_strokes:
            for i in range(len(stroke) - 1):
                start_point = (int(stroke[i]['x'] * w), int(stroke[i]['y'] * h))
                end_point = (int(stroke[i+1]['x'] * w), int(stroke[i+1]['y'] * h))
                brush_size = int(stroke[i]['brushSize']*1.2)  # Convert to integer
                cv2.line(binary_mask, start_point, end_point, 0, brush_size)
        

        
        if debug:
            ########## Image with original frame image ##########
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            # show_mask(masks[0], plt.gca())
            show_mask(binary_mask, plt.gca())
            # show_points(input_point, input_label, plt.gca())
            plt.gca().set_position([0, 0, 1, 1])
            plt.axis('off') # Remove the axis   

            # Save the figure
            # Make directory if it doesn't exist
            output_dir = os.path.join(os.getcwd(), '/'.join(output_path.split("/")[:-1]))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            path = os.path.join(os.getcwd(), '/'.join(output_path.split('/')[:-1]), output_path.split('/')[-1].split('.')[0]+'_with_frame.jpg')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            #print("Saved image: " +  path)
            plt.close()
            ########## Image with points ##########
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            if len(input_point) != 0:
                show_points(input_point, input_label, plt.gca())
            show_mask(binary_mask, plt.gca())
            plt.gca().set_position([0, 0, 1, 1])
            plt.axis('off') # Remove the axis

            # # Save the figure
            # plt.savefig(os.path.join(os.getcwd(), output_path.split('.')[0]+'_with_points.jpg'), bbox_inches='tight', pad_inches=0)
            # #print("Saved image: " + output_path.split('.')[0]+'_with_points.jpg')
            # plt.close()
            # Save the figure
            # If dir doesn't exist, create it
            output_dir = os.path.join(os.getcwd(), '/'.join(output_path.split("/")[:-1]))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                #print("Created directory: " + output_dir)
            plt.savefig(os.path.join(os.getcwd(), output_path), bbox_inches='tight', pad_inches=0)
            #print("Saved image: " + output_path)
            plt.close()

            ########## Mask only ##########

            plt.figure(figsize=(10,10))
            plt.imshow(binary_mask)
            plt.gca().set_position([0, 0, 1, 1])
            plt.axis('off') # Remove the axis

            # Save the figure
            # Make directory if it doesn't exist
            path = os.path.join(os.getcwd(), '/'.join(output_path.split('/')[:-1]), output_path.split('/')[-1].split('.')[0]+'_mask_only.jpg')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            #print("Saved image: " +  path)
            plt.close()
        else: 
            ########## Image with points ##########
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            if len(input_point) != 0:
                show_points(input_point, input_label, plt.gca())
            
            show_mask(binary_mask, plt.gca())
            plt.gca().set_position([0, 0, 1, 1])
            plt.axis('off') # Remove the axis

            # Save the figure
            # If dir doesn't exist, create it
            output_dir = os.path.join(os.getcwd(), '/'.join(output_path.split("/")[:-1]))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                #print("Created directory: " + output_dir)
            plt.savefig(os.path.join(os.getcwd(), output_path), bbox_inches='tight', pad_inches=0)
            #print("Saved image: " + output_path)
            plt.close()

        # Save the mask to a json file using RLE encoding to the input json
        rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

        # Convert counts to ASCII string
        rle['counts'] = rle['counts'].decode('ascii')
        data['new_mask']={"size": rle['size'], "counts": rle['counts']}

        with open(segmentation_data_path, 'w') as f:
            json.dump(data, f, indent=4)
        return 'success'
   

@app.route('/create-segmentation_data', methods=['POST'])
def create_segmentation_data():
    datas = request.json
    user = datas['user']
    user_data[user]['segmentation_done'] = False
    fps = user_data[user]['fps']
   
    # JSON.stringify({"positive-keypoints": positiveKeypoints, "negative-keypoints": negativeKeypoints}) // sending the coordinates

    # convert coordinates from dict to list
    positiveKeypoints = []
    negativeKeypoints = []
    for data in datas["positive-keypoints"]:
        point = [data['x'], data['y']]
        positiveKeypoints.append(point)
    for data in datas["negative-keypoints"]:
        point = [data['x'], data['y']]
        negativeKeypoints.append(point)
    
    if positiveKeypoints == [] and negativeKeypoints == []:
        return jsonify({'error': 'No points provided'}), 400
    
    category = datas["Category"]
    name = datas["SubCategory"]

    # Get currentFrameTime and image from user_data
    currentFrameTime = user_data[user]['current_video_time']
    image = user_data[user]['current_working_image']
    current_video = user_data[user]['selected_video']


    # Change the list to string "[[123,234], [234,345]]" , str() not working, use json.dumps()
    positiveKeypoints = str(positiveKeypoints)
    negativeKeypoints = str(negativeKeypoints)

    #print( "Creating segmentation data for video: ", current_video)
    
    # Create a json for segmentation
    output_file = {}
    prefix = 'http://'+frontend_host+':' + str(port) + '/video/'
    output_file['video_path'] = os.path.join(os.getcwd(), current_video.split(prefix)[-1])
    output_file['positive_points'] = positiveKeypoints
    output_file['negative_points'] = negativeKeypoints
    output_file['currentFrameTime'] = currentFrameTime
    output_file['image'] = image
    currentModelFilePaths = datas["currentModelFilePaths"]
    #e.g. parts = "0,1,2"
    parts = [currentModelFilePath.split('/')[-1].split('.')[0] for currentModelFilePath in currentModelFilePaths]
    if len(parts) == 0:
        #print('currentModelFilePaths', currentModelFilePaths)
        user_data[user]['segmentation_done'] = True
        return jsonify({"error": "No part selected. Please select at least one part for segmentation."}), 500
    # change 00 -> 0, 01 -> 1, 02 -> 2
    parts = [str(int(part)) for part in parts]

    # Check if previous mask exists
    count = 0
    data = {}
    data['previous_frame_data'] = [{}]
    data['same_parts_exist'] = False
    user = datas['user']
    data_json_path = users[user]['data_json_path']
    json_data = get_data(data_json_path)
    for prev_data in json_data:
        if prev_data["category"] == category and prev_data["name"] == name:
            step_id = current_video.split('/')[-3].split('_')[-1]
            video_id = current_video.split('/')[-1].split('_clip')[0]
            # frame_id = current_video.split('/')[-1].split('_')[-1].split('.')[0]
            frame_id = int(currentFrameTime*fps)
            for step in prev_data['steps']:
                if step['step_id'] == int(step_id):
                    for video in step['video']:
                        if video['video_id'].split('watch?v=')[-1] == video_id:
                            for frame in video['frames']:
                                if int(frame['frame_id']) == int(frame_id):
                                    data['previous_frame_data'] = frame
                                
                                    # count the number of masks that not == {}
                                    for mask in frame['mask']:
                                        if mask != {}:
                                            count += 1
                                    for i in range(len(frame['parts'])):
                                        if is_same_parts(','.join(parts), frame['parts'][i]):
                                            # #print(frame)
                                            # remove this mask from segmentation data, since we don't want to exclude this mask
                                            if frame['mask'][i] != {}:
                                                count -= 1
                                                data['same_parts_exist'] = True

                                            data['previous_frame_data']['mask'][i] = {}
                                            
                                            # #print(count)
                                            #print("Previous mask parts overlap with current mask, no previous mask loaded for parts: ", parts)
                                            
                                    break
                            break
                    break
            break

    part_ls = []
    # #print(data['previous_frame_data'])
    if count > 0:
        # #print("previous_frame_data: ", data['previous_frame_data'])
        for part in data['previous_frame_data']['parts']:
            part_ls.append(part.split(','))
            for part in parts:
                if part in part_ls:
                    user_data[user]['segmentation_done'] = True
                    return jsonify({"success": False, "msg": "Previous mask parts overlap with current mask." ,"video Path": current_video})

    if count > 0:
        output_file['previous_mask_exist'] = True
        output_file['previous_masks'] = data['previous_frame_data']['mask']
        output_file['previous_parts'] = data['previous_frame_data']['parts']
    else:
        output_file['previous_mask_exist'] = False

    output_file['parts'] = ','.join(parts)
    output_file["category"] = category
    output_file["name"] = name
    output_file["step_id"] = step_id
    output_file["frame_id"] = frame_id
    output_file["video_id"] = current_video.split('/')[-1].split('_clip')[0]
    output_file["currentFrameTime"] = currentFrameTime
    # if data['same_parts_exist']:
    #     output_file['output_path']  =  os.path.join('./dynamic_dataset/masks' , category, name, datas["Object"], video_path.split('/')[-1].split('.')[0]+ str(int(currentFrameTime*10))+'_parts'+ ''.join(parts) +'_new.jpg')
    # else:
    #     output_file['output_path']  =  os.path.join('./dynamic_dataset/masks' , category, name, datas["Object"], video_path.split('/')[-1].split('.')[0]+str(int(currentFrameTime*10))+'_parts'+ ''.join(parts) +'.jpg')
    output_file['output_path']  =  os.path.join('./dynamic_dataset/masks' , category, name, datas["Object"], user+'_'+current_video.split('/')[-1].split('.')[0]+str(int(currentFrameTime*fps))+'_parts'+ ''.join(parts) +'.jpg')

    segmentation_data_dir = os.path.join('./dynamic_dataset/masks/segmentation_data/' , category, name, datas["Object"])
    segmentation_data_path = segmentation_data_dir + '/' +  user+'_'+current_video.split('/')[-1].split('.')[0]+str(int(currentFrameTime*fps))+'_parts'+ ''.join(parts) +'.json'
    user_data[user]['segmentation_data_path'] = segmentation_data_path
    if   not os.path.exists(segmentation_data_dir):
        os.makedirs(segmentation_data_dir)
    with open(segmentation_data_path, 'w') as outfile:
        json.dump(output_file, outfile , indent=4)

@app.route('/segmentation', methods=['POST'])
def segmentation():
    datas = request.json
    user = datas['user']
    user_data[user]['segmentation_done'] = False
    fps = user_data[user]['fps']
    with_previous_mask = datas['with_prev_mask']
   
    # JSON.stringify({"positive-keypoints": positiveKeypoints, "negative-keypoints": negativeKeypoints}) // sending the coordinates

    # convert coordinates from dict to list
    positiveKeypoints = []
    negativeKeypoints = []
    for data in datas["positive-keypoints"]:
        point = [data['x'], data['y']]
        positiveKeypoints.append(point)
    for data in datas["negative-keypoints"]:
        point = [data['x'], data['y']]
        negativeKeypoints.append(point)
    
    # Extract brush stroke data
    positive_brush_strokes = datas["PositiveBrushStrokes"]
    negative_brush_strokes = datas["NegativeBrushStrokes"]
    print(f'positive brush strokes: {positive_brush_strokes}')
    print(f'negative brush strokes: {negative_brush_strokes}')
    # if positiveKeypoints == [] and negativeKeypoints == [] and positive_brush_strokes == [] and negative_brush_strokes == []:
    #     return jsonify({'error': 'No points provided'}), 400
    
    category = datas["Category"]
    name = datas["SubCategory"]

    # Get currentFrameTime and image from user_data
    currentFrameTime = user_data[user]['current_video_time']
    image = user_data[user]['current_working_image']
    current_video = user_data[user]['selected_video']


    # Change the list to string "[[123,234], [234,345]]" , str() not working, use json.dumps()
    positiveKeypoints = str(positiveKeypoints)
    negativeKeypoints = str(negativeKeypoints)

    #print( "Creating segmentation data for video: ", current_video)
    
    # Create a json for segmentation
    output_file = {}
    prefix = 'http://'+frontend_host+':' + str(port) + '/video/'
    output_file['video_path'] = os.path.join(os.getcwd(), current_video.split(prefix)[-1])
    output_file['positive_points'] = positiveKeypoints
    output_file['negative_points'] = negativeKeypoints
    output_file['currentFrameTime'] = currentFrameTime
    output_file['image'] = image
    currentModelFilePaths = datas["currentModelFilePaths"]
    #e.g. parts = "0,1,2"
    parts = [currentModelFilePath.split('/')[-1].split('.')[0] for currentModelFilePath in currentModelFilePaths]
    if len(parts) == 0:
        #print('currentModelFilePaths', currentModelFilePaths)
        user_data[user]['segmentation_done'] = True
        return jsonify({"error": "No part selected. Please select at least one part for segmentation."}), 500
    # change 00 -> 0, 01 -> 1, 02 -> 2
    parts = [str(int(part)) for part in parts]

    # Check if previous mask exists
    count = 0
    data = {}
    data['previous_frame_data'] = [{}]
    data['same_parts_exist'] = False
    user = datas['user']
    data_json_path = users[user]['data_json_path']
    json_data = get_data(data_json_path)
    step_id = current_video.split('/')[-3].split('_')[-1]
    video_id = current_video.split('/')[-1].split('_clip')[0]
    # frame_id = current_video.split('/')[-1].split('_')[-1].split('.')[0]
    frame_id = int(currentFrameTime*fps)


    part_ls = []
    # #print(data['previous_frame_data'])
    if count > 0:
        # #print("previous_frame_data: ", data['previous_frame_data'])
        for part in data['previous_frame_data']['parts']:
            part_ls.append(part.split(','))
            for part in parts:
                if part in part_ls:
                    user_data[user]['segmentation_done'] = True
                    return jsonify({"success": False, "msg": "Previous mask parts overlap with current mask." ,"video Path": current_video})

    if count > 0:
        output_file['previous_mask_exist'] = True
        output_file['previous_masks'] = data['previous_frame_data']['mask']
        output_file['previous_parts'] = data['previous_frame_data']['parts']
    else:
        output_file['previous_mask_exist'] = False

    

    output_file['positive_brush_strokes'] = positive_brush_strokes
    output_file['negative_brush_strokes'] = negative_brush_strokes

    output_file['parts'] = ','.join(parts)
    output_file["category"] = category
    output_file["name"] = name
    output_file["step_id"] = step_id
    output_file["frame_id"] = frame_id
    output_file["video_id"] = current_video.split('/')[-1].split('_clip')[0]
    output_file["currentFrameTime"] = currentFrameTime
    # if data['same_parts_exist']:
    #     output_file['output_path']  =  os.path.join('./dynamic_dataset/masks' , category, name, datas["Object"], video_path.split('/')[-1].split('.')[0]+ str(int(currentFrameTime*10))+'_parts'+ ''.join(parts) +'_new.jpg')
    # else:
    #     output_file['output_path']  =  os.path.join('./dynamic_dataset/masks' , category, name, datas["Object"], video_path.split('/')[-1].split('.')[0]+str(int(currentFrameTime*10))+'_parts'+ ''.join(parts) +'.jpg')
    output_file['output_path']  =  os.path.join('./dynamic_dataset/masks' , category, name, datas["Object"], user+'_'+current_video.split('/')[-1].split('.')[0]+str(int(currentFrameTime*fps))+'_parts'+ ''.join(parts) +'_temp.jpg')
    
    segmentation_data_dir = os.path.join('./dynamic_dataset/masks/segmentation_data/' , category, name, datas["Object"])
    segmentation_data_path = segmentation_data_dir + '/' +  user+'_'+current_video.split('/')[-1].split('.')[0]+str(int(currentFrameTime*fps))+'_parts'+ ''.join(parts) +'.json'
    user_data[user]['segmentation_data_path'] = segmentation_data_path
    if   not os.path.exists(segmentation_data_dir):
        os.makedirs(segmentation_data_dir)
    with open(segmentation_data_path, 'w') as outfile:
        json.dump(output_file, outfile , indent=4)

    # Run segmentation
    #print("Start segmentation...")
    if with_previous_mask:
        user = request.json['user']
        Category = request.json['Category']
        SubCategory = request.json['SubCategory']
        Step = request.json['step']
        Video = request.json['video']
        Part = request.json['part']
        Frame = request.json['frame']
        mask, _, _ = get_prev_mask_helper(user, Category, SubCategory, Step, Video, Part, Frame)
        low_quality_mask = mask
    else:
        low_quality_mask = None
    success = segmentation_sam(segmentation_data_path, debug=True, low_quality_mask=low_quality_mask)
    #print("Segmentation finished.")

    # Get absolute path of output image
    output_path = 'http://'+frontend_host+':' + str(port) + '/image/' + os.path.relpath(output_file["output_path"], os.getcwd())
    #print("output_path: ", output_path)

    user_data[user]['segmentation_done'] = True
    user_data[user]['output_path'] = output_path
    return jsonify({"imagePath": output_path, "messages": ['Segmentation finished!']})

#############Part not able to annotate################
@app.route('/part-not-able-to-present', methods=['POST'])
def part_not_able_to_present():
    #print('################## working on part_not_able_to_present #############')
    datas = request.json

    nextFrameTime = datas["nextFrameTime"]
    #print("nextFrameTime: ", nextFrameTime)
    user = datas['user']
    assert nextFrameTime != 0 # Should not be 'Start Annotation' call

    if(nextFrameTime != 0):
        #  wait until all the previous tasks are finished
        while not user_data[user]['segmentation_done']:
            #print(f"Waiting for segmentation to finish for user {user}...")
            time.sleep(0.1)
        while not user_data[user]['pose_estimation_done']:
            #print(f"Waiting for pose estimation to finish for user {user}...")
            time.sleep(0.1)
        while not user_data[user]['save_mask_done']:
            #print(f"Waiting for save mask to finish for user {user}...")
            time.sleep(0.1)
        while not user_data[user]['save_project_done']:
            #print(f"Waiting for save pose estimation to finish for user {user}...")
            time.sleep(0.1)


    data_json_path = users[user]['data_json_path']
    json_data = get_data(data_json_path)
    category = datas["Category"]
    name = datas["SubCategory"]
    step_id = datas["Step"]
    
    video_id = datas["videoPath"].split('/')[-1].split('_clip')[0]
    videoPath = '/video/'.join(datas["videoPath"].split('/frames/')).split('/step')[0]+'/'+video_id+'/'+video_id+'.mp4' # Get video from video path rather than video clip path
    currentModelFilePaths = datas["currentModelFilePaths"] #e.g. parts = "0,1,2" with base path
    currentFrameTime = user_data[user]['current_video_time']
    fps = user_data[user]['fps']
    Projection = datas['Projection']
    NotAbleToAnnotate = datas['NotAbleToAnnotate']
    ReportMessage = datas['ReportMessage']
    


    objs = []
    for currentModelFilePath in currentModelFilePaths:
        obj = currentModelFilePath.split('/')[-1].split('.')[0]
        # Convert back from 00, 01 to 0, 1
        obj = str(int(obj))
        objs.append(obj)
    
    obj_path = ','.join(objs)


    # Get the current frame time from user data
    currentFrameTime = user_data[user]['current_video_time']
    current_working_duration = user_data[user]['current_working_duration']
    annotation_durations = user_data[user]['annotation_durations']


    # Fill the mask/pose estimation data for the current frame time
    for furniture in json_data:
        if furniture['category'] == category and furniture['name'] == name:
            for step in furniture['steps']:
                if step['step_id'] == int(step_id.split('_')[-1]):
                    for video in step['video']:
                        if video['video_id'].split('watch?v=')[-1] == video_id:
                            for frame in video['frames']:
                                if int(frame['frame_id']) == int(currentFrameTime*fps):
                                    for part in frame['parts']:
                                        if is_same_parts(part, obj_path):
                                            if ReportMessage:
                                                message = datas['Message']
                                                if Projection:
                                                    frame['extrinsics'][frame['parts'].index(part)] = [message]
                                                    frame['intrinsics'][frame['parts'].index(part)] = [message]
                                                else:
                                                    frame['mask'][frame['parts'].index(part)] = {'error':message}
                                                
                                            elif Projection:
                                                if NotAbleToAnnotate:
                                                    frame['extrinsics'][frame['parts'].index(part)] = ['NotAbleToAnnotate']
                                                    frame['intrinsics'][frame['parts'].index(part)] = ['NotAbleToAnnotate']
                                                else:
                                                    frame['extrinsics'][frame['parts'].index(part)] = ['PartNotShowUp']
                                                    frame['intrinsics'][frame['parts'].index(part)] = ['PartNotShowUp']
                                            else:
                                                if NotAbleToAnnotate:
                                                    frame['mask'][frame['parts'].index(part)] = {'error':'NotAbleToAnnotate'}
                                                else:
                                                    frame['mask'][frame['parts'].index(part)] = {'error':'PartNotShowUp'}
                                            
                                    break
                            break
                    break
            break
    
    # Save the data to json file
    with open(data_json_path, 'w') as outfile:
        json.dump(json_data, outfile , indent=4)


    increment = 1
    # Check if currentFrameTime + 1 is still within the annotation_durations
    if currentFrameTime + increment >= float(annotation_durations[current_working_duration][0]) and currentFrameTime + increment <= float(annotation_durations[current_working_duration][1]):
        # If yes, increase the currentFrameTime by 1
        currentFrameTime = currentFrameTime + increment
    else:

        if current_working_duration >= len(annotation_durations)-1:
            return jsonify({"error": "You have reached the end of the video."}), 500
        # If not, increase the current_working_duration by 1
        current_working_duration += 1
        user_data[user]['current_working_duration'] = current_working_duration
        # Change current frame time to the start time of the next annotation duration
        currentFrameTime = annotation_durations[current_working_duration][0]

    user_data[user]['current_video_time'] = currentFrameTime
    print(f'##################[4]####### Set current_video_time to {currentFrameTime} for user {user}')
    
    # get the image at currentFrameTime in the video    
    #print("currentFrameTime: ", currentFrameTime)
    video = cv2.VideoCapture(videoPath)
    video.set(cv2.CAP_PROP_POS_MSEC, (currentFrameTime)*1000) #Get frame from the origin video rather than video clips
    success, image = video.read()
    if not success:
        #print('##########Can not read video.##########')
        return jsonify({"error": "Cannot read the video."}), 500

    # convert image to base64
    retval, buffer = cv2.imencode('.jpg', image)
    image = base64.b64encode(buffer).decode("utf-8")

    #print("currentFrameTime: ", currentFrameTime)
    # #print("image: ", image)
    user_data[user]['current_working_image'] = 'data:image/png;base64,' + image
    
    if previous_data_exist(json_data, category, name, step_id, currentFrameTime,datas['Projection'], obj_path,video_id):
        #e.g. parts = "0,1,2"
        parts = [currentModelFilePath.split('/')[-1].split('.')[0] for currentModelFilePath in currentModelFilePaths]
        if len(parts) == 0:
            #print('currentModelFilePaths', currentModelFilePaths)
            user_data[user]['segmentation_done'] = True
            return jsonify({"error": "No part selected. Please select at least one part for segmentation."}), 500
        # change 00 -> 0, 01 -> 1, 02 -> 2
        parts = [str(int(part)) for part in parts]
        previous_data_path = get_output_path(category, name, step_id, currentFrameTime, parts, user, video_id, fps, Projection=datas['Projection'])
        messages = ['Previous annotation exists! Be Careful for overwriting!']
    else:
        previous_data_path = None
        messages = ['Previous annotation NOT exists.']
    
    

    print(f'currentFrameTime: {currentFrameTime}, previous_data_path: {previous_data_path}')
    return jsonify({"base64Image": image, "currentFrameTime": currentFrameTime-user_data[user]['video_start_time'], "imagePath": previous_data_path, "messages": messages})


#############Back to prev anno for current user#############
@app.route('/back-to-prev-anno-frame', methods=['POST'])
def back_to_prev_annotation():
    #print('################## working on back_to_prev_annotation ##################')
    datas = request.json

    nextFrameTime = datas["nextFrameTime"]
    #print("nextFrameTime: ", nextFrameTime)
    user = datas['user']

    if(nextFrameTime != 0):
        #  wait until all the previous tasks are finished
        while not user_data[user]['segmentation_done']:
            #print(f"Waiting for segmentation to finish for user {user}...")
            time.sleep(0.1)
        while not user_data[user]['pose_estimation_done']:
            #print(f"Waiting for pose estimation to finish for user {user}...")
            time.sleep(0.1)
        while not user_data[user]['save_mask_done']:
            #print(f"Waiting for save mask to finish for user {user}...")
            time.sleep(0.1)
        while not user_data[user]['save_project_done']:
            #print(f"Waiting for save pose estimation to finish for user {user}...")
            time.sleep(0.1)


    data_json_path = users[user]['data_json_path']
    json_data = get_data(data_json_path)
    category = datas["Category"]
    name = datas["SubCategory"]
    step_id = datas["Step"]
    
    video_id = datas["videoPath"].split('/')[-1].split('_clip')[0]
    videoPath = '/video/'.join(datas["videoPath"].split('/frames/')).split('/step')[0]+'/'+video_id+'/'+video_id+'.mp4' # Get video from video path rather than video clip path
    currentModelFilePaths = datas["currentModelFilePaths"] #e.g. parts = "0,1,2" with base path
    currentFrameTime = user_data[user]['current_video_time']
    
    fps = user_data[user]['fps']


    objs = []
    for currentModelFilePath in currentModelFilePaths:
        obj = currentModelFilePath.split('/')[-1].split('.')[0]
        # Convert back from 00, 01 to 0, 1
        obj = str(int(obj))
        objs.append(obj)
    
    obj_path = ','.join(objs)
    #print("obj_path: ", obj_path)

    #print('nextFrameTime: ', nextFrameTime)
    # Get next frame time
    if(nextFrameTime == 0): # first time 
        annotation_durations, video_start = get_annotation_durations(json_data, category, name, step_id, video_id, fps,obj_path)
        #print("annotation_durations: ", annotation_durations)
        # Sort the annotation_durations by the start time
        annotation_durations = annotation_durations[annotation_durations[:,0].argsort()]
        #print("sorted annotation_durations: ", annotation_durations)
        user_data[user]['annotation_durations'] = annotation_durations
        current_working_duration = 0
        user_data[user]['current_working_duration'] = current_working_duration
        # #print(annotation_durations)
        user_data[user]['video_start_time'] = video_start
        # Get the current frame time from user data
    # currentFrameTime = user_data[user]['current_video_time']

    current_working_duration = user_data[user]['current_working_duration']
    annotation_durations = user_data[user]['annotation_durations']
    # Get the first frame time
    currentFrameTime = annotation_durations[0][0] # Go from the first frame
    lastFrameTime = annotation_durations[0][0]

    increment = 1
    print('annotation_durations', annotation_durations)
    print(f'finding for {currentFrameTime}')
    print(previous_data_exist(json_data, category, name, step_id, currentFrameTime,datas['Projection'], obj_path,video_id))
    while previous_data_exist(json_data, category, name, step_id, currentFrameTime,datas['Projection'], obj_path,video_id):
        lastFrameTime = currentFrameTime
        if currentFrameTime + increment >= float(annotation_durations[current_working_duration][0]) and currentFrameTime + increment <= float(annotation_durations[current_working_duration][1]):
            # If yes, increase the currentFrameTime by 1
            currentFrameTime = currentFrameTime + increment
            print('increase the currentFrameTime by 1')
        else:
            #print("current_working_duration: ", current_working_duration)
            #print("len(annotation_durations): ", len(annotation_durations))
            if current_working_duration >= len(annotation_durations)-1:
                return jsonify({"error": "You have reached the end of the video."}), 500
            print("currentFrameTime: ", currentFrameTime)
            #print("currentFrameTime + increment:", currentFrameTime + increment)
            # If not, increase the current_working_duration by 1
            current_working_duration += 1
            user_data[user]['current_working_duration'] = current_working_duration
            # Change current frame time to the start time of the next annotation duration
            currentFrameTime = annotation_durations[current_working_duration][0]
    # If no, return the lastFrameTime (back to the last frame you worked on )
    currentFrameTime = lastFrameTime
    print('finally find ', currentFrameTime)

    user_data[user]['current_video_time'] = currentFrameTime
    print(f'##################[1]####### Set current_video_time to {currentFrameTime} for user {user}')
    
    # get the image at currentFrameTime in the video    
    #print("currentFrameTime: ", currentFrameTime)
    video = cv2.VideoCapture(videoPath)
    video.set(cv2.CAP_PROP_POS_MSEC, (currentFrameTime)*1000) #Get frame from the origin video rather than video clips
    success, image = video.read()
    if not success:
        #print('##########Can not read video.##########')
        return jsonify({"error": "Cannot read the video."}), 500

    # convert image to base64
    retval, buffer = cv2.imencode('.jpg', image)
    image = base64.b64encode(buffer).decode("utf-8")

    #print("currentFrameTime: ", currentFrameTime)
    # #print("image: ", image)
    user_data[user]['current_working_image'] = 'data:image/png;base64,' + image
    
    if previous_data_exist(json_data, category, name, step_id, currentFrameTime,datas['Projection'], obj_path,video_id):
        #e.g. parts = "0,1,2"
        parts = [currentModelFilePath.split('/')[-1].split('.')[0] for currentModelFilePath in currentModelFilePaths]
        if len(parts) == 0:
            #print('currentModelFilePaths', currentModelFilePaths)
            user_data[user]['segmentation_done'] = True
            return jsonify({"error": "No part selected. Please select at least one part for segmentation."}), 500
        # change 00 -> 0, 01 -> 1, 02 -> 2
        parts = [str(int(part)) for part in parts]
        previous_data_path = get_output_path(category, name, step_id, currentFrameTime, parts, user, video_id, fps, Projection=datas['Projection'])
        messages = ['Previous annotation exists! Be Careful for overwriting!']
    else:
        previous_data_path = None
        messages = ['Previous annotation NOT exists.']
    

    print(f'currentFrameTime: {currentFrameTime}, previous_data_path: {previous_data_path}')
    return jsonify({"base64Image": image, "currentFrameTime": currentFrameTime-user_data[user]['video_start_time'], "imagePath": previous_data_path, "messages": messages})


def get_annotation_durations(json_data, category, name, step_id, video_id, fps,obj_path):
    annotation_durations = []
    video_start = -1
    for furniture in json_data:
        if furniture["category"] == category and furniture["name"] == name:
            for step in furniture["steps"]:
                if int(step["step_id"]) == int(step_id.split('_')[-1]):
                    for video in step["video"]:
                        if video["video_id"].split('watch?v=')[-1] == video_id:
                            video_start = video["step_start"]
                            for substep in video["substeps"]:
                                for part in substep['parts']:
                                    if is_same_parts(part, obj_path):
                                        # #print("part: ", part)
                                        annotation_durations.append([substep["substep_start"], substep["substep_end"]])
                                        break
                            break
                    break
            break
    annotation_durations = np.array(annotation_durations)
    return annotation_durations, video_start
####################  Next Frame #################### 
@app.route('/next-frame', methods=['POST'])
def next_frame():
    #print('################## working on the next frame function')
    datas = request.json

    nextFrameTime = datas["nextFrameTime"]
    user = datas['user']


    if(nextFrameTime != 0):
        #  wait until all the previous tasks are finished
        while not user_data[user]['segmentation_done']:
            #print(f"Waiting for segmentation to finish for user {user}...")
            time.sleep(0.1)
        while not user_data[user]['pose_estimation_done']:
            #print(f"Waiting for pose estimation to finish for user {user}...")
            time.sleep(0.1)
        while not user_data[user]['save_mask_done']:
            #print(f"Waiting for save mask to finish for user {user}...")
            time.sleep(0.1)
        while not user_data[user]['save_project_done']:
            #print(f"Waiting for save pose estimation to finish for user {user}...")
            time.sleep(0.1)


    data_json_path = users[user]['data_json_path']
    json_data = get_data(data_json_path)
    category = datas["Category"]
    name = datas["SubCategory"]
    step_id = datas["Step"]
    
    video_id = datas["videoPath"].split('/')[-1].split('_clip')[0]
    videoPath = '/video/'.join(datas["videoPath"].split('/frames/')).split('/step')[0]+'/'+video_id+'/'+video_id+'.mp4' # Get video from video path rather than video clip path
    currentModelFilePaths = datas["currentModelFilePaths"] #e.g. parts = "0,1,2" with base path
    currentFrameTime = user_data[user]['current_video_time']
    print(f'current_video_time is {currentFrameTime} for user {user}')
    fps = user_data[user]['fps']


    objs = []
    for currentModelFilePath in currentModelFilePaths:
        obj = currentModelFilePath.split('/')[-1].split('.')[0]
        # Convert back from 00, 01 to 0, 1
        obj = str(int(obj))
        objs.append(obj)
    
    obj_path = ','.join(objs)

    
    # if(nextFrameTime == 0): # first time (Start Annotation call) 
    annotation_durations, video_start = get_annotation_durations(json_data, category, name, step_id, video_id, fps, obj_path)
    print(f'annotation_durations is {annotation_durations} for user {user}, and video_start is {video_start}')
    # annotation_durations = annotation_durations[annotation_durations[:,0].argsort()]
    user_data[user]['annotation_durations'] = annotation_durations
    current_working_duration = 0
    for i in range(len(annotation_durations)):
        if currentFrameTime >= float(annotation_durations[i][0]) and currentFrameTime <= float(annotation_durations[i][1]):
            current_working_duration = i
            break
    # Get the first frame time
    # currentFrameTime = annotation_durations[0][0] 
    user_data[user]['current_working_duration'] = current_working_duration
    user_data[user]['video_start_time'] = video_start
    

    user_data[user]['current_video_time'] = currentFrameTime
    print(f'##################[2]####### Set current_video_time to {currentFrameTime} for user {user}')
    
    # get the image at currentFrameTime in the video    
    #print("currentFrameTime: ", currentFrameTime)
    video = cv2.VideoCapture(videoPath)
    video.set(cv2.CAP_PROP_POS_MSEC, (currentFrameTime)*1000) #Get frame from the origin video rather than video clips
    success, image = video.read()
    if not success:
        #print('##########Can not read video.##########')
        return jsonify({"error": "Cannot read the video."}), 500

    # convert image to base64
    retval, buffer = cv2.imencode('.jpg', image)
    image = base64.b64encode(buffer).decode("utf-8")

    user_data[user]['current_working_image'] = 'data:image/png;base64,' + image
    
    if previous_data_exist(json_data, category, name, step_id, currentFrameTime,datas['Projection'], obj_path,video_id):
        #e.g. parts = "0,1,2"
        parts = [currentModelFilePath.split('/')[-1].split('.')[0] for currentModelFilePath in currentModelFilePaths]
        if len(parts) == 0:
            #print('currentModelFilePaths', currentModelFilePaths)
            user_data[user]['segmentation_done'] = True
            return jsonify({"error": "No part selected. Please select at least one part for segmentation."}), 500
        # change 00 -> 0, 01 -> 1, 02 -> 2
        parts = [str(int(part)) for part in parts]
        previous_data_path = get_output_path(category, name, step_id, currentFrameTime, parts, user, video_id, fps, Projection=datas['Projection'])
        messages = ['Previous annotation exists! Be Careful for overwriting!']
    else:
        previous_data_path = None
        messages = ['Previous annotation NOT exists.']
    

    # #print(f'Return image: {image}, currentFrameTime: {currentFrameTime}, previous_data_path: {previous_data_path}')
    return jsonify({"base64Image": image, "currentFrameTime": currentFrameTime-user_data[user]['video_start_time'], "imagePath": previous_data_path, "messages": messages})

def get_output_path(category, name, step_id,  currentFrameTime, parts, user, current_video, fps, Projection):
    if not Projection:
        output_path = os.path.join('./dynamic_dataset/masks' , category, name, step_id, user+'_'+current_video.split('/')[-1].split('.')[0]+'_clip'+str(int(currentFrameTime*fps))+'_parts'+ ''.join(parts) +'.jpg')
        #  Check if th eimage exists 
        if not os.path.exists(os.getcwd() +'/'+ output_path):
            #print(f"not found {output_path}")
            # The annotation can be done by any user, so go through all the user names to find which one has the annotation:
            for user_name in users.keys():
                #print(f"finding {os.path.join('./dynamic_dataset/masks' , category, name, step_id, user_name+'_'+current_video.split('/')[-1].split('.')[0]+'_clip'+str(int(currentFrameTime*fps))+'_parts'+ ''.join(parts) +'.jpg')}")
                if os.path.exists(os.path.join('./dynamic_dataset/masks' , category, name, step_id, user_name+'_'+current_video.split('/')[-1].split('.')[0]+'_clip'+str(int(currentFrameTime*fps))+'_parts'+ ''.join(parts) +'.jpg')):
                    #print(f"found {user_name}")
                    output_path = os.path.join('./dynamic_dataset/masks' , category, name, step_id, user_name+'_'+current_video.split('/')[-1].split('.')[0]+'_clip'+str(int(currentFrameTime*fps))+'_parts'+ ''.join(parts) +'.jpg')
                    break
        output_path =  'http://'+frontend_host+':' + str(port) +'/image/' + output_path
        
    else:
        output_path = os.path.join('dynamic_dataset/pose-estimation' , category, name, step_id, user + '_' +  current_video.split('/')[-1].split('.')[0]+'_clip'+str(int(currentFrameTime*fps))+'_parts'+ ''.join(parts) + '.jpg')
        if not os.path.exists(os.getcwd() +'/'+output_path):
            #print(f"not found {output_path}")

            # Annotation is not done by current user
            for user_name in users.keys():
                #print(f"finding {os.path.join('./dynamic_dataset/pose-estimation' , category, name, step_id, user_name+'_'+current_video.split('/')[-1].split('.')[0]+'_clip'+str(int(currentFrameTime*fps))+'_parts'+ ''.join(parts) +'.jpg')}")

                if os.path.exists(os.path.join('./dynamic_dataset/pose-estimation' , category, name, step_id, user_name+'_'+current_video.split('/')[-1].split('.')[0]+'_clip'+str(int(currentFrameTime*fps))+'_parts'+ ''.join(parts) +'.jpg')):
                    #print(f"found {user_name}")

                    output_path = os.path.join('./dynamic_dataset/pose-estimation' , category, name, step_id, user_name+'_'+current_video.split('/')[-1].split('.')[0]+'_clip'+str(int(currentFrameTime*fps))+'_parts'+ ''.join(parts) +'.jpg')
                    break
        output_path = 'http://'+frontend_host+':' + str(port) +'/image/' + output_path
    return output_path 

def previous_data_exist(json_data, category, name, step_id, currentFrameTime, Projection, obj_path,video_id):
    print(f"finding previous data for category{category}, name{name}, step_id{step_id}, currentFrameTime{currentFrameTime}, Projection{Projection}, obj_path{obj_path},video_id{video_id}")
    for furniture in json_data:
        if furniture["category"] == category and furniture["name"] == name:
            for step in furniture["steps"]:
                if int(step["step_id"]) == int(step_id.split('_')[-1]):
                    for video in step["video"]:
                        print('video["video_id"].split("watch?v=")[-1], video_id', video["video_id"], video_id)
                        if video["video_id"].split('watch?v=')[-1] == video_id:
                            for frame in video["frames"]:
                                if abs(float(frame["frame_time"]) - float(currentFrameTime)) <0.1: # if 
                                    print(f"find frame time {frame['frame_time']} for {currentFrameTime}")
                                    for i, part in enumerate(frame['parts']):
                                        print(part)
                                        print(is_same_parts(part, obj_path))
                                        if is_same_parts(part, obj_path):
                                            if Projection:
                                                if frame["extrinsics"][i] != [] and frame["intrinsics"][i] != []:
                                                    return True
                                            else:
                                                if frame["mask"][i] != {} and frame["mask"][i] != []:
                                                    return True
    return False
        
#################### Table of Contents ####################
# get-categories, get-subcategories, get-steps, get-object
@app.route('/get-categories', methods=['POST'])
def get_categories():
    data = request.json
    user = data['user']
    data_json_path = users[user]['data_json_path']
    # Categories can be extracted from the json data
    categories = []
    json_data = get_data(data_json_path)
    for data in json_data:
        if data['category'] not in categories:
            categories.append(data['category'])
    

    # #print("categories: ", categories)
    return jsonify({"categories": categories})
    

@app.route('/get-segmentation-image-path-initial', methods=['POST'])
def get_segmentationImagePath_initial():
    datas = request.json
    selected_category = datas["Category"]
    selected_subcategory = datas["SubCategory"]
    selected_step = datas["Object"]
    selected_frame = datas["Frame"].split('/')[-1]
    original_image_path = datas["OriginalImagePath"]

    frame_num = int(selected_frame.split('_')[-1].split('.')[0])

    # Check if the segmentation and pose estimation image exists
    # If exists, return the path
    # If not, return the original image path

    segmentation_image_dir = os.path.join('dynamic_dataset/masks' , selected_category, selected_subcategory, selected_step)
    projection_image_dir = os.path.join( 'dynamic_dataset/pose-estimation' , selected_category, selected_subcategory, selected_step)
    return_info = {}
    segmentationImagePaths = []
    if os.path.exists(segmentation_image_dir):
        if len(os.listdir(segmentation_image_dir)) > 0:
            #print(os.listdir(segmentation_image_dir))
            for file in os.listdir(segmentation_image_dir):
                if file.split('_')[2]== str(frame_num):
                    #print ("segmentation image exists")

                    segmentationImagePaths.append('http://'+frontend_host+':'+ str(port) + '/image/' +os.path.join(segmentation_image_dir, file))
            return_info["segmentationImagePaths"] = segmentationImagePaths
    if "segmentationImagePaths" not in return_info:
        return_info["segmentationImagePaths"] = [original_image_path]


    # if the dir is not empty, return the list of projection images
    projectImagePaths = []
    if os.path.exists(projection_image_dir):
        if len(os.listdir(projection_image_dir)) > 0:
            #print ("projection image exists")
           
            for file in os.listdir(projection_image_dir):
                if file.split('_')[-2]== str(frame_num):
                    #print ("segmentation image not exists")

                    projectImagePaths.append('http://'+frontend_host+':' + str(port) + '/image/' +os.path.join(projection_image_dir, file))
            return_info["projectImagePaths"] = projectImagePaths
    if "projectImagePaths" not in return_info:
        #print ("projection image not exists")
        return_info["projectImagePaths"] = [original_image_path]
    
    #print ("return_info: ", return_info)
    return jsonify(return_info)


@app.route('/get-subcategories', methods=['POST'])
def get_subcategories():
    # Subcategories can be extracted from the json data
    datas = request.json
    selected_category = datas["category"]
    subcategories = []
    user = datas['user']
    data_json_path = users[user]['data_json_path']
    json_data = get_data(data_json_path)
    for data in json_data:
        if data['category'] == selected_category:
            if data['name'] not in subcategories:
                subcategories.append(data['name'])
    # #print("subcategories: ", subcategories)
    # user_data[user]['selected_category'] = selected_category

    return jsonify({"subcategories": subcategories})


# This command will give the list of steps in the selected category and subcategory
@app.route('/get-steps', methods=['POST'])
def get_steps():
    datas = request.json
    selected_category = datas["category"]
    selected_subcategory = datas["subCategory"]
    user = datas['user']
    steps = []
    filePaths = []

    
    parts_base_path = os.path.join('./dynamic_dataset/parts', selected_category, selected_subcategory)
    parts_base_to_return = 'http://'+frontend_host+':' + str(port) + '/image/' + os.path.join('./dynamic_dataset/parts', selected_category, selected_subcategory)
    

    user = datas['user']
    data_json_path = users[user]['data_json_path']
    json_data = get_data(data_json_path)
    for data in json_data:
        if data['category'] == selected_category and data['name'] == selected_subcategory:
            for step in data['steps']:
                if step['step_id'] not in steps:
                    steps.append('step_'+ str(step['step_id']))
    
        # Find all avaliable files that ends with .obj in the parts_base_path using os
    obj_num_ls = []
    for file in os.listdir(parts_base_path):
        if file.endswith(".obj"):
            filePaths.append(os.path.join(parts_base_to_return, file))
            obj_num_ls.append(int(file.split('_')[-1].split('.')[0]))
    # Sort the file paths based on the obj number
    filePaths = [x for _,x in sorted(zip(obj_num_ls, filePaths))]
    

    return jsonify({"steps": steps, "filePaths": filePaths})

@app.route('/get-video-manual-parts', methods=['POST'])
def get_video_manual_parts():
    datas = request.json
    selected_category = datas["category"]
    selected_subcategory = datas["subCategory"]
    selected_step = datas["step"]
    user = datas['user']
    data_json_path = users[user]['data_json_path']
    json_data = get_data(data_json_path)

    originalVideoPaths = []
    image_base_path =  os.path.join('./dynamic_dataset/frames', selected_category, selected_subcategory, selected_step)

    video_id_ls = []
    for furniture in json_data:
        if furniture["category"] == selected_category and furniture["name"] == selected_subcategory:
            for step in furniture["steps"]:
                if step["step_id"] == int(selected_step.split('_')[-1]):
                    for video in step["video"]:
                        video_id_ls.append(video["video_id"].split('watch?v=')[-1])
        
    for video_id in video_id_ls:
        if video_id not in ''.join(originalVideoPaths):
            originalVideoPaths.append('http://'+frontend_host+':' + str(port) + '/video/' + os.path.join(image_base_path, video_id, video_id + '_clip.mp4'))

    # Also return the manual image path
    manual_image_dir = os.path.join('dynamic_dataset/manual_img' , selected_category, selected_subcategory, selected_step)
    if os.path.exists(manual_image_dir):
        if len(os.listdir(manual_image_dir)) == 1:
            #print ("manual image exists")
            manualImagePath = os.path.join(manual_image_dir, os.listdir(manual_image_dir)[0])
        elif len(os.listdir(manual_image_dir)) > 1:
            print ("manual image exists, but more than one")
        
    
    if "manualImagePath" not in locals():
        manualImagePath = originalVideoPaths[0]
        

    
    return jsonify({"originalVideoPaths": originalVideoPaths, "manualImagePath": 'http://'+frontend_host+':' + str(port) + '/image/' + manualImagePath})

@app.route('/get-model-path-lists', methods=['POST'])
def get_model_path_lists():
    datas = request.json
    selected_category = datas["Category"]
    selected_subcategory = datas["SubCategory"]
    selected_step = datas["Step"]
    selected_video = datas["Video"]
    selected_frame = datas["Frame"]

    #print("selected_video: ", selected_video)
    video_id = selected_video.split('/')[-1].split('_clip')[0]
    #print("video_id: ", video_id)
    if video_id == '':
        return jsonify({"error": "Please select a video."}), 500
    ### Return the list of parts like [[0,1], [2], [3]] with base path (xx/xx/xxx.obj), so that the frontend can load the parts
    modelFilePathLists = []
    
    parts_base_path = os.path.join('./dynamic_dataset/parts', selected_category, selected_subcategory)
    parts_base_to_return = 'http://'+frontend_host+':' + str(port) + '/image/' + os.path.join('./dynamic_dataset/parts', selected_category, selected_subcategory)

    # Find all the subparts list instead of get all avaliable files that ends with .obj in the parts_base_path using os
    obj_num_ls = []
    
    user = datas['user']
    data_json_path = users[user]['data_json_path']
    json_data = get_data(data_json_path)

    # Parts are in the format of "parts": ["0,2","1"]
    for data in json_data:
        if data['category'] == selected_category and data['name'] == selected_subcategory:
            for step in data['steps']:
                if step['step_id'] == int(selected_step.split('_')[-1]):

                    for video in step['video']:
                        fps = video['fps']
                        if video['video_id'].split('watch?v=')[-1] == video_id:
                            for frame in video['frames']:
                                if frame['frame_id'] == int(selected_frame):
                                    for part in frame['parts']:
                                        modelFilePathLists.append(part)
                                    user_data[user]['current_video_time'] =  frame['frame_time'] 
                                    print(f'##################[3]####### Set current_video_time to {frame["frame_time"]}')
                                    break
                            break
                    break
            break

        

    # Secondly, change the format of the parts to list of paths
    for i in range(len(modelFilePathLists)):
        print(f'modelFilePathLists[i]: {modelFilePathLists[i]}')
        modelFilePathLists[i] = modelFilePathLists[i].split(',')
        for j in range(len(modelFilePathLists[i])):
            modelFilePathLists[i][j] = modelFilePathLists[i][j].zfill(2)
            modelFilePathLists[i][j] = os.path.join(parts_base_to_return, modelFilePathLists[i][j] + '.obj')
    
    # #print the list of parts
    #print("modelFilePathLists: ", modelFilePathLists)

    # Whenever the video is changed, the video current time should be reset to 0
    user = datas['user']
    user_data[user]['selected_video'] = selected_video
    user_data[user]['fps'] = fps

    return jsonify({"modelFilePathLists": modelFilePathLists, 'currentFrameTime': user_data[user]['current_video_time']})



@app.route('/get-frames', methods=['POST'])
def get_frames():
    print(request.json)
    user = request.json['user']
    Category = request.json['Category']
    SubCategory = request.json['SubCategory']
    Step = request.json['Step']
    Video = request.json['Video']
    step_id = Step.split('_')[-1]

    data_json_path = users[user]['data_json_path']

    
    video_id = Video.split('/')[-2]


    data_json_path = users[user]['data_json_path']
    frames_ls = []
    data = get_data(data_json_path)
    for furniture in data:
        if furniture['category'] == Category and furniture['name'] == SubCategory:
            for step in furniture['steps']:
                if int(step['step_id']) == int(step_id):
                    for video in step['video']:
                        if video['video_id'].split('watch?v=')[-1] == video_id:
                            
                            for frame in video['frames']:
                                frames_ls.append(frame['frame_id'])

    



    
    return jsonify({"frames": frames_ls})


#################### Projection Modification ####################
def rotate(axis, degree):
    if axis == 'x':
        return [[1, 0, 0],
                [0, math.cos(math.radians(degree)), -math.sin(math.radians(degree))],
                [0, math.sin(math.radians(degree)), math.cos(math.radians(degree))]]
    elif axis == 'y':
        return [[math.cos(math.radians(degree)), 0, math.sin(math.radians(degree))],
                [0, 1, 0],
                [-math.sin(math.radians(degree)), 0, math.cos(math.radians(degree))]]
    elif axis == 'z':
        return [[math.cos(math.radians(degree)), -math.sin(math.radians(degree)), 0],
                [math.sin(math.radians(degree)), math.cos(math.radians(degree)), 0],
                [0, 0, 1]]
    
@app.route('/change-current-projection', methods=['POST'])
def change_current_projection():
    data = request.json
    selected_axis = data["Input1"]
    angle_or_distance= float(data["Input2"])
    rotation_or_translation = data["Input3"]

    user = data['user']
    pose_estimation_data_path = user_data[user]['pose_estimation_data_path']

    prev_pose_data = json.load(open(pose_estimation_data_path))
    ext_mat = np.array(prev_pose_data["extrinsic"])
    if rotation_or_translation == 'r':
        rotation_matrix = rotate(selected_axis, angle_or_distance)
        ext_mat[:3, :3] = np.dot(ext_mat[:3, :3], rotation_matrix)
    elif rotation_or_translation == 't':
        if selected_axis == 'x':
            ext_mat[0, 3] += angle_or_distance
        elif selected_axis == 'y':
            ext_mat[1, 3] += angle_or_distance
        elif selected_axis == 'z':
            ext_mat[2, 3] += angle_or_distance
    prev_pose_data["extrinsic"] = ext_mat.tolist()

    with open(pose_estimation_data_path, 'w') as outfile:
        json.dump(prev_pose_data, outfile, indent=4)

    # Plot diagram
    cmd_to_execute = "python ./project_part.py --json " + pose_estimation_data_path
    conda_cmd = f'conda run -n ikea-video-backend {cmd_to_execute}'
    ret = subprocess.run(conda_cmd, shell=True, capture_output=True, text=True)
    
    #print("Re-projection finished")
    return_code = ret.returncode
    #print('ret',ret)
    #print('return_code', return_code)
    

    return jsonify({"success": True})


#################### Save Functions ####################

def handle_frame(frame, parts, ext_mat=None, int_mat=None, new_mask=None, segmentation_data=None):
    parts_exist = False
    data_exist = False
    success = False
    # Check if parts already exists
    for i in range(len(frame['parts'])):
        if is_same_parts(frame['parts'][i],parts):
            parts_exist = True
            if ext_mat is not None and int_mat is not None: # Update extrinsic and intrinsic
                frame, data_exist = update_ext_int(frame, i, ext_mat, int_mat)
                #print("update_ext_int")
                success = True
                #print("update_ext_int : ", success)
            if new_mask is not None and segmentation_data is not None:# Update mask
                frame, data_exist = update_mask(frame, i, new_mask, segmentation_data)
                #print("update_mask")
                success = True
                #print("update_mask : ", success)
    if not parts_exist:
        for i in range(len(frame['parts'])):
            if contain_this_parts(parts, frame['parts'][i]):
                parts = frame['parts'][i]
                if ext_mat is not None and int_mat is not None:
                    frame, data_exist = update_ext_int(frame, i, ext_mat, int_mat)
                    #print("update_ext_int")
                    success = True
                    #print("update_ext_int : ", success)
                    parts_exist = True

        # frame, data_exist,success = add_parts(frame, parts, ext_mat, int_mat, new_mask) # We dont want to add part at current stage
        #print("add_parts : ", success)
    #print("handle_frame : ", success)
    return parts_exist, data_exist, success, parts

def add_parts(frame, parts, ext_mat, int_mat, new_mask):
    #  No need to remove and rename
    new_frame = frame.copy()
    data_exist = False
    new_frame['parts'].append(parts)

    success = False

    if ext_mat is not None and int_mat is not None: # Pose estimation mode
        new_frame['extrinsics'].append(ext_mat)
        new_frame['intrinsics'].append(int_mat)
        success = True
    else:
        new_frame['extrinsics'].append([])
        new_frame['intrinsics'].append([])
    
    if new_mask is not None: # Segmentation mode
        new_frame['mask'].append(new_mask)
        success = True
    else:
        new_frame['mask'].append({})

    return new_frame, data_exist, success

def update_ext_int(frame, index, ext_mat, int_mat):
    new_frame = frame.copy()
    data_exist = True
    assert len(new_frame['extrinsics']) == len(new_frame['intrinsics'])
    if new_frame['extrinsics'] == [] or new_frame['intrinsics'] == []:  
        new_frame['extrinsics'] = [[]] * len(new_frame['parts'])
        new_frame['intrinsics'] = [[]] * len(new_frame['parts'])
        data_exist = False
    elif new_frame['extrinsics'][index] == [] or new_frame['intrinsics'][index] == []:
        data_exist = False

    new_frame['extrinsics'][index] = ext_mat
    new_frame['intrinsics'][index] = int_mat


    return new_frame, data_exist

def update_mask(frame, index, new_mask, segmentation_data):
    new_frame = frame.copy()
    data_exist = True
    if new_frame['mask'] == []:  
        new_frame['mask'] = [{}] * len(new_frame['parts'])
        data_exist = False
    elif new_frame['mask'][index] == {}:
        data_exist = False


    new_frame['mask'][index] = new_mask


    return new_frame, data_exist

def remove_and_rename(old_image_path, new_image_path, image_type):
    new_image_path = new_image_path[:-4] + image_type + '.jpg'
    try:
        os.remove(old_image_path + image_type + '.jpg')
    except:
        print("Old image" + old_image_path + image_type + ".jpg not found")
    else:
        print("removing :", old_image_path + image_type + '.jpg')

    try:
        os.rename(new_image_path, old_image_path + image_type + '.jpg')
    except:
        print("New image" + new_image_path + " not found")
    else:
        print("rename: ", new_image_path.split('/')[-1], old_image_path.split('/')[-1] + image_type + '.jpg')


def add_frame(frame_num, parts, currentFrameTime = None,segmentation_data = None, pose_estimation_data = None):
    new_frame = {}
    new_frame['frame_id'] = frame_num
    new_frame['frame_time'] = currentFrameTime
    new_frame['parts'] = [parts]
    new_frame['connections'] = []
    if segmentation_data is not None:
        new_frame['mask'] = [segmentation_data['new_mask']]
    else:
        new_frame['mask'] = [{}]
    if pose_estimation_data is not None:
        new_frame['intrinsics'] = [pose_estimation_data['intrinsic']]
        new_frame['extrinsics'] = [pose_estimation_data['extrinsic']]
    else:
        new_frame['extrinsics'] = [[]]
        new_frame['intrinsics'] = [[]]
    return new_frame

@app.route('/save-mask', methods=['POST'])
def save_mask():
    print("Start saving mask")
    print(user_data)
    print(request.json)
    datas = request.json
    user = datas['user']
    user_data[user]['save_mask_done'] = False
    fps = user_data[user]['fps']
    currentFrameTime = user_data[user]['current_video_time']
    return_message ={}

    if 'frame' in datas.keys():
        frame_id = datas['frame']
    else:
        frame_id = -1


    
    segmentation_data_path = user_data[user]['segmentation_data_path']
    with open(segmentation_data_path) as f:
        segmentation_data = json.load(f)
    # Update frame number delay caused by async
    

    #### Clean the temp images
    new_image_path =''
    new_output_path_mask_only = ''
    new_output_path_with_frame = ''
    try:
        # #print(f'get output path from user data {user_data[user]}')
        output_path = user_data[user]['output_path'].split('/image/')[-1]
        video_id = '_'.join(output_path.split('_clip')[0].split('_')[1:]) 
        new_image_path = output_path.replace("_temp", "")
        print(f'renaming {output_path} to {new_image_path}')
        os.rename(output_path, new_image_path)
        print("Done renaming")
        output_path_mask_only = (output_path.split('.jpg')[0]+'_mask_only.jpg')
        new_output_path_mask_only = output_path_mask_only.replace("_temp", "")
        print(f'renaming {output_path_mask_only} to {new_output_path_mask_only}')
        os.rename(output_path_mask_only, new_output_path_mask_only)
        print("Done renaming")
        output_path_with_frame = (output_path.split('.jpg')[0]+'_with_frame.jpg')
        new_output_path_with_frame = output_path_with_frame.replace("_temp", "")
        print(f'renaming {output_path_with_frame} to {new_output_path_with_frame}')
        os.rename(output_path_with_frame, new_output_path_with_frame)
        print("Done renaming")
    except:
        print('ERROR: output not in user data')

    segmentation_data['frame_id'] = int(currentFrameTime*fps)
    segmentation_data['currentFrameTime'] = currentFrameTime
    with open(segmentation_data_path, 'w') as outfile:
        json.dump(segmentation_data, outfile, indent=4)

    part_idxs = segmentation_data["parts"]
    category = segmentation_data["category"]
    name = segmentation_data["name"]
    step_id = segmentation_data["step_id"]
    frame_num = int(currentFrameTime*fps)
    video_id = segmentation_data["video_id"]

    json_lock = False
    data_json_path = users[user]['data_json_path']
    json_data = get_data(data_json_path)
    video = get_video(json_data, category, name, step_id, video_id)
    success = False
    frame_exist = False
    done = False
    if video is not None:
        for frame in video['frames']:
            if int(frame['frame_id']) == int(frame_num) or (frame_id != -1 and int(frame['frame_id']) == int(frame_id)):
                frame_exist = True
                frame['frame_time'] = currentFrameTime
                parts_exist, data_exist, success, parts = handle_frame(frame, part_idxs,new_mask=segmentation_data['new_mask'], segmentation_data=segmentation_data)
                break
    else:
        #print("Saving mask done")
        user_data[user]['save_mask_done'] = True
        return_message = {"messages": ["Video does not exist, write new video, mask NOT saved!"], 'imagePath':new_image_path}
        done = True
    
    if not frame_exist and not done:
        user_data[user]['save_mask_done'] = True
        # return_message = {"messages": ["Mask Saved! Frame " + str(frame_num) + ", write new frame."], 'imagePath':new_image_path}
        return_message = {"messages": ["Frame " + str(frame_num) + " does not exist, mask not saved!"], 'imagePath':new_image_path}
        done = True
    if not done:
        with open(data_json_path, 'w') as outfile:
            json_lock = True
            json.dump(json_data, outfile, indent=4)
            json_lock = False


    if not success and not done:
        #print("Saving mask done")
        user_data[user]['save_mask_done'] = True

        return_message = {"messages": ["Save failed for " + str(frame_num) + " parts " + parts + ', check your data!'], 'imagePath':new_image_path}
    else:
        if not done:
            
            if not frame_exist:
                #print("Saving mask done")
                user_data[user]['save_mask_done'] = True
                
                return_message = {"messages": ["Mask Saved! Frame " + str(frame_num) + " does not exist. New frame saved."], 'imagePath':new_image_path}
            elif not parts_exist:
                #print("Saving mask done")
                user_data[user]['save_mask_done'] = True

                # return_message = {"messages": ["Mask Saved! Frame " + str(frame_num) + " exist, but parts " + parts + " data does not exist. New parts data saved."], 'imagePath':new_image_path}
                return_message = {"messages": ["Frame " + str(frame_num) + " exist, but parts " + parts + " data does not exist. Mask not saved !"], 'imagePath':new_image_path}

            elif not data_exist:
                #print("Saving mask done")
                user_data[user]['save_mask_done'] = True

                return_message = {"messages": ["Mask Saved! Frame " + str(frame_num) + " , parts " + parts + " exist, but mask data does not exist. New mask data saved."], 'imagePath':new_image_path}
            else:
                user_data[user]['save_mask_done'] = True
                return_message = {"messages": ["Mask Saved! Frame " + str(frame_num) +" Parts " + parts + " already exist. Update mask!"], 'imagePath':new_image_path }
    while json_lock:
        wait(0.1)
    ## Viz mask
    category = datas['category']
    name = datas['name']
    step_id = datas['step']
    current_video = user_data[user]['selected_video']
    video_id = current_video.split('/')[-1].split('_clip')[0]
    currentModelFilePaths = datas['part']
    # parts = [currentModelFilePath.split('/')[-1].split('.')[0] for currentModelFilePath in currentModelFilePaths]
    parts = [str(int(part)) for part in parts.split(',')]
    parts = ''.join(parts)
    json_path = os.path.join(os.getcwd() +'/'+ users[user]['data_json_path'])
    cmd_to_execute = f"python ./data/viz_mask_one.py --json {json_path} --name {name} --category {category} --step {step_id} --video='{video_id}' --frame {int(currentFrameTime*fps)} --part {parts} --user {user}"
    conda_cmd = f'conda run -n ikea-video-backend {cmd_to_execute}'
    ret = subprocess.run(conda_cmd, shell=True, capture_output=True, text=True)
    print('ret',ret)
    print("Viz mask finished")
    ## Viz mask
    return jsonify(return_message), 200

@app.route('/save-projection', methods=['POST'])
def save_projection():
    datas = request.json
    user = datas['user']
    useShareFrames = datas['useShareFrames']
    shareFrames = datas['shareFrames']
    shareFrames = [int(shareFrame) for shareFrame in shareFrames]
    user_data[user]['save_project_done'] = False
    currentFrameTime = user_data[user]['current_video_time']
    fps = user_data[user]['fps']
    


    pose_estimation_data_path = user_data[user]['pose_estimation_data_path']
    pose_estimation_data = get_data(pose_estimation_data_path)
    
    # Update frame number delay caused by async
    pose_estimation_data['frame_id'] = int(currentFrameTime*fps)
    pose_estimation_data['currentFrameTime'] = currentFrameTime

    #### Clean the temp images
    new_image_path =''
    try:
        output_path = user_data[user]['output_path'].split('/image/')[-1]
        new_image_path = output_path.replace("_temp", "")
        os.rename(output_path, new_image_path)
    except:
        print('ERROR: output not in user data')

    with open(pose_estimation_data_path, 'w') as outfile:
        json.dump(pose_estimation_data, outfile, indent=4)

    ext_mat = pose_estimation_data["extrinsic"]
    int_mat = pose_estimation_data["intrinsic"]
    part_idxs = pose_estimation_data["part_idxs"]
    category = pose_estimation_data["category"]
    name = pose_estimation_data["name"]
    step_id = pose_estimation_data["step_id"]
    frame_num = int(currentFrameTime*fps)
    video_id = pose_estimation_data["video_id"]
    json_lock = False

    data_json_path = users[user]['data_json_path']
    
    json_data = get_data(data_json_path)
    video = get_video(json_data, category, name, step_id, video_id)


    
    return_message ={}
    if not useShareFrames:
        shareFrames = [frame_num]
    else:
        shareFrames.append(frame_num)

    for frame_num in shareFrames:
        currentFrameTime = frame_num/fps
        print(f'Saving projection for frame {frame_num}')
        done = False
        success = False
        parts_exist = False
        
        frame_exist = False
        if video is not None:
            success = False
            for frame in video['frames']:
                if int(frame['frame_id']) == int(frame_num):
                    frame_exist = True
                    parts_exist, data_exist, success, parts = handle_frame(frame, part_idxs, ext_mat=ext_mat, int_mat=int_mat)
                    break
        else:
            user_data[user]['save_project_done'] = True
            return_message = {"messages": ["Video does not exist, write new video, projection NOT saved!"], 'imagePath':new_image_path}
            done = True


        if not frame_exist and not done:
            # new_frame = add_frame(frame_num, parts,currentFrameTime,pose_estimation_data = pose_estimation_data)
            # video['frames'].append(new_frame)
            # with open(data_json_path, 'w') as outfile:
            #     json.dump(json_data, outfile, indent=4)
            user_data[user]['save_project_done'] = True
            # return_message = {"messages": ["Projection Saved! Frame " + str(frame_num) + ", write new frame, mask saved for " + str(frame_num) + " parts " + parts + '!'], 'imagePath':new_image_path}
            return_message = {"messages": ["Frame " + str(frame_num) + " does not exist, projection not saved!"], 'imagePath':new_image_path}
            done = True
        
        
        if not done:
            with open(data_json_path, 'w') as outfile:
                json_lock = True
                json.dump(json_data, outfile, indent=4)
                json_lock = False

        if not success and not done:
            user_data[user]['save_project_done'] = True
            return_message = {"messages": ["Save failed for " + str(frame_num) + " parts " + parts + ', check your data!'], 'imagePath':new_image_path}
        else:
            if not done:
            
                if not frame_exist:
                    user_data[user]['save_project_done'] = True

                    # return_message = {"messages": ["Projection Saved!  Frame " + str(frame_num) + " does not exist. New frame saved."], 'imagePath':new_image_path}
                    return_message = {"messages": ["Frame " + str(frame_num) + " does not exist, projection not saved!"], 'imagePath':new_image_path}
                elif not parts_exist:
                    user_data[user]['save_project_done'] = True

                    # return_message = {"messages": ["Projection Saved!  Frame " + str(frame_num) + " exist, but parts " + parts + " data does not exist. New parts data saved."], 'imagePath':new_image_path}
                    return_message = {"messages": ["Frame " + str(frame_num) + " exist, but parts " + parts + " data does not exist. Data not saved"], 'imagePath':new_image_path}

                elif not data_exist:
                    user_data[user]['save_project_done'] = True

                    return_message = {"messages": ["Projection Saved!  Frame " + str(frame_num) + " , parts " + parts + " exist, but projection data does not exist. New projection data saved."], 'imagePath':new_image_path}
                else:
                    # After change the logic for data saving, the data will be saved automatically, no need to remove and rename anymore
                    # old_image_path = pose_estimation_data['output_path'][:-8] # Before _new.jpg
                    # new_image_path = pose_estimation_data['output_path']
                    # remove_and_rename(old_image_path, new_image_path, '')
                    user_data[user]['save_project_done'] = True

                    return_message = {"messages": ["Projection Saved!  Frame " + str(frame_num) +" Parts " + parts + " already exist. Update projection!"], 'imagePath':new_image_path}
        while json_lock:
            wait(0.1)
        ### Plot Pose-estimation data to the 
        if parts_exist and frame_exist and success:
            category = datas['category']
            name = datas['name']
            step_id = datas['step']
            current_video = user_data[user]['selected_video']
            video_id = current_video.split('/')[-1].split('_clip')[0]
            currentModelFilePaths = datas['part']
            # part = [currentModelFilePath.split('/')[-1].split('.')[0] for currentModelFilePath in currentModelFilePaths]
            parts = [str(int(p)) for p in parts.split(',')]
            parts = ''.join(parts)
            json_path = os.path.join(os.getcwd() +'/'+ users[user]['data_json_path']) # Same user, so should be same data_json_path, no need to change here
            cmd_to_execute = f"python ./data/viz_pose_one.py --json {json_path} --name {name} --category {category} --step {step_id} --video='{video_id}' --frame {frame_num} --part {parts} --user {user}"
            conda_cmd = f'conda run -n ikea-video-backend {cmd_to_execute}'
            ret = subprocess.run(conda_cmd, shell=True, capture_output=True, text=True)
            print('ret', ret)
            print("Viz pose finished for frame ", frame_num)

    return jsonify(return_message), 200


if __name__ == '__main__':
    # run on 0.0.0.0 

    app.run(host = '0.0.0.0', port=port, threaded=True) #Enable threaded=True to allow multiple connections.
