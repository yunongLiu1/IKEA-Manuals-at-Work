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
from utils.get_mse import get_mse_score
import time
from utils.users import users


#### Global Variables ####
OUTPUT_FOLDER = "output"
# Placeholders (to be replaced or set via env vars)
OBJ_FOLDER = "OBJ_FOLDER_PLACEHOLDER"
BASE_JSON_PATH = "BASE_JSON_PATH_PLACEHOLDER" #e.g. './data.json'
VIDEO_FOLDER = "VIDEO_FOLDER_PLACEHOLDER"

#### Port and Host Configurations ####
app = Flask(__name__)
CORS(app) 
port = 5000
host = 'HOST_PLACEHOLDER' #e.g. 'localhost'

#### User Data ####
user_data = {}
json_writing_lock = {}

for user in users:
    json_writing_lock[user] = False

#### Load base json data #####
with open(BASE_JSON_PATH, 'r') as f:
    BASE_JSON_DATA = json.load(f)

################### Load frame and save frame #######################
def load_frame(data, category, name, frame_id, video_id):
    for furniture in data:
        print('Checking furniture', furniture['name'])
        if furniture['category'] == category and furniture['name'] == name:
            print('Found furniture', furniture['name'])
            for step in furniture['steps']:
                for video in step['video']:
                    if video['video_id'].split('watch?v=')[-1] == video_id:
                        for frame in video['frames']:
                            if int(frame['frame_id']) == int(frame_id):
                                print('Found frame', frame['frame_id'])
                                return frame
    
    print('Frame not found')
    return None

def save_frame(user_id, json_path, data, category, name, frame_id, video_id, frame):
    print('######### Saving frame #########')
    for furniture in data:
        if furniture['category'] == category and furniture['name'] == name:
            for step in furniture['steps']:
                for video in step['video']:
                    if video['video_id'].split('watch?v=')[-1] == video_id:
                        for i, curr_frame in enumerate(video['frames']):
                            if int(curr_frame['frame_id']) == int(frame_id):
                                video['frames'][i] = frame
                                
                                while json_writing_lock[user_id]:
                                    print('Waiting for json writing lock')
                                    time.sleep(0.1)

                                json_writing_lock[user_id] = True
                                with open(json_path, 'w') as f:
                                    json.dump(data, f, indent=4)
                                
                                json_writing_lock[user_id] = False
                                print('Frame saved')
                                return True
    return False


#################### Transform object ############################

@app.route("/transform-objects", methods=["POST"])
def transform_objects():
    user_id = request.args.get('user_id')
    json_path = users[user_id]['data_json_path']

    while json_writing_lock[user_id]:
        print('Waiting for json writing lock')
        time.sleep(0.1)
    with open(json_path) as f:
        json_data = json.load(f)
    data = request.get_json()
    category = request.args.get('category')
    name = request.args.get('name')
    frame_id = request.args.get('frame_id')
    video_id = request.args.get('video_id')
    rotations = data["rotations"] # format: [90,60,30]
    translations = data["translations"] # format: [1,2,3]
    print('rotations', rotations)
    print('translations', translations)
    obj_paths = data["objPaths"]
    frame = get_ext_from_prev_frame_with_same_int_and_len_parts(json_path, category, name, frame_id, video_id, user_id)
    extrinsics = frame['extrinsics']
    extrinsics = np.array(extrinsics)
    print(extrinsics[0])
    if extrinsics[0].shape != (4, 4):
        print('Extrinsics not annotated')
        frame = load_frame(BASE_JSON_DATA, category, name, frame_id, video_id)
        if frame is None:
            print('Frame not found when getting obj paths')
            return jsonify({"objPaths": []})
    extrinsics = frame['extrinsics']

    rotations = rotations.values()
    translations = translations.values()
    
    new_extrinsics = extrinsics.copy()
    if frame is None:
        return jsonify({"success": False})
    for rotation, translation, obj_path in zip(rotations, translations, obj_paths):
        new_extrinsic, curr_part_index = transform_object(json_path, request, obj_path, rotation, translation, extrinsics)
        new_extrinsics[curr_part_index] = new_extrinsic.tolist()
        # Save the updated extrinsics
    frame['extrinsics'] = new_extrinsics
    save_frame(user_id,json_path, json_data, category, name, frame_id, video_id, frame)

    return jsonify({"success": True})
    
    

    
def transform_object(json_path, request, obj_paths, rotation, translation, extrinsics):
    print('#### Transforming object')
    data = request.get_json()
    category = request.args.get('category')
    name = request.args.get('name')
    frame_id = request.args.get('frame_id')
    video_id = request.args.get('video_id')
    user_id = request.args.get('user_id')

    while json_writing_lock[user_id]:
        print('Waiting for json writing lock')
        time.sleep(0.1)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    
    ids_from_obj_paths = obj_paths[0].split('.')[0].split('/')[-1]
    

    print('rotation ',rotation,', translation ', translation)
    frame = load_frame(json_data, category, name, frame_id, video_id)
    
    part_ids = frame['parts']

    # objPaths 0112070203081006050904 to curr_part_id 01,12,07,02,03,08,10,06,05,09,04
    curr_part_id = []
    for i in range(len(ids_from_obj_paths)//2):
        curr_part_id.append(ids_from_obj_paths[i*2:i*2+2])
    curr_part_id = [str(int(part_id)) for part_id in curr_part_id]

    # curr_part_id = [str(int(obj_path.split("/")[-1].split(".")[0])) for obj_path in obj_paths]
    curr_part_id = ",".join(curr_part_id)
    print('curr_part_id', curr_part_id)
    curr_part_index = part_ids.index(curr_part_id)
    print('curr_part_index', curr_part_index)

    extrinsics = np.array(extrinsics)
    print('extrinsics[curr_part_index]' , extrinsics[curr_part_index])
    # Scale down the translation
    translation = np.array(translation)

    # Load the object mesh using trimesh
    mesh = None
    for obj_path in obj_paths:
        mesh = mesh + trimesh.load(obj_path) if mesh is not None else trimesh.load(obj_path)

    obj_paths = [[f"{OBJ_FOLDER}/{category}/{name}/{part_id.zfill(2)}.obj" for part_id in curr_part_ids.split(",")] for curr_part_ids in part_ids]

    print('obj_paths', obj_paths)
    print('rotation',  rotation)
    # print('rotatino_mat', get_rotation_matrix(rotation_angle, rotation_axis))
    print('translation', translation)

    for i, objFiles in enumerate(obj_paths):
        curr_meshes = None
        for objFile in objFiles:
            cam_to_t1 = np.linalg.inv(extrinsics[0]) @ extrinsics[i]  # T0_Ti = T0_Cam @ Cam_Ti
            print('extrinsics[0]', extrinsics[0])
            print('extrinsics[i]', extrinsics[i])
            if i == curr_part_index:
                print('cam_to_t1', cam_to_t1)
                if rotation != [0,0,0]:
                    print('Applying rotation')
                    rotation_matrix = get_rotation_matrix(rotation)
                    print('total rotation matrix', rotation_matrix)
                    # rotation_matrix = extrinsics[0] @ rotation_matrix
                    cam_to_t1 =  rotation_matrix @ cam_to_t1
                cam_to_t1[:3, 3] += translation
                print('final transformation matrix', cam_to_t1)
                new_extrinsic = extrinsics[0] @ cam_to_t1
                print('new_extrinsic, curr_part_index', new_extrinsic, curr_part_index)
                return new_extrinsic, curr_part_index
                

def get_rotation_matrix(rotation):
    """
    Compute the rotation matrix for a given rotation angle (in radians) around a specified axis.
    
    Args:
    rotation (float): The rotation angle in radians.
    Returns:
    numpy.ndarray: A 3x3 rotation matrix.
    """
    # Initialize the rotation matrix as a 3x3 identity matrix
    total_rotation_matrix = np.eye(4)

    for i in range(3):
        
        rotation_matrix = np.eye(4)
        rotation_angle = np.radians(rotation[i]) # Convert the rotation angle to radians
        # rotation_angle = rotation[i]

    
        # Calculate cosine and sine of the rotation angle
        c = np.cos(rotation_angle)
        s = np.sin(rotation_angle)
    
        # Construct the rotation matrix based on the specified axis
        if i == 0:
            rotation_matrix[1, 1] = c
            rotation_matrix[1, 2] = -s
            rotation_matrix[2, 1] = s
            rotation_matrix[2, 2] = c
        elif i == 1:
            rotation_matrix[0, 0] = c
            rotation_matrix[0, 2] = s
            rotation_matrix[2, 0] = -s
            rotation_matrix[2, 2] = c
        elif i == 2:
            rotation_matrix[0, 0] = c
            rotation_matrix[0, 1] = -s
            rotation_matrix[1, 0] = s
            rotation_matrix[1, 1] = c
        
        print('rotation_matrix', rotation_matrix)
        
        total_rotation_matrix = total_rotation_matrix @ rotation_matrix
    
    return total_rotation_matrix



################### Load camera parameters #######################
@app.route("/camera-parameters-and-obj-paths")
def get_camera_parameters():
    category = request.args.get('category')
    name = request.args.get('name')
    frame_id = request.args.get('frame_id')
    video_id = request.args.get('video_id')
    user_id = request.args.get('user_id')

    json_path = users[user_id]['data_json_path']

    if user_id not in user_data:
        user_data[user_id] = {}
    
    print('Getting camera parameters')
    while json_writing_lock[user_id]:
        print('Waiting for json writing lock')
        time.sleep(0.1)
    with open(json_path) as f:
        json_data = json.load(f)
    
    frame = get_ext_from_prev_frame_with_same_int_and_len_parts(json_path, category, name, frame_id, video_id, user_id)
    
    if frame is None:
        return jsonify({"extrinsics": [], "intrinsics": []})
            

    extrinsics = frame['extrinsics']
    intrinsics = frame['intrinsics']
    ### Set user data
    user_data[user_id]['extrinsics'] = extrinsics
    user_data[user_id]['intrinsics'] = intrinsics

    print(extrinsics)
    print(intrinsics)
    return_obj_paths = get_meshes_based_on_camera_parameters(user_id, category, name, frame_id, video_id, frame)

    return jsonify({"extrinsics": extrinsics, "intrinsics": intrinsics, "objPaths": return_obj_paths})
################### Load object paths #######################

def get_meshes_based_on_camera_parameters(user_id, category, name, frame_id, video_id, frame):
    while json_writing_lock[user_id]:
        print('Waiting for json writing lock')
        time.sleep(0.1)
    if frame is None:
        print('Frame not found when getting obj paths')
        return jsonify({"objPaths": []})
    extrinsics = frame['extrinsics']

    part_ids = frame['parts']
    print('Loading extrinsics', extrinsics)
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    scene = trimesh.Scene()

    return_obj_paths = []
    obj_paths = [[f"{OBJ_FOLDER}/{category}/{name}/{part_id.zfill(2)}.obj" for part_id in curr_part_ids.split(",")] for curr_part_ids in part_ids]
    curr_output_dir = os.path.join(OUTPUT_FOLDER, category, name, video_id,frame_id)
    if not os.path.exists(curr_output_dir):
        os.makedirs(curr_output_dir)
    obj_output_names = [[part.zfill(2) for part in parts.split(',')] for parts in part_ids]
    obj_output_names = [''.join(parts) for parts in obj_output_names]
    print(obj_paths)
    
    for i, objFiles in enumerate(obj_paths):
        curr_meshes = None
        curr_return_obj_paths = []
        for objFile in objFiles:
            mesh = trimesh.load(objFile)
            cam_to_t1 = np.linalg.inv(extrinsics[0]) @ extrinsics[i]
            mesh.apply_transform(cam_to_t1)
            # mesh.apply_transform(extrinsics[i])
            scene.add_geometry(mesh, node_name=str(i))
            curr_meshes = mesh if curr_meshes is None else curr_meshes + mesh
            # export the mesh to a obj file
        curr_meshes.export(os.path.join(f"{curr_output_dir}/{obj_output_names[i]}.obj"))
        curr_return_obj_paths.append(os.path.join(f"{curr_output_dir}/{obj_output_names[i]}.obj"))
        return_obj_paths.append(curr_return_obj_paths)

    
    print('return_obj_paths',return_obj_paths)
    return return_obj_paths

def get_ext_from_prev_frame_with_same_int_and_len_parts(json_path, category, name, frame_id, video_id,user_id):
    ####################### Get Camera Parameters ############
    # 1. if annotated, use what saved in file
    # 2. If not, if prev frame exist in the user data, use it
    # 3. otherwise, if prev frame exist in the total data, use it.
    # 4. Otherwise, use what saved in total data for current frame
    with open(json_path) as f:
        json_data = json.load(f)
    frame = load_frame(json_data, category, name, frame_id, video_id)
    if frame is None:
        return None
    extrinsics = frame['extrinsics']
    intrinsic = frame['intrinsics'][0]
    try:
        extrinsics = np.array(extrinsics)
        for e, extrinsinc in enumerate(frame['extrinsics']):
            assert np.array(extrinsinc).shape ==(4,4)
    except:
        print('Extrinsics not annotated')
        # First check if the previous frame exist
        prev_frame_exist = True
        prev_frame_exist_with_same_int = False
        prev_frame = is_prev_frame_exist(user_id, json_path, category, name, frame_id, video_id)
        
        if prev_frame is None or np.array(prev_frame['extrinsics'])[0].shape != (4, 4):
            ## If exist in combined data
            prev_frame = is_prev_frame_exist(user_id, BASE_JSON_PATH, category, name, frame_id, video_id)
            if prev_frame is None:
                prev_frame_exist = False
        print('Previous Frame Exist', prev_frame_exist)
        if prev_frame_exist:
            prev_int = prev_frame['intrinsics'][0]
            print(np.array(prev_int), np.array(intrinsic))
            if np.allclose(np.array(prev_int), np.array(intrinsic)):
                # for e, extrinsinc in enumerate(prev_frame['extrinsics']):
                #     if np.array(extrinsinc).shape !=(4,4):
                #         prev_frame['extrinsics'][e] = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

                #         print('Change extrinisnc to ',np.eye(4))
                if len(prev_frame['parts']) != len(frame['parts']):
                    if len(prev_frame['parts']) > len(frame['parts']):
                        prev_frame['extrinsics'] = prev_frame['extrinsics'][:len(frame['parts'])]
                        prev_frame['intrinsics'] = prev_frame['intrinsics'][:len(frame['parts'])]
                    else:
                        for i in range(len(prev_frame['parts']), len(frame['parts'])):
                            prev_frame['extrinsics'].append(prev_frame['extrinsics'][-1])
                            prev_frame['intrinsics'].append(prev_frame['intrinsics'][-1])
            
                frame['extrinsics'] = prev_frame['extrinsics']
                prev_frame_exist_with_same_int = True
        
        if not prev_frame_exist_with_same_int: # 
            frame = load_frame(BASE_JSON_DATA, category, name, frame_id, video_id)
            if frame is None:
                print('Frame not found when getting obj paths')
                return None
        
    return frame

############### Helper functions to load video and frame image ############################

def load_video(category, name, video_id):
    video_path = os.path.join(VIDEO_FOLDER, category, name, video_id, f"{video_id}.mp4")
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f'Could not open video {video_path}')
    
    return video

def get_frame_image(user_id, json_path, category, name, frame_id, video_id):
    
    video = load_video(category, name, video_id)
    while json_writing_lock[user_id]:
        print('Waiting for json writing lock')
        time.sleep(0.1)
    with open(json_path) as f:
        json_data = json.load(f)
    frame = load_frame(json_data, category, name, frame_id, video_id)
    frame_time = frame['frame_time']
    video.set(cv2.CAP_PROP_POS_MSEC, frame_time*1000)
    ret, frame_img = video.read()
    if not ret:
        raise RuntimeError(f'Could not read frame {frame_id} from video {video_path}')
    
    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
    return frame_img
# Add a new route to serve the image path


########################## Overlay image ############################
@app.route("/overlay-image", methods=["GET"])
def get_overlay_image():
    category = request.args.get('category')
    name = request.args.get('name')
    frame_id = request.args.get('frame_id')
    video_id = request.args.get('video_id')
    user_id = request.args.get('user_id')
    json_path = users[user_id]['data_json_path']
    
    while json_writing_lock[user_id]:
        print('Waiting for json writing lock')
        time.sleep(0.1)
    with open(json_path) as f:
        json_data = json.load(f)
    frame = load_frame(json_data, category, name, frame_id, video_id)
    frame_img = get_frame_image(user_id, json_path, category, name, frame_id, video_id)
    img_width, img_height = frame_img.shape[1], frame_img.shape[0]

    ### Set user data
    if user_id not in user_data:
        user_data[user_id] = {}

    user_data[user_id]['img_width'] = img_width
    user_data[user_id]['img_height'] = img_height
    
    img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (img_width//3, img_height//3))
    output_dir = os.path.join('images', category, name, video_id, frame_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(f'./images/{category}/{name}/{video_id}/{frame_id}.png', img)
    
    return jsonify({"imgPath": f'../images/{category}/{name}/{video_id}/{frame_id}.png'})


################### Load masks data #######################
@app.route("/masks-data")
def get_mask_paths():
    category = request.args.get('category')
    name = request.args.get('name')
    frame_id = request.args.get('frame_id')
    video_id = request.args.get('video_id')
    user_id = request.args.get('user_id')
    json_path = users[user_id]['data_json_path']

    while json_writing_lock[user_id]:
        print('Waiting for json writing lock')
        time.sleep(0.1)
    with open(json_path) as f:
        json_data = json.load(f)
    frame = load_frame(json_data, category, name, frame_id, video_id)
    masks = frame['mask']
    
    masks_paths = []
    frame_img = get_frame_image(user_id, json_path, category, name, frame_id, video_id)
    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
    img_height, img_width = frame_img.shape[:2]
    for mi, mask in enumerate(masks):
        # print(mask)
        try:
            decoded_mask = mask_utils.decode(mask)
            print('Decode mask success, shape:', decoded_mask.shape)
            # Save image to ./images
            image = decoded_mask*255
            image = cv2.resize(image, (img_width, img_height))
            # Color
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(frame_img, 0.5, image, 0.5, 0)
        except:
            # Handle decoding error
            print('Decoding error, using dummy mask')
            overlay = frame_img
        # Save the image with low resolution to reduce the size
        overlay = cv2.resize(overlay, (img_width//3, img_height//3))
        output_dir = os.path.join('images', category, name, video_id, frame_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(f'./images/{category}/{name}/{video_id}/{frame_id}_{mi}.png', overlay)
        masks_paths.append(f'../images/{category}/{name}/{video_id}/{frame_id}_{mi}.png')

    return jsonify({"masksPaths": masks_paths})


######################## Load cat, name, frame_id, video_id ############################
@app.route("/user_ids")
def get_user_ids():
    return jsonify({"userIds": list(users.keys())})

@app.route("/categories")
def get_categories():
    user_id = request.args.get('user_id')
    if user_id == 'undefined':
        return jsonify({"categories": ['Select Category']})
    json_path = users[user_id]['data_json_path']

    while json_writing_lock[user_id]:
        print('Waiting for json writing lock')
        time.sleep(0.1)
    with open(json_path) as f:
        json_data = json.load(f)
    categories = [furniture['category'] for furniture in json_data]
    return jsonify({"categories": list(set(categories))})

@app.route("/names")
def get_names():
    user_id = request.args.get('user_id')
    if user_id == 'undefined':
        return jsonify({"names": ['Select Name']})
    json_path = users[user_id]['data_json_path']

    while json_writing_lock[user_id]:
        print('Waiting for json writing lock')
        time.sleep(0.1)
    with open(json_path) as f:
        json_data = json.load(f)
    category = request.args.get('category')
    user_data['category'] = category
    print('Set category', category)
    names = [furniture['name'] for furniture in json_data if furniture['category'] == category]
    return jsonify({"names": list(set(names))})

@app.route("/frame-ids")
def get_frame_ids():
    user_id = request.args.get('user_id')
    if user_id == 'undefined':
        return jsonify({"frameIds": ['Select Frame']})
    json_path = users[user_id]['data_json_path']

    while json_writing_lock[user_id]:
        print('Waiting for json writing lock')
        time.sleep(0.1)
    with open(json_path) as f:
        json_data = json.load(f)
    category = request.args.get('category')
    name = request.args.get('name')
    user_data['name'] = name
    print('Set name', name)
    video_id = request.args.get('video_id')
    user_data['video_id'] = video_id
    print('Set video_id', video_id)
    frame_ids = []
    for furniture in json_data:
        if furniture['category'] == category and furniture['name'] == name:
            for step in furniture['steps']:
                for video in step['video']:
                    if video['video_id'].split('watch?v=')[-1] == video_id:
                        for frame in video['frames']:
                            frame_ids.append(frame['frame_id'])
    
    user_data['frame_ids'] = frame_ids
    return jsonify({"frameIds": frame_ids})

@app.route("/video-ids")
def get_video_ids():
    user_id = request.args.get('user_id')
    if user_id == 'undefined':
        return jsonify({"videoIds": ['Select Video']})
    json_path = users[user_id]['data_json_path']

    while json_writing_lock[user_id]:
        print('Waiting for json writing lock')
        time.sleep(0.1)
    with open(json_path) as f:
        json_data = json.load(f)
    category = request.args.get('category')
    user_data['category'] = category
    name = request.args.get('name')
    user_data['name'] = name
    video_ids = []
    for furniture in json_data:
        if furniture['category'] == category and furniture['name'] == name:
            for step in furniture['steps']:
                video_ids.extend([video['video_id'].split('watch?v=')[-1] for video in step['video']])
    return jsonify({"videoIds": list(set(video_ids))})



def is_prev_frame_exist(user_id, json_path, category, name, frame_id, video_id):

    while json_writing_lock[user_id]:
        print('Waiting for json writing lock')
        time.sleep(0.1)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    for furniture in json_data:
        print('Checking furniture', furniture['name'])
        if furniture['category'] == category and furniture['name'] == name:
            print('Found furniture', furniture['name'])
            for step in furniture['steps']:
                for video in step['video']:
                    if video['video_id'].split('watch?v=')[-1] == video_id:
                        for fi, frame in enumerate(video['frames']):
                            if int(frame['frame_id']) == int(frame_id):
                                if fi > 0:
                                    return video['frames'][fi-1]
                                else:
                                    return None

@app.route("/load-pose-from-the-prev-frame")
def load_pose_from_the_prev_frame():
    ## TODO: make sure len(objPath) the same
    category = request.args.get('category')
    name = request.args.get('name')
    video_id = request.args.get('video_id')
    frame_id = request.args.get('frame_id')
    user_id = request.args.get('user_id')
    json_path = users[user_id]['data_json_path']

    if user_id not in user_data:
        user_data[user_id] = {}
    
    print('Getting camera parameters')
    while json_writing_lock[user_id]:
        print('Waiting for json writing lock')
        time.sleep(0.1)
    with open(json_path) as f:
        json_data = json.load(f)

    prev_frame = is_prev_frame_exist(user_id, json_path, category, name, frame_id, video_id)
    if prev_frame is None:
        return jsonify({'success': False})

    frame = load_frame(json_data, category, name, frame_id, video_id)
    
    extrinsics = prev_frame['extrinsics']
    extrinsics = np.array(extrinsics)
    if extrinsics[0].shape != (4, 4):
        print('Extrinsics not annotated')
        prev_frame = is_prev_frame_exist(user_id, BASE_JSON_PATH, category, name, frame_id, video_id)
        if prev_frame is None:
            return jsonify({'success': False})
    
        

    prev_intrinsic =  np.array(prev_frame['intrinsics'][0])

    curr_intrinsic = np.array(frame['intrinsics'][0])

    ## Check if they are close to each other 
    if (prev_intrinsic == prev_intrinsic).all():
        
        intrinsics = prev_frame['intrinsics']
        extrinsics = prev_frame['extrinsics']

        ### Set user data
        user_data[user_id]['extrinsics'] = extrinsics
        user_data[user_id]['intrinsics'] = intrinsics

        print(extrinsics)
        print(intrinsics)


        frame['extrinsics'] = extrinsics
        frame['intrinsics'] = intrinsics
    data = json_data
    save_frame(user_id, json_path, data, category, name, frame_id, video_id, frame)

    return jsonify({'success': True})


@app.route("/get-mse-score")
def mse_score():
    user_id = request.args.get('user_id')
    json_path = users[user_id]['data_json_path']
    category = request.args.get('category')
    name = request.args.get('name')
    frame_id = request.args.get('frame_id')
    video_id = request.args.get('video_id')

    print('Getting mse score')

    while json_writing_lock[user_id]:
        print('Waiting for json writing lock')
        time.sleep(0.1)
    result = get_mse_score(category, name, frame_id, video_id, json_path)

    mse_scores = result['mse_scores']

    return jsonify({"mseScores": mse_scores})



################# Progress Bar #############
@app.route('/progress-bar')
def get_progress_bar():
    category = request.args.get('category')
    name = request.args.get('name')
    video_id = request.args.get('video_id')
    frame_id = request.args.get('frame_id')
    user_id = request.args.get('user_id')
    print('User ID:', user_id)
    json_path = users[user_id]['data_json_path']
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    total = 0
    annotated = 0
    annotated_image = 0
    total_image = 0
    return_message = 'Not Annotated: \n'
    for furniture in data:
        for step in furniture['steps']:
            for video in step['video']:
                for fi, frame in enumerate(video['frames']):
                    total += len(frame['parts'])
                    is_annotated = True
                    total_image +=1
                    print('frame', frame['frame_id'])
                    for pi, part in enumerate(frame['parts']):  
                        if frame['extrinsics'][pi] != []:
                            annotated +=1
                        else:
                            is_annotated = False
                    if not is_annotated:
                        return_message += f"Category: {furniture['category']} Name {furniture['name']} Video ID {video['video_id'].split('watch?v=')[-1]} Frame {frame['frame_id']}"
                        return_message += '\n'
                        is_annotated = True
                    else:
                        annotated_image +=1

    return jsonify({'progress': int((annotated/total)*100), 'total': total, 'annotated': annotated, 'message': return_message, 'progressImage': int((annotated_image/total_image)*100), 'totalImage': total_image, 'annotatedImage': annotated_image})

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port=port, threaded=True) #Enable threaded=True to allow multiple connections.
