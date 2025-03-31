
import os
import json
import argparse



def get_key_and_video_id(img):
    offset = 0

    if 'pose-estimation' in img:
        img = img.split('pose-estimation/')[-1]
    else:
        img = img.split('masks/')[-1]
    if 'pose-estimation' in img or 'masks' in img:
        offset = 1
    
    # print(img)
    furniture_category= img.split('/')[offset]
    furniture_name = img.split('/')[offset+1]
    step_id = img.split('/')[offset+2]
    step_id = step_id.split('_')[-1]
    frame_id = img.split('/')[offset+3].split('_clip')[-1].split('_')[0]
    video_id = '_'.join(img.split('/')[offset+3].split('_clip')[0].split('_')[1:])
    part_id = img.split('/')[offset+3].split('_parts')[-1].split('_')[0].split('.')[0]
    key = furniture_category + '_' + furniture_name

    # print(f'key {key}, video_id {video_id}, frame_id {frame_id}, part_id {part_id}')
    

    return key, video_id, int(frame_id), part_id


def get_prefix(img):
    key, video_id, frame_id, part_id = get_key_and_video_id(img)
    return key + '_' + video_id + '_' + str(frame_id) + '_' + part_id




if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--username', type=str, default='') #Searching keywords
    arg_parser.add_argument('--folder_path', type=str, default='')

    
    # folder_path = arg_parser.parse_args().folder_path
    user_name = arg_parser.parse_args().username
    if user_name == '':
        user_name = '_'
    folder_paths = ['pose-estimation', 'masks']
    keys = user_name.split(',')

    output ={}
    image_files = []

    base_path = '/root/Extend-IKEA-Manual-Annotation-Interface/backend/dynamic_dataset/'
    for folder_path in folder_paths:
        abs_folder_path = os.path.join(base_path, folder_path)
        # print(f'abs_folder_path: {abs_folder_path}, os.path.exists(abs_folder_path): {os.path.exists(abs_folder_path)}, os.path.isdir(abs_folder_path): {os.path.isdir(abs_folder_path)}')
        if os.path.exists(abs_folder_path) and os.path.isdir(abs_folder_path):
            for root, dirs, files in os.walk(abs_folder_path):
                # print(f'root: {root}, dirs: {dirs}, files: {files}')
                for f in files:
                    # print(f)
                    if f.endswith(('.png', '.jpg', '.jpeg')) and (f.split('/')[-1].startswith('User')) and not f.endswith(('_temp.jpg')): #Exclude annotation done by our
                        contains_key = True
                        for key in keys:
                            if key not in f:
                                contains_key = False
                        if contains_key:
                            image_files.append(os.path.join(root.split('backend/')[-1],f))
        
    image_files = sorted(image_files, key = get_key_and_video_id)
    return_image_files = []
    counter = 0

    while counter <len(image_files):
        prefix_1 = get_prefix(image_files[counter])
        if counter + 1 >= len(image_files):
            prefix_2 = None
            prefix_3 = None
            prefix_4 = None
            prefix_5 = None
        else:
            prefix_2 = get_prefix(image_files[counter+1])
            if counter + 2 >= len(image_files):
                prefix_3 = None
                prefix_4 = None
                prefix_5 = None
            else:   

                prefix_3 = get_prefix(image_files[counter+2])
                if counter + 3 >= len(image_files):
                    prefix_4 = None
                    prefix_5 = None
                else:

                    prefix_4 = get_prefix(image_files[counter+3])
                    if counter + 4 >= len(image_files):
                        prefix_5 = None
                    else:
                        prefix_5 = get_prefix(image_files[counter+4])
        # print(f'prefix1 : {prefix_1} prefix2 : {prefix_2} prefix3 : {prefix_3} prefix4 : {prefix_4} prefix5 : {prefix_5}')
        if prefix_1 == prefix_2 and prefix_2 == prefix_3 and prefix_3 == prefix_4 and prefix_4 == prefix_5:
            # print(f'prefix_1 == prefix_2 {prefix_1 == prefix_2}, prefix_2 == prefix_3  {prefix_2 == prefix_3 }, prefix_3 == prefix_4 {prefix_3 == prefix_4}, prefix_4 == prefix_5 {prefix_4 == prefix_5}')
            return_image_files.append(image_files[counter])
            return_image_files.append(image_files[counter+1])
            return_image_files.append(image_files[counter+2])
            return_image_files.append(image_files[counter+3])
            return_image_files.append(image_files[counter+4])
            counter += 5
        elif prefix_1 == prefix_2 == prefix_3 == prefix_4:
            # print(f'prefix_1 == prefix_2 == prefix_3 == prefix_4 {prefix_1 == prefix_2 == prefix_3 == prefix_4}')
            return_image_files.append(image_files[counter])
            return_image_files.append(image_files[counter+1])
            return_image_files.append(image_files[counter+2])
            return_image_files.append(image_files[counter+3])
            return_image_files.append('dynamic_dataset/NoImage.png')
            counter += 4
        elif prefix_1 == prefix_2 == prefix_3:
            # print(f'prefix_1 == prefix_2 == prefix_3 {prefix_1 == prefix_2 == prefix_3}')
            return_image_files.append(image_files[counter])
            return_image_files.append(image_files[counter+1])
            return_image_files.append(image_files[counter+2])
            return_image_files.append('dynamic_dataset/NoImage.png')
            return_image_files.append('dynamic_dataset/NoImage.png')
            counter += 3
        elif prefix_1 == prefix_2:
            return_image_files.append(image_files[counter])
            return_image_files.append(image_files[counter+1])
            return_image_files.append('dynamic_dataset/NoImage.png')
            return_image_files.append('dynamic_dataset/NoImage.png')
            return_image_files.append('dynamic_dataset/NoImage.png')
            counter += 2
        else:
            return_image_files.append(image_files[counter])
            return_image_files.append('dynamic_dataset/NoImage.png')
            return_image_files.append('dynamic_dataset/NoImage.png')
            return_image_files.append('dynamic_dataset/NoImage.png')
            return_image_files.append('dynamic_dataset/NoImage.png')
            counter += 1


    output['image_files'] = return_image_files
    print(json.dumps(output))
