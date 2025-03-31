import json
import os

def get_data_stat(check_folder, check_path, projection_only = False, mask_only = False):
    detailed_string = ''
    check_data = json.load(open(os.path.join(check_folder, check_path)))
    annotated_mask = 0
    annotated_pose_estimation = 0
    total = 0
    pose_counter = 0
    mask_counter = 0

    for check_furniture in check_data:
        
        for check_step in check_furniture['steps']:
            for check_video in check_step['video']:
                pose_counter = 0
                mask_counter = 0
                for check_frame in check_video['frames']:  
                    for check_part in check_frame['parts']:
                        total += 1
                        if check_frame['mask'][check_frame['parts'].index(check_part)] != {}:
                            annotated_mask += 1 
                        else:
                            mask_counter += 1
                            detailed_string += f'{check_furniture["name"]} {check_furniture["category"]} step_{check_step["step_id"]} {check_video["video_id"].split("watch?v=")[-1]} frame_{check_frame["frame_id"]} {check_part} MASK not be annotated\n'
                        if check_frame['extrinsics'][check_frame['parts'].index(check_part)] != []:
                            if check_frame['intrinsics'][check_frame['parts'].index(check_part)] != []:
                                annotated_pose_estimation += 1
                        else:
                            pose_counter += 1
        # if pose_counter != 0 and not mask_only:
        #     detailed_string += f'{check_furniture["name"]} {check_furniture["category"]} step_{check_step["step_id"]} {check_video["video_id"].split("watch?v=")[-1]} has {pose_counter} frames PROJECTION not be annotated\n'
        # if mask_counter != 0 and not projection_only:
        #     detailed_string += f'{check_furniture["name"]} {check_furniture["category"]} step_{check_step["step_id"]} {check_video["video_id"].split("watch?v=")[-1]} has {mask_counter} frames MASK not be annotated\n'
            
    return annotated_mask, annotated_pose_estimation, total, pose_counter, mask_counter, detailed_string

if __name__ == '__main__':
    check_folder = '/root/Extend-IKEA-Manual-Annotation-Interface/backend/annotator_data_0423/'

    total_annotated_mask = 0
    total_pose_estimation = 0
    total_annotated_pose_estimation = 0
    output_string = ''
    detailed_string = ''
    total_mask = 0

    detailed_string += f'----------------Details ----------------\n'
    for i in range(109):
        detailed_string += f'----------------\n Annotator {i}\n'
        check_path = f'2nd_round_data_{i}.json'
        if i > 96:
            check_path = f'verification_{i-97}.json'
        if not os.path.exists(os.path.join(check_folder, check_path)):
            continue
        annotated_mask, annotated_pose_estimation, total, pose_counter, mask_counter,curr_detailed_string = get_data_stat(check_folder, check_path)
        detailed_string += curr_detailed_string


        # total_annotated_mask += annotated_mask
        total_annotated_pose_estimation += annotated_pose_estimation
        total_pose_estimation += total
        total_annotated_mask += annotated_mask
        total_mask += total
        output_string += f'----------------\n Annotator {i}\n Number of Annotated mask: {annotated_mask} / {total}\n'


        # total_annotated_mask += annotated_mask
        total_annotated_pose_estimation += annotated_pose_estimation
        total_pose_estimation += total



    # output_string += f'----------------\n Total annotated pose estimation: {total_annotated_pose_estimation} / {total_pose_estimation}\n Total annotated masks: {total_annotated_mask} / {total_mask}\n'
    output_string += f'----------------\n Total annotated masks: {total_annotated_mask} / {total_mask}\n'
    # print(output_string)
    print(output_string + '\n' + detailed_string)
    output_string = output_string + '\n' + detailed_string

    # with open('data_stat.txt', 'w') as f:
    #     f.write(output_string)t