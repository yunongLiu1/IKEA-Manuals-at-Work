import json

def get_video(json_data, category, name, step_id, video_id):
    for data in json_data:
        if data['category'] == category and data['name'] == name:
            # for video in data['steps'][int(step_id.split('_')[-1])]['video']:
            for step in data['steps']:
                if step['step_id'] == int(step_id.split('_')[-1]):
                    for video in step['video']:
                        if video['video_id'].split('watch?v=')[-1] == video_id:
                            return video
            
    return None


def get_data(data_path):
    try:
        with open(data_path) as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"File {data_path} not found.")
        return None

def is_same_parts(parts1, parts2):
    # "0,3,4" and "3,4,0" are same parts
    parts1 = parts1.split(',')
    parts2 = parts2.split(',')
    if len(parts1) != len(parts2):
        return False
    for part in parts1:
        if part not in parts2:
            return False
    return True

def contain_this_parts(part_str, target_part_str):
    part = part_str.split(',')
    target_part = target_part_str.split(',')
    for i in range(len(part)):
        if not part[i] in target_part:
            return False
    # Every part in part is in target_part
    return True

def parts_overlap(part_str1, part_str2):
    overlap = 0
    part1 = part_str1.split(',')
    part2 = part_str2.split(',')
    for p1 in part1:
        if p1 in part2:
            overlap += 1
    return overlap

            