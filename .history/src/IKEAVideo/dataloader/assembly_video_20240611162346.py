import os
import json
from tqdm import tqdm
import cv2
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import numpy as np
import PyPDF2
import fitz
from PIL import Image

# Decode mask
from pycocotools import mask as mask_utils


def canonicalize_subassembly_parts(parts):

    if type(parts) == list:
        parts_list = parts
        is_list = True
    elif type(parts) == str:
        parts_list = [parts]
        is_list = False
    else:
        raise ValueError(f"Invalid type for parts: {type(parts)}")

    new_parts_list = []
    for parts in parts_list:
        new_parts = parts.split(",")
        new_parts = sorted(new_parts)
        new_parts = [str(p) for p in new_parts]
        new_parts = ",".join(new_parts)
        new_parts_list.append(new_parts)

    if is_list:
        return new_parts_list
    else:
        return new_parts_list[0]


def get_video_id_from_url(video_url):
    return video_url.split('/watch?v=')[-1]


def load_video(video_dir, category, name, video_url):
    video_id = get_video_id_from_url(video_url)
    video_path = os.path.join(video_dir, category, name, video_id, f"{video_id}.mp4")
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f'Could not open video {video_path}')
    return video


def load_frame(video, time):
    try:
        video.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
        success, image = video.read()
        assert success
    except Exception as e:
        print(e)
        raise e
    return image


def generate_combinations(elements):
    all_combinations = []
    # Get all combinations of length 2, 3, 4, ..., len(elements)
    for r in range(2, len(elements) + 1):
        combinations_object = itertools.combinations(elements, r)
        combinations_list = list(combinations_object)
        all_combinations.extend(combinations_list)
    return all_combinations


def load_annotation(annotation_file, sort_frames_by_time=True, verbose=False):
    """

    :param annotation_file:
    :param sort_frames_by_time: If False, sorted by steps
    :return:
    """

    # load json from file
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    cat_name_video_to_frames = defaultdict(list)
    frame_count = 0
    for furniture_d in tqdm(data[:1], desc="Loading annotation"):
        name = furniture_d['name']
        category = furniture_d['category']
        manual_id = furniture_d['manual_id']
        furniture_ids = furniture_d['furniture_ids']
        variants = furniture_d['variants']
        pip_urls = furniture_d['pipUrls']
        image_urls = furniture_d['mainImageUrls']
        manual_urls = furniture_d['manualUrls']
        video_urls = furniture_d['videoUrls']
        # source = furniture_d['source']
        steps = furniture_d['steps']
        print(furniture_d.keys())
        # 'name', 'category', 'manual_id', 'furniture_ids', 'variants', 'pipUrls', 'mainImageUrls', 'manualUrls', 'videoUrls', 'source', 'steps'
        print(name, category, manual_id, furniture_ids, variants, pip_urls, image_urls, manual_urls, video_urls)

        for step_d in steps:
            step_id = step_d['step_id']
            manual_d = step_d['manual']
            step_videos_d = step_d['video']
            print(manual_d.keys())
            # for manual_key in manual_d.keys():
            #     print(manual_key, manual_d[manual_key])

            for step_video_d in step_videos_d:
                print(step_video_d.keys())
                video_url = step_video_d['video_id']
                step_start = step_video_d['step_start']
                step_end = step_video_d['step_end']
                step_duration = step_video_d['step_duration']
                fps = step_video_d['fps']
                substeps = step_video_d['substeps']

                for substep_d in substeps:
                    print(substep_d.keys())
                    substep_id = substep_d['substep_id']
                    substep_start = substep_d['substep_start']
                    substep_end = substep_d['substep_end']
                    substep_parts = substep_d['parts']
                

                frames = step_video_d['frames']
                video_intrinsics = step_video_d['video_intrinsics']
                print(f'Video intrinsics: {video_intrinsics}')
                all_durations = []
                for video_intrinsics in video_intrinsics.values():
                    print(f'Video intrinsics: {video_intrinsics}')
                    for duration in video_intrinsics['durations']:
                        all_durations.append(duration)

                
                print(f'All durations: {all_durations}')

                for f, frame_d in enumerate(frames):
                    frame_id = frame_d['frame_id']
                    frame_time = frame_d['frame_time']
                    frame_parts = frame_d['parts']

                    frame_masks = frame_d['mask']  # a list of masks
                    intrinsics = frame_d['intrinsics']  # a list of intrinsics
                    extrinsics = frame_d['extrinsics']  # a list of extrinsics

                    if len(frame_masks) != len(intrinsics):
                        print(f"Lengths of frame mask {len(frame_masks)} and intrinsics {len(intrinsics)} do not match for furniture {category} {name} frame {frame_id}")

                    if not(len(intrinsics) == len(extrinsics) >= len(frame_parts)):
                        print("Lengths of intrinsics {}, extrinsics {}, and parts {} do not match.".format(len(intrinsics), len(extrinsics), len(frame_parts)))
                        print("furniture: {}, name: {}, frame_id: {}".format(category, name, frame_id))

                    frame_count += 1
                    ## Add metadata to the frame
                    frames[f]['category'] = category
                    frames[f]['name'] = name
                    frames[f]['video_url'] = video_url
                    frames[f]['other_video_urls'] = video_urls 
                    frames[f]['manual_id'] = manual_id
                    frames[f]['furniture_ids'] = furniture_ids
                    frames[f]['variants'] = variants
                    frames[f]['pip_urls'] = pip_urls
                    frames[f]['image_urls'] = image_urls
                    frames[f]['manual_urls'] = manual_urls
                    frames[f]['video_urls'] = video_urls
                    
                    frames[f]['step_id'] = step_id
                    frames[f]['step_start'] = step_start
                    frames[f]['step_end'] = step_end
                    frames[f]['step_duration'] = step_duration
                    
                    frames[f]['substep_id'] = substep_id
                    frames[f]['substep_start'] = substep_start
                    frames[f]['substep_end'] = substep_end

                    frames[f]['fps'] = fps

                    frames[f]['manual'] = manual_d
                    
                    for num_of_duration, duration in enumerate(all_durations):
                        if frame_time >= duration[0] and frame_time <= duration[1]:
                            frames[f]['num_of_duration'] = num_of_duration
                            print(f'Frame {frame_id} is in duration {num_of_duration}')
                            break


                cat_name_video_to_frames[(category, name, video_url)].extend(frames)

    if sort_frames_by_time:
        for k in cat_name_video_to_frames:
            cat_name_video_to_frames[k] = sorted(cat_name_video_to_frames[k], key=lambda x: x['frame_time'])

    if verbose:
        print("=" * 50)
        print("Loading annotation from file: ", annotation_file)
        print("Total frame count: ", frame_count)
        frames_for_cat_name_video = [len(cat_name_video_to_frames[k]) for k in cat_name_video_to_frames]
        min_n_frames = min(frames_for_cat_name_video)
        max_n_frames = max(frames_for_cat_name_video)
        mean_n_frames = sum(frames_for_cat_name_video) / len(frames_for_cat_name_video)
        print(f"{len(cat_name_video_to_frames)} unique (cat, name, video) pairs")
        print(f"Number of frames per (cat, name, video): min {min_n_frames}, max {max_n_frames}, mean {mean_n_frames}")
    
    return cat_name_video_to_frames

def load_pdf_page(pdf_file, page_num):
    pdf_document = fitz.open(pdf_file)
    page = pdf_document[page_num]
    image = page.get_pixmap()


    # Assuming you have a Pixmap object named 'pixmap'
    width, height = image.width, image.height
    data = image.samples
    # Create a new PIL Image object from the Pixmap data
    image = Image.frombytes("RGB", (width, height), data)
    np_image = np.array(image)
    return np_image

def find_subass_frames(cat_name_video_to_frames, video_dir, save_dir, save_subass_frame_imgs=False, verbose=False, debug=False):

    cat_name_video_to_subass_frames = defaultdict(list)
    cat_name_video_to_before_subass_frames = defaultdict(list)
    for (cat, name, video_url) in tqdm(cat_name_video_to_frames, desc="Checking frames for each video"):

        if verbose:
            print("=" * 50)
            print("Category: ", cat)
            print("Name: ", name)
            print("Video URL: ", video_url)

        frames = cat_name_video_to_frames[(cat, name, video_url)]

        subassemblies = set()
        subass_frames = []
        before_subass_frames = []

        for fi in range(len(frames)):

            frame_d = frames[fi]

            frame_id = frame_d['frame_id']
            frame_time = frame_d['frame_time']
            frame_parts = frame_d['parts']

            # there are cases like ['0,2,3,5,6,7,1']
            frame_parts = canonicalize_subassembly_parts(frame_parts)
            frame_parts = sorted(frame_parts)

            if verbose:
                print(f"frame_id: {frame_id} | frame_time: {frame_time} | frame_parts: {frame_parts}")

            is_subass_frame = False
            for parts in frame_parts:
                if parts not in subassemblies:
                    subassemblies.add(parts)
                    is_subass_frame = True

            if is_subass_frame:
                subass_frames.append(frame_d)
                if fi - 1 >= 0:
                    before_subass_frames.append(frames[fi - 1])
                else:
                    before_subass_frames.append(None)

        cat_name_video_to_subass_frames[(cat, name, video_url)] = subass_frames
        cat_name_video_to_before_subass_frames[(cat, name, video_url)] = before_subass_frames

        # go through subass frames for visualization
        if save_subass_frame_imgs or debug:
            video = load_video(video_dir, cat, name, video_url)
            subass_frame_imgs = []
            subass_frame_names = []
            before_subass_frames_imgs = []
            before_subass_frames_names = []

        if verbose:
            print("+" * 20, "\nsubass_frames: ")
        for frame_d, before_frame_d in zip(subass_frames, before_subass_frames):

            frame_id = frame_d['frame_id']
            frame_time = frame_d['frame_time']

            frame_parts = frame_d['parts']
            # there are cases like ['0,2,3,5,6,7,1']
            frame_parts = canonicalize_subassembly_parts(frame_parts)
            frame_parts = sorted(frame_parts)

            if before_frame_d is not None:
                before_frame_id = before_frame_d['frame_id']
                before_frame_time = before_frame_d['frame_time']
                before_frame_parts = before_frame_d['parts']
                before_frame_parts = canonicalize_subassembly_parts(before_frame_parts)
                before_frame_parts = sorted(before_frame_parts)
            else:
                before_frame_id = None
                before_frame_time = None
                before_frame_parts = None

            if verbose:
                print(f"frame_id: {frame_id} | frame_time: {frame_time} | frame_parts: {frame_parts} |-----| before_frame_id: {before_frame_id} | before_frame_time: {before_frame_time}| before_frame_parts: {before_frame_parts}")

            if save_subass_frame_imgs or debug:
                # load frame image
                frame_img = load_frame(video, frame_time)
                subass_frame_imgs.append(frame_img)
                subass_frame_names.append(f"{frame_id}-{frame_time}")

                if before_frame_d is not None:
                    before_frame_img = load_frame(video, before_frame_time)
                    before_subass_frames_imgs.append(before_frame_img)
                    before_subass_frames_names.append(f"{before_frame_id}-{before_frame_time}")

                if debug:
                    # visualize frame image
                    plt.imshow(frame_img[:, :, ::-1])
                    plt.imshow(before_frame_img[:, :, ::-1])
                    plt.show()

        if save_subass_frame_imgs:

            print(f"subass_frame count: {len(subass_frame_imgs)}")
            frames_save_dir = os.path.join(save_dir, "subass_frames")
            video_id = get_video_id_from_url(video_url)
            subdir = f"{cat}-{name}-{video_id}"
            this_frames_save_dir = os.path.join(frames_save_dir, subdir)
            os.makedirs(this_frames_save_dir, exist_ok=True)
            for frame_name, frame_img in zip(subass_frame_names, subass_frame_imgs):
                frame_img_path = os.path.join(this_frames_save_dir, f"{frame_name}.jpg")
                cv2.imwrite(frame_img_path, frame_img)

            print(f"Before subass_frame count: {len(before_subass_frames_imgs)}")
            before_frames_save_dir = os.path.join(save_dir, "before_subass_frames")
            this_before_frames_save_dir = os.path.join(before_frames_save_dir, subdir)
            os.makedirs(this_before_frames_save_dir, exist_ok=True)
            for frame_name, frame_img in zip(before_subass_frames_names, before_subass_frames_imgs):
                frame_img_path = os.path.join(this_before_frames_save_dir, f"{frame_name}.jpg")
                cv2.imwrite(frame_img_path, frame_img)

        if debug:
            input("next")

    return cat_name_video_to_subass_frames, cat_name_video_to_before_subass_frames


def decode_mask(mask):
    try:
        mask = mask_utils.decode(mask)
    except Exception as e:
        print(e)
        mask = None
    return mask


def find_keyframes(cat_name_video_to_subass_frames, cat_name_video_to_before_subass_frames, video_dir, save_dir, save_keyframe_imgs=False, verbose=False, debug=False):

    cat_name_video_to_keyframes = defaultdict(list)
    for (cat, name, video_url) in tqdm(cat_name_video_to_subass_frames, desc="Checking frames for each video"):

        if verbose:
            print("=" * 50)
            print("Category: ", cat)
            print("Name: ", name)
            print("Video URL: ", video_url)

        keyframes = []
        subass_frames = cat_name_video_to_subass_frames[(cat, name, video_url)]
        before_subass_frames = cat_name_video_to_before_subass_frames[(cat, name, video_url)]

        if verbose:
            print("+" * 20, "\nsubass_frames: ")
        for frame_d, before_frame_d in zip(subass_frames, before_subass_frames):

            # e.g., frame_id: 4930 | frame_time: 164.34 | frame_parts: ['1', '3,8'] |-----| before_frame_id: 4898 | before_frame_time: 163.29| before_frame_parts: ['1', '3', '8']
            # should return ['3', '8']

            # e.g., frame_id: 7061 | frame_time: 235.39 | frame_parts: ['1,3,5,8'] |-----| before_frame_id: 7031 | before_frame_time: 234.39| before_frame_parts: ['1,3,8', '5']
            # should return ['1,3,8', '5']

            if before_frame_d is None:
                continue

            frame_parts = frame_d['parts']
            before_frame_parts = before_frame_d['parts']

            if verbose:
                print(f"frame_id: {frame_d['frame_id']} | frame_time: {frame_d['frame_time']} | frame_parts: {frame_parts} |-----| before_frame_id: {before_frame_d['frame_id']} | before_frame_time: {before_frame_d['frame_time']}| before_frame_parts: {before_frame_parts}")

            # there are cases like ['0,2,3,5,6,7,1']
            frame_parts = canonicalize_subassembly_parts(frame_parts)
            frame_parts = sorted(frame_parts)
            before_frame_parts = canonicalize_subassembly_parts(before_frame_parts)
            before_frame_parts = sorted(before_frame_parts)

            # for all combinations of before_subass in before_frame_parts
            combinations = []
            for c in generate_combinations(before_frame_parts):
                c_str = list(c)
                c_str = ",".join(c_str)
                c_str = canonicalize_subassembly_parts(c_str)
                combinations.append((c_str, c))

            assembled_subass = []
            for before_subass_str, before_subass in combinations:
                if before_subass_str in frame_parts:
                    assembled_subass.append(before_subass)

            if len(assembled_subass) > 0:
                keyframe = {}
                keyframe['next_frame_d'] = frame_d
                keyframe['keyframe_d'] = before_frame_d
                keyframe['assembled_subass'] = assembled_subass
                keyframes.append(keyframe)

        if verbose:
            print("+" * 20, "\nkeyframes: ")

        keyframe_imgs = []
        keyframe_names = []
        for keyframe in keyframes:
            next_frame_d = keyframe['next_frame_d']
            keyframe_d = keyframe['keyframe_d']
            assembled_subass = keyframe['assembled_subass']
            if verbose:
                print(f"frame_id: {next_frame_d['frame_id']} | frame_time: {next_frame_d['frame_time']} | frame_parts: {next_frame_d['parts']} |-----| keyframe_id: {keyframe_d['frame_id']} | keyframe_time: {keyframe_d['frame_time']}| keyframe_parts: {keyframe_d['parts']} | assembled_subass: {assembled_subass}")

            if save_keyframe_imgs or debug:
                video = load_video(video_dir, cat, name, video_url)
                # load frame image
                frame_img = load_frame(video, keyframe_d['frame_time'])
                if debug:
                    # visualize frame image
                    plt.imshow(frame_img[:, :, ::-1])
                    plt.show()
                keyframe_imgs.append(frame_img)
                keyframe_names.append(f"{keyframe_d['frame_id']}-{keyframe_d['frame_time']}")

        cat_name_video_to_keyframes[(cat, name, video_url)] = keyframes

        if save_keyframe_imgs:
            print(f"keyframe count: {len(keyframe_imgs)}")
            frames_save_dir = os.path.join(save_dir, "keyframes")
            video_id = get_video_id_from_url(video_url)
            subdir = f"{cat}-{name}-{video_id}"
            this_frames_save_dir = os.path.join(frames_save_dir, subdir)
            os.makedirs(this_frames_save_dir, exist_ok=True)
            for frame_img, frame_name in zip(keyframe_imgs, keyframe_names):
                frame_img_path = os.path.join(this_frames_save_dir, f"{frame_name}.jpg")
                cv2.imwrite(frame_img_path, frame_img)

        if debug:
            input("next")

    return cat_name_video_to_keyframes