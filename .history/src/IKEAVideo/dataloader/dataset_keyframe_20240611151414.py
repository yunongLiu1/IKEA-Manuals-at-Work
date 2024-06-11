import os
import sys
import json
import random

from typing import Tuple, List, Dict
import trimesh
import torch
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from IKEAVideo.dataloader.assembly_video import load_annotation, load_video, load_frame, canonicalize_subassembly_parts, find_keyframes, find_subass_frames, load_pdf_page, decode_mask


colors = [
    '#5A9BD5', '#FF6F61', '#E5C07B', '#77B77A', '#A67EB1', '#FF89B6', '#FFB07B',
    '#C5A3CF', '#FFA8B6', '#A3C9E0', '#FFC89B', '#E58B8B',
    '#A3B8D3', '#D4C3E8', '#66B2AA', '#E4A878', '#6882A4', '#D1AEDD', '#E8A4A6',
    '#A5DAD7', '#C6424A', '#E1D1F4', '#FFD8DC', '#F4D49B', '#8394A8'
]

class KeyframeDataset(torch.utils.data.Dataset):

    def __init__(self, annotation_file: str, video_dir: str, obj_dir: str, transform, load_into_mem: bool = False, verbose: bool = False, debug: bool = False, return_obj_paths: bool = True, manual_img_dir: str = None, pdf_dir: str = None):
        """
        Dataset class for loading keyframes in IKEA videos.

        Definitions:
        - subassembly: a specific part that is denoted by its id (e.g., '3') or a subassembly that consists of multiple individual parts (e.g., '3,7')
        - subassembly_frame: a frame where a new part / subassembly appears. This could happen because 1) parts /subassemblies are combined to form a new subassembly, or 2) a new part / subassembly appears.
        - keyframe: a frame where a new subassembly is formed.

        Each sample should contain the following:
        - a video consists of frames
        - an image of the query frame
        - binary label indicating whether the query frame is a keyframe
        - object meshes for the parts in the query frame

        :param annotation_file: path to the annotation file
        :param video_dir: directory containing the video files
        :param obj_dir: directory containing the object mesh files
        :param transform: image transform to be applied
        :param load_into_mem: whether to load frames into memory
        :param verbose: whether to print verbose information
        :param debug: whether to enable debug mode
        """

        self.verbose = verbose
        self.debug = debug
        self.load_into_mem = load_into_mem

        # image transform
        self.transform = transform
        self.video_dir = video_dir
        self.obj_dir = obj_dir
        self.return_obj_paths = return_obj_paths
        self.manual_img_dir = manual_img_dir
        self.pdf_dir = pdf_dir

        self.obj_meshes_cache = {}  # cache for object meshes

        # load frames
        self.cat_name_video_to_frames = load_annotation(annotation_file, sort_frames_by_time=True)
        # find frames where a new subassembly is added and the frames before that
        self.cat_name_video_to_subass_frames, cat_name_video_to_before_subass_frames = find_subass_frames(
            self.cat_name_video_to_frames, video_dir, save_dir=None, save_subass_frame_imgs=False, verbose=verbose, debug=False)
        # find keyframes for each subassembly, i.e., the frame where the subassembly is formed
        self.cat_name_video_to_keyframes = find_keyframes(self.cat_name_video_to_subass_frames,
                                                     cat_name_video_to_before_subass_frames,
                                                    video_dir,
                                                     save_dir=None,
                                                     save_keyframe_imgs=False, debug=False)
        self.data = self.build_frame_data()
        self.video_ids = sorted(list(set([d['video_id'] for d in self.data])))

        num_keyframes = sum([1 for frame_data in self.data if frame_data['is_keyframe']])
        if self.verbose:
            print(f"Number of samples: {len(self.data)}")
            print(f"Number of keyframes: {num_keyframes}")
            print(f"Number of videos: {len(self.video_ids)}")

    def build_frame_data(self) -> List[Dict]:
        all_frame_data = []
        for category, name, video_url in tqdm(self.cat_name_video_to_frames, desc="Building frame data"):
            keyframes = self.cat_name_video_to_keyframes[(category, name, video_url)]
            keyframe_ids = [frame_d['keyframe_d']['frame_id'] for frame_d in keyframes]
            key_frame_next_frame_ids = [frame_d['next_frame_d']['frame_id'] for frame_d in keyframes]
            frames = self.cat_name_video_to_frames[(category, name, video_url)]
            video = load_video(self.video_dir, category, name, video_url)
            video_id = f"{category}_{name}_{video_url}"
            for frame_d in frames:
                frame_id = frame_d['frame_id']
                frame_time = frame_d['frame_time']
                frame_parts = frame_d['parts']
                # there are cases like ['0,2,3,5,6,7,1']
                frame_parts = canonicalize_subassembly_parts(frame_parts)
                frame_parts = sorted(frame_parts)
                if frame_id in keyframe_ids:
                    is_keyframe = True
                else:
                    is_keyframe = False
                
                if frame_id in key_frame_next_frame_ids:
                    is_frame_after_keyframe = True
                else:
                    is_frame_after_keyframe = False

                frame_data = {}
                ### Basic Information
                frame_data['category'] = category
                frame_data['name'] = name
                frame_data['video_url'] = video_url
                frame_data['video_id'] = video_id
                frame_data['frame_parts'] = frame_parts
                frame_data['frame_id'] = frame_id
                frame_data['is_keyframe'] = is_keyframe
                frame_data['is_frame_after_keyframe'] = is_frame_after_keyframe

                ### Meta Information
                frame_data['other_video_urls'] = frame_d['other_video_urls']
                frame_data['manual_id'] = frame_d['manual_id']
                frame_data['furniture_ids'] = frame_d['furniture_ids']
                frame_data['variants'] = frame_d['variants']
                frame_data['pip_urls'] = frame_d['pip_urls']
                frame_data['image_urls'] = frame_d['image_urls']
                frame_data['manual_urls'] = frame_d['manual_urls']
                frame_data['video_urls'] = frame_d['video_urls']

                ## Step Information
                frame_data['step_id'] = frame_d['step_id']
                frame_data['step_start'] = frame_d['step_start']
                frame_data['step_end'] = frame_d['step_end']
                frame_data['step_duration'] = frame_d['step_duration']

                ## Substep Information
                frame_data['substep_id'] = frame_d['substep_id']
                frame_data['substep_start'] = frame_d['substep_start']
                frame_data['substep_end'] = frame_d['substep_end']
                frame_data['manual'] = frame_d['manual']
                frame_data['mask'] = frame_d['mask']
                frame_data['extrinsics'] = frame_d['extrinsics']
                frame_data['intrinsics'] = frame_d['intrinsics']

                manual_image_dir  = os.path.join(self.manual_img_dir, category, name, f'step_{frame_data["step_id"]}')
                print('manual_image_dir : ', manual_image_dir)
                if os.path.exists(manual_image_dir):
                    if len(os.listdir(manual_image_dir)) == 1:
                        #print ("manual image exists")
                        manualImagePath = os.path.join(manual_image_dir, os.listdir(manual_image_dir)[0])
                    elif len(os.listdir(manual_image_dir)) > 1:
                        print ("manual image exists, but more than one")
                
                frame_data['manual_img'] = cv2.imread(manualImagePath)

                manual_pdf_dir = os.path.join(self.pdf_dir, category, name, '0.pdf')
                page_num = frame_data['manual']['page_id'] -1
                pdf_img = load_pdf_page(manual_pdf_dir, page_num)
                frame_data['pdf_img'] = pdf_img


                if self.load_into_mem:
                    frame_img = load_frame(video, frame_time)
                    frame_data['img'] = frame_img
                else:
                    frame_data['frame_time'] = frame_time
                all_frame_data.append(frame_data)
        return all_frame_data

    def get_obj_meshes(self, category: str, name: str, part_ids: List[str]) -> List[trimesh.Trimesh]:
        """
        Get the object meshes for the given category, name, and part IDs.
        If the meshes are already loaded in the cache, return them from the cache.
        Otherwise, load the meshes and store them in the cache for future use.

        :param category: category of the object
        :param name: name of the object
        :param part_ids: list of part IDs
        :return: list of object meshes
        """
        obj_meshes = []
        for part_id in part_ids:
            if ',' in part_id:
                # handle subassembly case, return multiple meshes
                for sub_part_id in part_id.split(','):
                    mesh_key = (category, name, sub_part_id)
                    if mesh_key in self.obj_meshes_cache:
                        mesh = self.obj_meshes_cache[mesh_key]
                    else:
                        mesh_path = os.path.join(self.obj_dir, category, name, f"{sub_part_id.zfill(2)}.obj")
                        print(mesh_path)
                        mesh = trimesh.load(mesh_path)
                        self.obj_meshes_cache[mesh_key] = mesh
                    obj_meshes.append(mesh)
            else:
                mesh_key = (category, name, part_id)
                if mesh_key in self.obj_meshes_cache:
                    mesh = self.obj_meshes_cache[mesh_key]
                else:
                    mesh_path = os.path.join(self.obj_dir, category, name, f"{part_id.zfill(2)}.obj")
                    mesh = trimesh.load(mesh_path)
                    self.obj_meshes_cache[mesh_key] = mesh
                obj_meshes.append(mesh)
        return obj_meshes

    def __len__(self) -> int:
        return len(self.video_ids)



    def __getitem__(self, idx, frame_id=None) -> Dict:
        video_id = self.video_ids[idx]
        video_frames = [d for d in self.data if d['video_id'] == video_id]

        start_idx = 0 if frame_id is None else max(0, frame_id)
        end_idx = len(video_frames) if frame_id is None else min(len(video_frames), frame_id + 1)

        imgs = []
        is_keyframes = []
        frame_times = []
        frame_metas = []
        obj_paths = []
        frame_parts = []
        is_frame_after_keyframes = []
        video_frames = video_frames[start_idx:end_idx]

        for f, frame_data in enumerate(video_frames):
            is_keyframe = frame_data['is_keyframe']
            is_frame_after_keyframe = frame_data['is_frame_after_keyframe']
            category = frame_data['category']
            name = frame_data['name']
            frame_time = frame_data['frame_time']
            video_url = frame_data['video_url']
            video_id = frame_data['video_id'].split('/watch?v=')[-1]
            # step_num = frame_data['step_num']

            if self.load_into_mem:
                img = frame_data['img']
            else:
                video = load_video(self.video_dir, category, name, video_url)
                img = load_frame(video, frame_time)

            if self.debug:
                print(f"Category: {category}")
                print(f"Name: {name}")
                print(f"Video URL: {video_url}")
                print(f"Frame Time: {frame_time}")
                print(f"Is Keyframe: {is_keyframe}")
                print(f"Is Frame After Keyframe: {is_frame_after_keyframe}")
                print(f"Frame Parts: {frame_data['frame_parts']}")
                
                
                print(f"Furniture IDs: {frame_data['furniture_ids']}")
                print(f"Variants: {frame_data['variants']}")
                print(f"PIP URLs: {frame_data['pip_urls']}")
                print(f"Image URLs: {frame_data['image_urls']}")
               
                print(f"Video URLs: {frame_data['video_urls']}")
                print(f"Step ID: {frame_data['step_id']}")
                print(f"Step Start: {frame_data['step_start']}")
                print(f"Step End: {frame_data['step_end']}")
                print(f"Step Duration: {frame_data['step_duration']}")
                print(f"Substep ID: {frame_data['substep_id']}")
                print(f"Substep Start: {frame_data['substep_start']}")
                print(f"Substep End: {frame_data['substep_end']}")
                
                print(f"Frame ID: {frame_data['frame_id']}")
                print(f"Other Video URLs: {frame_data['other_video_urls']}")
                

                print(f"Frame Image: ")
                # Display the frame using Matplotlib
                plt.figure(figsize=(8, 6))
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()

                ## Display mask on the frame
                masks = frame_data['mask']
                
                decoded_masks = []
                for mask in masks:
                    mask = decode_mask(mask)
                    if mask is None:
                        decoded_masks.append(np.zeros_like(img))
                        continue
                    decoded_masks.append(mask)
                    print(mask.shape)
                
                ## Overlay the mask on the frame with the color
                for i, mask in enumerate(decoded_masks):
                    color = colors[i]
                    img = cv2.addWeighted(img, 1, mask, 0.5, 0)
                plt.figure(figsize=(8, 6))
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
                

                print(f"Manual URLs: {frame_data['manual_urls']}")
                print(f"Manual ID: {frame_data['manual_id']}")
                print(f"Manual: {frame_data['manual']}")
                print(f"Croped Manual Image: ")
                print
                ## Manual Image
                plt.figure(figsize=(8, 6))
                plt.imshow(frame_data['manual_img'])
                plt.axis('off')
                plt.show()
                

                print(f"PDF Page: {frame_data['manual']['page_id']}")
                
                print(f"PDF Image: ")
                ## PDF Image
                plt.figure(figsize=(8, 6))
                plt.imshow(frame_data['pdf_img'])
                plt.axis('off')
                plt.show()




            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            if self.transform:
                img = self.transform(img)


            video_frames[f]['img'] = img
            video_frames[f]['is_keyframe'] = is_keyframe
            video_frames[f]['is_frame_after_keyframe'] = is_frame_after_keyframe
            
            # imgs.append(img)
            # is_keyframes.append(is_keyframe)
            # frame_times.append(frame_time)
            # frame_metas.append(frame_data['meta'])
            # is_frame_after_keyframes.append(is_frame_after_keyframe)

            if self.return_obj_paths:
                frame_obj_paths = []
                for part_id in frame_data['frame_parts']:
                    if ',' in part_id:
                        part_obj_paths = []
                        for sub_part_id in part_id.split(','):
                            obj_path = os.path.join(self.obj_dir, category, name, f"{sub_part_id.zfill(2)}.obj")
                            part_obj_paths.append(obj_path)
                        frame_obj_paths.append(part_obj_paths)
                    else:
                        obj_path = os.path.join(self.obj_dir, category, name, f"{part_id.zfill(2)}.obj")
                        frame_obj_paths.append([obj_path])
                obj_paths.append(frame_obj_paths)
                frame_parts.append(frame_data['frame_parts'])

        # Add the first and last frames to the list of frames to keyframes
        # is_keyframes[-1] = True

        # sample = {
        #     "imgs": imgs,
        #     "is_keyframes": is_keyframes,
        #     "category": category,
        #     "name": name,
        #     "video_url": video_url,
        #     "frame_times": frame_times,
        #     "frame_metas": frame_metas,
        #     "video_id": video_id,
        #     "is_frame_after_keyframe": is_frame_after_keyframes
        # }
        sample = video_frames

        # if self.return_obj_paths:
        #     sample["obj_paths"] = obj_paths
        #     sample["parts"] = frame_parts

        return sample
         