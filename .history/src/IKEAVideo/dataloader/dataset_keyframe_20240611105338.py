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

from IKEAVideo.data.assembly_video import load_annotation, load_video, load_frame, canonicalize_subassembly_parts, find_keyframes, find_subass_frames



class KeyframeDataset(torch.utils.data.Dataset):

    def __init__(self, annotation_file: str, video_dir: str, obj_dir: str, transform, load_into_mem: bool = False, verbose: bool = False, debug: bool = False, return_obj_paths: bool = True):
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
                frame_data['category'] = category
                frame_data['name'] = name
                frame_data['video_url'] = video_url
                frame_data['video_id'] = video_id
                frame_data['frame_parts'] = frame_parts
                frame_data['frame_id'] = frame_id
                frame_data['is_keyframe'] = is_keyframe
                frame_data['is_frame_after_keyframe'] = is_frame_after_keyframe
                frame_data['meta'] = frame_d
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

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        video_frames = [d for d in self.data if d['video_id'] == video_id]

        imgs = []
        is_keyframes = []
        frame_times = []
        frame_metas = []
        obj_paths = []
        frame_parts = []
        is_frame_after_keyframes = []

        for frame_data in video_frames:
            is_keyframe = frame_data['is_keyframe']
            is_frame_after_keyframe = frame_data['is_frame_after_keyframe']
            category = frame_data['category']
            name = frame_data['name']
            frame_time = frame_data['frame_time']
            video_url = frame_data['video_url']
            video_id = frame_data['video_id'].split('/watch?v=')[-1]

            if self.load_into_mem:
                img = frame_data['img']
            else:
                video = load_video(self.video_dir, category, name, video_url)
                img = load_frame(video, frame_time)

            if self.debug:
                print(f"{category} | {name} | {video_url} | {frame_time} | keyframe {is_keyframe}")
                cv2.imshow("frame", img)
                cv2.waitKey(0)
                cv2.destroyWindow("frame")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
            is_keyframes.append(is_keyframe)
            frame_times.append(frame_time)
            frame_metas.append(frame_data['meta'])
            is_frame_after_keyframes.append(is_frame_after_keyframe)

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
        # # Add the first and last frames to the list of frames to keyframes
        # is_keyframes[0] = True
        is_keyframes[-1] = True

        sample = {
            "imgs": imgs,
            "is_keyframes": is_keyframes,
            "category": category,
            "name": name,
            "video_url": video_url,
            "frame_times": frame_times,
            "frame_metas": frame_metas,
            "video_id": video_id,
            "is_frame_after_keyframe": is_frame_after_keyframes
        }

        if self.return_obj_paths:
            sample["obj_paths"] = obj_paths
            sample["parts"] = frame_parts

        return sample