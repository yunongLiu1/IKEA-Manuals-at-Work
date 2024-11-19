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
from PIL import Image, ImageDraw, ImageFont
from IKEAVideo.dataloader.assembly_video import load_annotation, load_video, load_frame, canonicalize_subassembly_parts, find_keyframes, find_subass_frames, load_pdf_page, decode_mask
import IKEAVideo.utils.transformations as tra

def hex_to_rgb(hex_color):
    """
    Convert hexadecimal color code to RGB tuple.
    :param hex_color: Hexadecimal color code (e.g. '#FFFFFF')
    :return: RGB color tuple with values from 0-255
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

colors_hex = [
        '#5A9BD5', '#FF6F61', '#77B77A', '#A67EB1', '#FF89B6', '#FFB07B',
        '#C5A3CF', '#FFA8B6', '#A3C9E0', '#FFC89B', '#E58B8B',
        '#A3B8D3', '#D4C3E8', '#66B2AA', '#E4A878', '#6882A4', '#D1AEDD', '#E8A4A6',
        '#A5DAD7', '#C6424A', '#E1D1F4', '#FFD8DC', '#F4D49B', '#8394A8'
    ]

colors = [hex_to_rgb(color) for color in colors_hex]

class KeyframeDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file: str, video_dir: str, obj_dir: str, transform, load_into_mem: bool = False, 
                 verbose: bool = False, debug: bool = False, return_obj_paths: bool = True, 
                 manual_img_dir: str = None, pdf_dir: str = None, 
                 demo_print: bool = False, demo_viz: bool = False, 
                 num_of_data: int = None):
        """
        Initialize dataset for IKEA assembly videos with annotations.
        :param annotation_file: Path to annotation JSON file
        :param video_dir: Directory containing video files  
        :param obj_dir: Directory containing 3D object files
        :param transform: Transform to apply to images
        :param load_into_mem: Whether to load frames into memory
        :param verbose: Enable verbose output 
        :param debug: Enable debug mode
        :param return_obj_paths: Return object file paths
        :param manual_img_dir: Directory containing manual images
        :param pdf_dir: Directory containing PDF manuals
        :param demo_print: Enable printing demo information
        :param demo_viz: Enable visualization
        :param num_of_data: Number of data samples to load
        """
        
        self.verbose = verbose
        self.debug = debug
        self.load_into_mem = load_into_mem
        self.transform = transform
        self.video_dir = video_dir
        self.obj_dir = obj_dir
        self.return_obj_paths = return_obj_paths
        self.manual_img_dir = manual_img_dir
        self.pdf_dir = pdf_dir
        self.demo_print = demo_print  # New parameter
        self.demo_viz = demo_viz      # New parameter
        
        self.obj_meshes_cache = {}  # cache for object meshes

        # load frames
        self.cat_name_video_to_frames = load_annotation(annotation_file, sort_frames_by_time=True, num_of_data=num_of_data)
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
        """
        Build frame data dictionary with annotations and images.
        :return: List of dictionaries containing:
            - Basic info (category, name, frame ID etc.)  
            - Meta info (variants, URLs etc.)
            - Step info
            - Images and masks 
            - Manual and PDF data
        """
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
                frame_data['frame_parts'] = frame_d['parts']
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
                frame_data['num_of_camera_changes'] = frame_d['num_of_duration']
                frame_data['iaw_metadata'] = frame_d['iaw_metadata']

                manual_image_dir  = os.path.join(self.manual_img_dir, category, name, f'step_{frame_data["step_id"]}')

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
                subassembly = None
                # handle subassembly case, return multiple meshes
                for sub_part_id in part_id.split(','):
                    mesh_key = (category, name, sub_part_id)
                    if mesh_key in self.obj_meshes_cache:
                        mesh = self.obj_meshes_cache[mesh_key]
                    else:
                        mesh_path = os.path.join(self.obj_dir, category, name, f"{sub_part_id.zfill(2)}.obj")
                        print(mesh_path)
                        mesh = trimesh.load(mesh_path, force='mesh')
                        self.obj_meshes_cache[mesh_key] = mesh
                    subassembly = mesh if subassembly is None else trimesh.util.concatenate([subassembly, mesh])
                obj_meshes.append(subassembly)
            else:
                mesh_key = (category, name, part_id)
                if mesh_key in self.obj_meshes_cache:
                    mesh = self.obj_meshes_cache[mesh_key]
                else:
                    mesh_path = os.path.join(self.obj_dir, category, name, f"{part_id.zfill(2)}.obj")
                    mesh = trimesh.load(mesh_path, force='mesh')
                    self.obj_meshes_cache[mesh_key] = mesh
                obj_meshes.append(mesh)
        return obj_meshes

    def __len__(self) -> int:
        return len(self.video_ids)



    def __getitem__(self, idx, frame_id=None) -> Dict:
        """
        Get sample from dataset.
        :param idx: Index of video to retrieve 
        :param frame_id: Specific frame ID to get. If None, returns all frames
        :return: Dictionary containing:
            - Frame images and masks
            - Manual and PDF images
            - Part meshes and transforms 
            - Annotations and metadata
        """
        video_id = self.video_ids[idx]
        video_frames = [d for d in self.data if d['video_id'] == video_id]

        start_idx = 0 if frame_id is None else max(0, frame_id)
        end_idx = len(video_frames) if frame_id is None else min(len(video_frames), frame_id + 1)


        video_frames = video_frames[start_idx:end_idx]

        for f, frame_data in enumerate(video_frames):
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

            viz_imgs = []
            
            if self.demo_print:  # Controlled by demo_print parameter
                print(f"Category: {category}")
                print(f"Name: {name}")
                print(f"Video URL: {video_url}")
                print(f'Other Video URLs for the Same Furniture: {frame_data["other_video_urls"]}')
                print(f"Frame Time: {frame_time}")
                print(f"Is Keyframe: {is_keyframe}")
                print(f"Is Frame After Keyframe: {is_frame_after_keyframe}")
                print(f'Number of Camera Changes: {frame_data["num_of_camera_changes"]}')
                print(f"Frame Parts: {frame_data['frame_parts']}")
                
                
                print(f"Furniture IDs: {frame_data['furniture_ids']}")
                print(f"Variants: {frame_data['variants']}")
                print(f"Furniture URLs: {frame_data['pip_urls']}")
                print(f"Furniture Main Image URLs: {frame_data['image_urls']}")
               
                print(f"Video URLs: {frame_data['video_urls']}")
                print(f'Manual Step ID: {frame_data["manual"]["step_id_global"]}')
                print(f"Step ID: {frame_data['step_id']}")
                print(f"Step Start: {frame_data['step_start']}")
                print(f"Step End: {frame_data['step_end']}")
                print(f"Step Duration: {frame_data['step_duration']}")
                print(f"Substep ID: {frame_data['substep_id']}")
                print(f"Substep Start: {frame_data['substep_start']}")
                print(f"Substep End: {frame_data['substep_end']}")

                
                print(f"Frame ID: {frame_data['frame_id']}")
                print(f"Other Video URLs: {frame_data['other_video_urls']}")

            if self.demo_viz: 

                print(f"Frame Image: ")
                plt.figure(figsize=(8, 6))
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()

            masks = frame_data['mask']
            decoded_masks = []
            for mask in masks:
                mask = decode_mask(mask)
                if mask is None:
                    decoded_masks.append(np.zeros_like(img))
                    continue
                decoded_masks.append(mask*255)

            # Create a copy of the original image to overlay the masks
            overlay_img = img.copy()
            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

            # Iterate over the decoded masks and overlay them on the image
            for i, mask in enumerate(decoded_masks):
                color_id = min([int(p) for p in frame_data['frame_parts'][i].split(',')])
                color = colors[color_id % len(colors)]
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))  # Resize the mask to match the image dimensions
                mask_indices = mask == 255  # Get the indices where the mask is present
                if mask_indices.sum() == 0:
                    continue
                overlay_img[mask_indices] = overlay_img[mask_indices] * 0.5 + np.array(color) * 0.5

            if self.demo_viz: 
                plt.figure(figsize=(8, 6))
                plt.imshow(overlay_img)
                plt.axis('off')
                plt.show()

                print(f"Croped Manual Image: ")
                plt.figure(figsize=(8, 6))
                plt.imshow(frame_data['manual_img'])
                plt.axis('off')
                plt.show()

            if self.demo_print:

                print(f"Manual URLs: {frame_data['manual_urls']}")
                print(f"Manual ID: {frame_data['manual_id']}")
                print(f"Manual Parts: {frame_data['manual']['parts']}")
                print(f"Manual Connections: {frame_data['manual']['connnections']}")
                print(f"PDF Page: {frame_data['manual']['page_id']}")

            # print(f"PDF Image: ")
            # ## PDF Image
            # plt.figure(figsize=(8, 6))
            # plt.imshow(frame_data['pdf_img'])
            # plt.axis('off')
            # plt.show()

            ## PDF with annotated mask

            pdf_masks = frame_data['manual']['mask']
            decoded_masks = []
            for mask in pdf_masks:
                mask = decode_mask(mask)
                if mask is None:
                    decoded_masks.append(np.zeros_like(img))
                    continue
                decoded_masks.append(mask*255)

            pdf_overlay_img = frame_data['pdf_img'].copy()

            for i, mask in enumerate(decoded_masks):
                color_id = min([int(p) for p in frame_data['manual']['parts'][i].split(',')])
                color = colors[color_id % len(colors)]
                mask = cv2.resize(mask, (pdf_overlay_img.shape[1], pdf_overlay_img.shape[0]))
                mask_indices = mask == 255
                if mask_indices.sum() == 0:
                    continue
                pdf_overlay_img[mask_indices] = pdf_overlay_img[mask_indices] * 0.5 + np.array(color) * 0.5

            if self.demo_viz: 
                print(f"PDF Image with Annotated Mask: ")
                plt.figure(figsize=(8, 6))
                plt.imshow(pdf_overlay_img)
                plt.axis('off')
                plt.show()

            if self.demo_print:
                print(f'IAW Metadata:')
                for key, value in frame_data['iaw_metadata'].items():
                    print(f'{key}: {value}')

            viz_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), overlay_img, pdf_overlay_img]
            # Create a single image with the three images side by side
            img_width = max([img.shape[1] for img in viz_imgs])
            img_height = max([img.shape[0] for img in viz_imgs])
            combined_img = np.zeros((img_height, img_width * 3, 3), dtype=np.uint8)

            viz_imgs = [np.pad(img, ((0, img_height - img.shape[0]), (0, img_width - img.shape[1]), (0, 0)), mode='constant', constant_values=255) for img in viz_imgs]

            for i, img in enumerate(viz_imgs):
                combined_img[:, i * img_width:(i + 1) * img_width] = img
            
            if self.debug: # Save the combined image
                os.makedirs(f'../debug/{video_id}', exist_ok=True)
                combined_img_path = os.path.join(f'../debug/{video_id}/', f'viz_{frame_data["frame_id"]}.jpg')
                Image.fromarray(combined_img).save(combined_img_path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            if self.transform:
                img = self.transform(img)

            video_frames[f]['img'] = img
            video_frames[f]['is_keyframe'] = is_keyframe
            video_frames[f]['is_frame_after_keyframe'] = is_frame_after_keyframe
            video_frames[f]['iaw_metadata'] = frame_data['iaw_metadata']

            meshes = self.get_obj_meshes(category, name, frame_data['frame_parts'])
            meshes_transformed = []

            for m, mesh in enumerate(meshes.copy()):
                mesh_transformed = mesh.copy()

                if self.demo_print:
                    print('mesh: ', frame_data['frame_parts'][m])
                    print('extrinsic: ', frame_data['extrinsics'][m])

                mesh_transformed.apply_transform(tra.euler_matrix(0, 0, np.pi) @ frame_data['extrinsics'][m])
                mesh_transformed.apply_transform(tra.euler_matrix(0,np.pi,0))
                meshes_transformed.append(mesh_transformed.copy())

            video_frames[f]['meshes'] = meshes_transformed

            manual_meshes = self.get_obj_meshes(category, name, frame_data['manual']['parts'])
            manual_meshes_transformed = []
            for m, mesh in enumerate(manual_meshes.copy()):
                mesh_transformed = mesh.copy()
                mesh_transformed.apply_transform(frame_data['manual']['extrinsics'][m])
                manual_meshes_transformed.append(mesh_transformed.copy())

            video_frames[f]['manual_meshes'] = manual_meshes_transformed

        sample = video_frames
        return sample