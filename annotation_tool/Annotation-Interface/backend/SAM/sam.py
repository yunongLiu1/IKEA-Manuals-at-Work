import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.modeling import Sam
import json
import argparse
import cv2
import numpy as np
import random
from pycocotools import mask as mask_utils
from typing import Optional, Tuple



# class ExtendedSamPredictor(SamPredictor):
#     def __init__(self, sam_model: Sam) -> None:
#         super().__init__(sam_model=sam_model)

    # def load_image_embedding(self, embedding: np.ndarray, original_size: Tuple[int, int], input_size: Tuple[int, int]) -> None:
    #     """
    #     Loads the image embedding from a previously calculated result.
        
    #     Arguments:
    #       embedding (np.ndarray): The previously calculated image embedding.
    #       original_size (tuple(int, int)): The size of the original image, in (H, W) format.
    #       input_size (tuple(int, int)): The size of the transformed image, in (H, W) format.
    #     """
    #     # Convert numpy array to torch tensor
    #     embedding_torch = torch.from_numpy(embedding).to(self.device)
        
    #     # Set the features of the model to the loaded embedding
    #     self.features = embedding_torch

    #     # Set original and input sizes
    #     self.original_size = original_size
    #     self.input_size = input_size
        
    #     # Since an image embedding is set, we update the flag
    #     self.is_image_set = True


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)



'''
The decode_rle_to_mask function is used to decode the RLE format mask into a binary mask. 
The get_boundary_and_inner_points function is used to obtain the boundary points and inner points of the mask. 
The num_inner_points parameter is the number of points you want to randomly draw from inside the mask. 
If the number of points inside the mask is less than num_inner_points, then all interior points are returned.

>>>  mask = decode_rle_to_mask(rle, h, w)
>>> boundary_points, inner_points = get_boundary_and_inner_points(mask)

'''


def get_inner_points(mask, num_inner_points=2):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Convert contours to points, just need a few points
    boundary_points = []
    for contour in contours:
        for point in contour:
            boundary_points.append(point[0])
    
    # boundary_points = random.sample(boundary_points, 1)
    
    # Get inner points
    y_indices, x_indices = np.where(mask == 1)
    indices = list(zip(x_indices, y_indices))
    
    if len(indices) > num_inner_points:
        # Get random sample inside the mask
        inner_points = random.sample(indices, num_inner_points)

    else:
        inner_points = indices
    
    return np.array(inner_points)

def remove_mask_overlap(previous_masks, binary_mask):
    for prev_mask in previous_masks:
        if prev_mask == {}:
            continue
        else:
            rle = {
                'counts': prev_mask['counts'].encode('ascii'),
                'size': prev_mask['size'],
            }

            mask = mask_utils.decode(prev_mask)
            binary_mask[mask == 1] = 0
        return binary_mask

if __name__ == "__main__":

    HOME = os.getcwd() + "/SAM"
    # HOME = os.getcwd()
    CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)


    # Use argparse to get the input

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--json_path', type=str, help='Path to the json file', default='./user_data/yunong_segmentation_data.json' )
    args = parser.parse_args()
 
    with open(segmentation_data_path, 'r') as f:
        data = json.load(f)

    video_path = data['video_path']
    output_path = data['output_path']
    currentFrameTime = float(data['currentFrameTime'])
    input_point = np.array(eval(data['positive_points']) + eval(data['negative_points']))
    input_label = np.array([1] * len(eval(data['positive_points'])) + [0] * len(eval(data['negative_points'])))
    # image_embedding_path = data['image_embedding_path']
    print('input_point', input_point)
    print('input_label', input_label)

    print('currentFrameTime:', currentFrameTime)

    img_data = data['image'].split(",")[1]  # Remove "data:image/png;base64," prefix
    img_bytes = base64.b64decode(img_data)
    image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    original_image = image


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
    h, w = image.shape[:2]
    input_point[:, 0] *= w
    input_point[:, 1] *= h

    if data['previous_mask_exist']:
        previous_masks = data['previous_masks']
        # Convert the previous mask to negative points
        negative_points_from_previous_mask = []
        for prev_mask in previous_masks:
            if prev_mask == {}:
                continue
            else:
                rle = {
                    'counts': prev_mask['counts'].encode('ascii'),
                    'size': prev_mask['size'],
                }

                mask = mask_utils.decode(prev_mask)
                inner_points = get_inner_points(mask)
                negative_points_from_previous_mask.append(inner_points)
        negative_points_from_previous_mask = np.concatenate(negative_points_from_previous_mask, axis=0)

        # Add the negative points from the previous mask to the input points
        input_point = np.concatenate([input_point, negative_points_from_previous_mask], axis=0)
        input_label = np.concatenate([input_label, np.zeros(len(negative_points_from_previous_mask))], axis=0)
                            


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

    # Remove overlapping with previous masks
    if data['previous_mask_exist']:
        binary_mask = remove_mask_overlap(previous_masks, binary_mask)
    


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
        plt.savefig(os.path.join(os.getcwd(), '/'.join(output_path.split('/')[:-1]), output_path.split('/')[0].split('.')[0]+'_with_frame.jpg'), bbox_inches='tight', pad_inches=0)
        print("Saved image: " + os.path.join(os.getcwd(), '/'.join(output_path.split('/')[:-1]), output_path.split('/')[-1].split('.')[0]+'_with_frame.jpg'))
        plt.close()
        ########## Image with points ##########
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_points(input_point, input_label, plt.gca())
        show_mask(binary_mask, plt.gca())
        plt.gca().set_position([0, 0, 1, 1])
        plt.axis('off') # Remove the axis

        # # Save the figure
        # plt.savefig(os.path.join(os.getcwd(), output_path.split('.')[0]+'_with_points.jpg'), bbox_inches='tight', pad_inches=0)
        # print("Saved image: " + output_path.split('.')[0]+'_with_points.jpg')
        # plt.close()
        # Save the figure
        # If dir doesn't exist, create it
        output_dir = os.path.join(os.getcwd(), '/'.join(output_path.split("/")[:-1]))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("Created directory: " + output_dir)
        plt.savefig(os.path.join(os.getcwd(), output_path), bbox_inches='tight', pad_inches=0)
        print("Saved image: " + output_path)
        plt.close()

        ########## Mask only ##########

        plt.figure(figsize=(10,10))
        plt.imshow(binary_mask)
        plt.gca().set_position([0, 0, 1, 1])
        plt.axis('off') # Remove the axis

        # Save the figure
        # Make directory if it doesn't exist
        plt.savefig(os.path.join(os.getcwd(), '/'.join(output_path.split('/')[:-1]), output_path.split('/')[0].split('.')[0]+'_mask_only.jpg'), bbox_inches='tight', pad_inches=0)
        print("Saved image: " + os.path.join(os.getcwd(), '/'.join(output_path.split('/')[:-1]), output_path.split('/')[-1].split('.')[0]+'_mask_only.jpg'))
        plt.close()
    else: 
        ########## Image with points ##########
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_points(input_point, input_label, plt.gca())
        show_mask(binary_mask, plt.gca())
        plt.gca().set_position([0, 0, 1, 1])
        plt.axis('off') # Remove the axis

        # Save the figure
        # If dir doesn't exist, create it
        output_dir = os.path.join(os.getcwd(), '/'.join(output_path.split("/")[:-1]))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("Created directory: " + output_dir)
        plt.savefig(os.path.join(os.getcwd(), output_path), bbox_inches='tight', pad_inches=0)
        print("Saved image: " + output_path)
        plt.close()

    # Save the mask to a json file using RLE encoding to the input json
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    # Convert counts to ASCII string
    rle['counts'] = rle['counts'].decode('ascii')
    data['new_mask']={"size": rle['size'], "counts": rle['counts']}

    with open(segmentation_data_path, 'w') as f:
        json.dump(data, f, indent=4)
   
#     # Load the json file
#     with open(args.json_path, 'r') as f:
#         data = json.load(f)

#     image_path = data['image_path']
#     ouptut_path = data['output_path']
#     input_point = np.array(eval(data['positive_points']) + eval(data['negative_points']))
#     input_label = np.array([1] * len(eval(data['positive_points'])) + [0] * len(eval(data['negative_points'])))
#     image_embedding_path = data['image_embedding_path']
#     print(input_point)
#     print(input_label)


#     image_bgr = cv2.imread(image_path)
#     image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     original_image = image

#     sam.to(device=DEVICE)
#     image_embedding = np.load(image_embedding_path)

#     # Create a new predictor
#     new_predictor = ExtendedSamPredictor(sam)

#     # Get the original size
#     original_size = original_image.shape[:2]  # Height and width of the original image

#     # Apply the transform to get the input size
#     input_image = new_predictor.transform.apply_image(original_image)
#     input_size = input_image.shape[:2]  # Input size after transformation

#     # Load the saved embedding and sizes
#     new_predictor.load_image_embedding(image_embedding, original_size, input_size)


    
    
    
#     # # predictor = SamPredictor(sam)
#     # predictor = ExtendedSamPredictor(sam)
#     # # predictor.set_image(image)
#     # image_embedding = np.load(image_embedding_path)
#     # predictor.load_image_embedding(image_embedding, original_size, input_size)
    



#     # The points coordinates are in the range [0, 1], and should be rescaled to the image size
#     h, w = image.shape[:2]
#     input_point[:, 0] *= w
#     input_point[:, 1] *= h

#     if data['previous_mask_exist']:
#         previous_masks = data['previous_masks']


#         # Convert the previous mask to negative points
#         negative_points_from_previous_mask = []
#         for prev_mask in previous_masks['masks']:
#             mask = mask_utils.decode(prev_mask, h, w)
#             boundary_points, inner_points = get_inner_points(mask)
#             negative_points_from_previous_mask.append(inner_points)
#             negative_points_from_previous_mask.append(boundary_points)
#         negative_points_from_previous_mask = np.concatenate(negative_points_from_previous_mask, axis=0)

#         # Add the negative points from the previous mask to the input points
#         input_point = np.concatenate([input_point, negative_points_from_previous_mask], axis=0)
#         input_label = np.concatenate([input_label, np.zeros(len(negative_points_from_previous_mask))], axis=0)
                            


#     masks, scores, logits = new_predictor.predict(
#         point_coords=input_point,
#         point_labels=input_label,
#         multimask_output=True,
#     )

#     plt.figure(figsize=(10,10))
#     plt.imshow(image)
#     show_mask(masks[0], plt.gca())
#     show_points(input_point, input_label, plt.gca())
#     plt.gca().set_position([0, 0, 1, 1])

#     # Remove the axis   
#     plt.axis('off')
#     # Save the figure
#     # Make directory if it doesn't exist
#     output_dir = os.path.join(HOME, '/'.join(ouptut_path.split("/")[:-1]))

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     plt.savefig(os.path.join(HOME, ouptut_path), bbox_inches='tight', pad_inches=0)
#     plt.close()

#     # Save the mask to a json file using RLE encoding to the input json
#     rle = mask_utils.encode(np.asfortranarray(masks[0].astype(np.uint8)))
#     rle['counts'] = rle['counts'].decode('ascii')
#     data['new_mask']={"size": [masks[0].shape[0], masks[0].shape[1]], "counts": rle['counts']}
#     with open(args.json_path, 'w') as f:
#         json.dump(data, f, indent=4)


# # Example of how to use the SAM model:
# # python sam.py --image_path ../../public/sample_frame_images/T4ijaGT1eaM_cut_2.png --output_path ./output_images --positive_points "[[450, 375]]" --negative_points "[]"