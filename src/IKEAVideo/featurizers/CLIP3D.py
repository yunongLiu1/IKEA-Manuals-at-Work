import torch
import einops
from kdm.model.model.clip_classifier import get_clip_dimension_dict
import torchvision.transforms.functional as Tf
from kdm.utils.pointnet import farthest_point_sample, index_points, square_distance, query_ball_point
from kdm.model.point_transformer import sample_and_group
from tqdm import tqdm
import time


def extract_clip3d_features(rgb_img, xyzm, clip_model,
                            npoint, knn, query_ball_radius, nsample,
                            debug=False, **kwargs):
    """
    Extracts features for the CLIP3D model

    :param rgb_img: B, 3, H, W
    :param xyzm: B, H, W, 4
    :param clip_model:
    :param npoint:
    :param knn: whether to use knn or not
    :param query_ball_radius:
    :param nsample:
    :param debug:
    :return: [xyz : feats] with size B, npoint, 3+C
    """

    grid_size = get_clip_dimension_dict(clip_model)["grid_size"]

    B, H, W, _ = xyzm.shape

    assert B == 1, "batch size must be 1 because we can't batch pcs with different number of points"

    if debug:
        t1 = time.time()

    with torch.no_grad():
        patch_features = clip_model.get_patch_encodings(rgb_img)
    if debug:
        t2 = time.time()
        print("getting patch features took", t2 - t1)

    feats = einops.rearrange(patch_features, 'b (h w) c -> b c h w', h=grid_size, w=grid_size)
    feats = Tf.resize(feats, (H, W))
    feats = einops.rearrange(feats, 'b c h w -> b h w c')
    if debug:
        t3 = time.time()
        print("resizing took", t3 - t2)

    xyz = xyzm[..., :3]  # B, H, W, 3
    mask = xyzm[..., 3]  # B, H, W
    xyz = xyz[mask > 0]  # N, 3
    feats = feats[mask > 0]  # N, C

    xyz = xyz.unsqueeze(0)  # 1, N, 3
    feats = feats.unsqueeze(0)  # 1, N, C

    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)  # 1, npoint, 3
    if debug:
        t4 = time.time()
        print("fps took", t4 - t3)

    if not knn:
        group_idx = query_ball_point(query_ball_radius, nsample, xyz, new_xyz)
    else:
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        group_idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    # grouped_xyz = index_points(xyz, group_idx)  # B, npoint, nsample, 3
    feats = index_points(feats, group_idx)  # B, npoint, nsample, C

    # TODO: weighted average by distance
    # input: B, npoint, nsample, C, output: B, npoint, C
    feats = einops.reduce(feats, 'b n s c -> b n c', 'mean')
    if debug:
        t5 = time.time()
        print("grouping took", t5 - t4)

    # concat with xyz
    feats = torch.cat([new_xyz, feats], dim=-1)  # 1, npoint, 3+C

    # feats = feats.to(torch.float16)

    return feats