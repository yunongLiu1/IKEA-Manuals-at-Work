import trimesh
import numpy as np

import IKEAVideo.utils.transformations as tra


def make_pose(trans, rot):
    """Make 4x4 matrix from (trans, rot)"""
    pose = tra.quaternion_matrix(rot)
    pose[:3, 3] = trans
    return pose


def get_camera_preset(name):

    if name == "azure_depth_nfov":
        # Setting for depth camera is pretty different from RGB
        height, width, fov = 576, 640, 75
    if name == "azure_720p":
        # This is actually the 720p RGB setting
        # Used for our color camera most of the time
        #height, width, fov = 720, 1280, 90
        height, width, fov = 720, 1280, 60
    elif name == "realsense":
        height, width, fov = 480, 640, 60
    elif name == "simple256":
        height, width, fov = 256, 256, 60
    elif name == "simple512":
        height, width, fov = 512, 512, 60
    else:
        raise RuntimeError(f'camera {name} not supported')
    return height, width, fov


def get_look_at_from_cam_pose(pose):
    look_offset = np.eye(4)
    look_offset[2, 3] = 2
    look_pose = pose.dot(look_offset)
    # at: The position to point the transform towards
    # up: The unit direction pointing upwards
    # eye: (optional) The position to place the object
    at = look_pose[:3, 3]
    up = (0, 0, 1)
    eye = pose[:3, 3]
    return at, up, eye

def get_cam_pose_from_look_at(at, eye, up=(0,0,1)):
    # https://stackoverflow.com/questions/349050/calculating-a-lookat-matrix

    at = np.array(at)
    eye = np.array(eye)
    up = np.array(up)

    # Compute rotation matrix
    zaxis = (at - eye) / np.linalg.norm(at - eye)
    xaxis = np.cross(zaxis, up) / np.linalg.norm(np.cross(zaxis, up))
    yaxis = np.cross(zaxis, xaxis)
    cam_pose = np.eye(4)
    cam_pose[:3, 0] = xaxis
    cam_pose[:3, 1] = yaxis
    cam_pose[:3, 2] = zaxis

    cam_pose[:3, 3] = eye
    return cam_pose



class GenericCameraReference(object):
    """ Class storing camera information and providing easy image capture """

    def __init__(self, proj_near=0.01, proj_far=5., proj_fov=60., img_width=640, img_height=480):

        self.proj_near = proj_near
        self.proj_far = proj_far
        self.proj_fov = proj_fov
        self.img_width = img_width
        self.img_height = img_height
        self.x_offset = self.img_width / 2.
        self.y_offset = self.img_height / 2.

        # Compute focal params
        aspect_ratio = self.img_width / self.img_height
        e = 1 / (np.tan(np.radians(self.proj_fov/2.)))
        t = self.proj_near / e
        b = -t
        r = t * aspect_ratio
        l = -r
        # pixels per meter
        alpha = self.img_width / (r-l)
        self.focal_length = self.proj_near * alpha
        self.fx = self.focal_length
        self.fy = self.focal_length

    #     self.pose = None
    #     self.inv_pose = None
    #
    # def set_pose(self, trans, rot):
    #     self.pose = make_pose(trans, rot)
    #     self.inv_pose = tra.inverse_matrix(self.pose)
    #
    # def set_pose_matrix(self, matrix):
    #     self.pose = matrix
    #     self.inv_pose = tra.inverse_matrix(matrix)
    #
    # def transform_to_world_coords(self, xyz):
    #     """ transform xyz into world coordinates """
    #     #cam_pose = tra.inverse_matrix(self.pose).dot(tra.euler_matrix(np.pi, 0, 0))
    #     #xyz = trimesh.transform_points(xyz, self.inv_pose)
    #     #xyz = trimesh.transform_points(xyz, cam_pose)
    #     #pose = tra.euler_matrix(np.pi, 0, 0) @ self.pose
    #     pose = self.pose
    #     xyz = trimesh.transform_points(xyz, pose)
    #     return xyz