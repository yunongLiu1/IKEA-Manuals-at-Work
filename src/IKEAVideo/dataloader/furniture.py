import os
import cv2
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from IKEAVideo.utils.visualization import assign_colors, render_scene, get_cam_pose_from_look_at, change_scene_camera, change_textured_mesh_color, save_to_mp4, add_text_to_img


def get_category_colors(num_cls=20):
    colors = plt.cm.get_cmap('tab20', num_cls)
    return [colors(i) for i in range(num_cls)]


def visualize_color_legend(colors, labels):
    # visualize the legend with a horizontal bar chart
    fig, ax = plt.subplots()
    for i, color in enumerate(colors):
        ax.barh(i, 1, color=color, label=labels[i])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.legend()
    plt.show()


class Furniture:

    def __init__(self, asset_dir, category, name):

        self.asset_dir = asset_dir
        self.category = category
        self.name = name

        asset_path = os.path.join(asset_dir, "parts", category, name)
        self.part_to_path = {}
        for file in os.listdir(asset_path):
            if file.endswith(".obj"):
                obj_path = os.path.join(asset_path, file)
                self.part_to_path[file] = obj_path

        part_names = sorted(list(self.part_to_path.keys()))
        print(f"Loaded {len(part_names)} parts: {part_names}")

        # get an unique color for each part
        colors = get_category_colors(len(part_names))
        self.part_to_color = {part: np.array(color) * 255.0 for part, color in zip(part_names, colors)}
        print(f"part_to_color: {self.part_to_color}")

        visualize_color_legend(colors, part_names)

        self.part_to_mesh = {}
        for part, path in self.part_to_path.items():
            # important: force="mesh" to load .obj file as mesh instead of scene
            part_mesh = trimesh.load(path, force="mesh")
            if type(part_mesh.visual) == trimesh.visual.texture.TextureVisuals:
                change_textured_mesh_color(part_mesh, self.part_to_color[part])
            else:
                part_mesh.visual.face_colors = self.part_to_color[part]
            self.part_to_mesh[part] = part_mesh

    def get_furniture_mesh(self):
        return trimesh.util.concatenate(list(self.part_to_mesh.values()))

    def get_furniture_scene(self):
        return trimesh.Scene(list(self.part_to_mesh.values()))

    def save_furniture_scene(self, save_dir, debug=False):

        os.makedirs(save_dir, exist_ok=True)

        scene = self.get_furniture_scene()

        # add axis to scene
        scene.add_geometry(trimesh.creation.axis(axis_length=1, axis_radius=0.01))

        # get bounds of scene
        bounds = scene.bounds

        # visualize scene from four different angles
        # the camera positions are at the four upper corners of the bounding box. up is always [0, 1, 0]
        cam_positions = [
            [bounds[1, 0], bounds[1, 1], bounds[1, 2]],
            [bounds[1, 0], bounds[1, 1], bounds[0, 2]],
            [bounds[0, 0], bounds[1, 1], bounds[1, 2]],
            [bounds[0, 0], bounds[1, 1], bounds[0, 2]],
        ]
        cam_positions = np.array(cam_positions)
        cam_positions = cam_positions * 2  # move the camera further away from the object

        imgs = []
        for cam_position in cam_positions:
            cam_pose = get_cam_pose_from_look_at([0, 0, 0], cam_position, up=[0, 1, 0])
            img = render_scene(scene, cam_pose, resolution=[2000, 2000])
            if debug:
                scene.show()
            imgs.append(img)

        # concat imgs and save to one image
        img = np.concatenate(imgs, axis=1)
        if debug:
            # visualize img
            plt.imshow(img)
            plt.show()
        img_path = os.path.join(save_dir, f"{self.category}_{self.name}.jpg")
        cv2.imwrite(img_path, img)

    def get_subassembly_mesh(self, subassembly_parts):
        subassembly_meshes = []
        for part in subassembly_parts:
            subassembly_meshes.append(self.part_to_mesh[self.get_part_name_from_id(part)])
        print(f"subassembly_meshes: {subassembly_meshes}")
        return trimesh.util.concatenate(subassembly_meshes)

    def get_part_name_from_id(self, part_id):
        if type(part_id) == str:
            part_id = int(part_id)
        # part id should be two digit integer with leading 0 if less than 10
        assert 0 <= part_id <= 99
        return f"{part_id:02d}.obj"