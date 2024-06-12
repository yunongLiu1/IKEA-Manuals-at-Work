import trimesh
import imageio
import numpy as np
import io
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt


def save_to_mp4(imgs, filename, fps=30, img_size=(200, 200)):
    assert ".mp4" in filename, "Filename must end in .mp4, right now it is {}".format(filename)
    print(f"Saving segment to {filename}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filename, fourcc, fps, img_size)
    for image in imgs:
        video.write(image)
    cv2.destroyAllWindows()
    video.release()


def render_scene(scene, camera_pose, resolution=[1280, 720]):
    # https://github.com/mikedh/trimesh/issues/447
    # camera_transform = np.linalg.inv(camera_transform)
    camera_transform = camera_pose @ np.diag([1, -1, -1, 1])
    scene.camera_transform = camera_transform
    data = scene.save_image(resolution=resolution, visible=True, line_settings= {'point_size': 4})
    image = np.array(Image.open(io.BytesIO(data)))
    return image


def change_scene_camera(scene, camera_pose):
    camera_transform = camera_pose @ np.diag([1, -1, -1, 1])
    scene.camera_transform = camera_transform


def save_trimesh_scene(scene, filename, visible=True):
    png = scene.save_image(resolution=[1000, 1000], visible=visible)
    with open(filename, 'wb') as f:
        f.write(png)
        f.close()


def visualize_scenes_as_gif(scenes, camera_position, look_at, fps=30, filename=None):
    frames = []
    for scene in scenes:
        camera_pose = get_cam_pose_from_look_at(look_at, camera_position)
        frame = render_scene(scene, camera_pose)
        frames.append(frame)

    # save to gif
    if filename is not None:
        imageio.mimsave(filename, frames, fps=fps)

    # visualize gif, not really works
    # # Convert frames to a GIF
    # gif_bytes = imageio.mimsave(imageio.RETURN_BYTES, frames, format='GIF', fps=fps)
    # # Display the GIF using Tkinter
    # root = tk.Tk()
    # root.title("Animated GIF")
    # image = Image.open(io.BytesIO(gif_bytes))
    # photo = ImageTk.PhotoImage(image)
    # label = tk.Label(root, image=photo)
    # label.pack()
    # root.mainloop()

    else:
        # cv2 uses BGR
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]

        cv2.namedWindow("segment")
        for img in frames:
            cv2.imshow("segment", img)
            # milliseconds: 30 fps = 33.3 ms
            waitKey = (cv2.waitKey(int(100.0 / (fps / 30))) & 0xFF)
            if waitKey == ord('q'):  # if Q pressed you could do something else with other keypress
                print("closing video and exiting")
                cv2.destroyWindow("segment")
                break
        key = cv2.waitKey()
        if key == ord("q"):
            cv2.destroyAllWindows()


def assign_colors(a, b):
    # Check if a < b
    if a >= b:
        raise ValueError("a must be less than b")

    # Create a range of values between a and b
    values = np.linspace(a, b, b-a+1)

    # Get the viridis colormap
    colormap = plt.cm.viridis

    # Normalize values to the range [0, 1] for colormap
    norm = plt.Normalize(a, b)

    # Assign colors to each value
    colors = [colormap(norm(value)) for value in values]
    # colors = np.array(colors) * 255
    # colors = colors.astype(np.uint8)
    return colors


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


def change_textured_mesh_color(mesh, rgba):
    assert type(mesh.visual) == trimesh.visual.texture.TextureVisuals, "mesh should be a textured mesh"
    texture_image = mesh.visual.material.image
    r, g, b, a = texture_image.split()
    # Set green and blue channels to 0 (leaving red channel as is)
    r = r.point(lambda i: rgba[0])
    g = g.point(lambda i: rgba[1])
    b = b.point(lambda i: rgba[2])
    a = a.point(lambda i: rgba[3])
    # Put channels back together, merge with image
    texture_image = Image.merge('RGBA', (r, g, b, a))
    mesh.visual.material.image = texture_image


def get_xyzrgb_from_rgbd(rgb_img, depth_img, camera, camera_config, camera_pose=None):
    # need to convert fov to be in radians
    xyz_img = camera.distance_map_to_point_cloud(depth_img, camera_config["fov"] / 180.0 * np.pi,
                                                 camera_config["width"], camera_config["height"])
    xyz = xyz_img.reshape(-1, 3)
    rgb = rgb_img.reshape(-1, 3)

    xyzrgb = np.concatenate([xyz, rgb], axis=1)

    if camera_pose is not None:
        xyzrgb[:, :3] = trimesh.transform_points(xyzrgb[:, :3], camera_pose)

    return xyzrgb

def add_text_to_img(img, text, position=(100, 100), font_scale=1.5, font=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0), thickness=1):
    text_img = cv2.putText(img.copy(), text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return text_img