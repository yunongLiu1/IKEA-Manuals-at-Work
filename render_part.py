import os
import open3d as o3d
from PIL import Image
from pyvirtualdisplay import Display
import numpy as np

# Start a virtual display for headless rendering (useful on servers without a GUI)
display = Display().start()

# Define a list of RGBA colors for rendering different parts distinctly
colors = [
    [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], 
    [1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0], 
    [0.5, 0.0, 0.0, 1.0], [0.0, 0.5, 0.0, 1.0], [0.0, 0.0, 0.5, 1.0]
]

def render_part(
    obj_path,        # Base path to the directory containing OBJ files, e.g., "./parts/"
    part_ids,        # List of part IDs (e.g., ["1,2"]) to render
    ext_mat,         # Extrinsic matrix (4x4) defining camera pose
    int_mat,         # Intrinsic matrix (3x3) defining camera properties
    img_width,       
    img_height,      
    save_path,       # Path where the rendered image will be saved
    colors=colors    
):
    """
    Renders 3D mesh parts into a 2D image using Open3D, projecting vertices based on 
    camera extrinsic and intrinsic matrices, and saves the result.
    """

    # Create a pinhole camera intrinsic object from the intrinsic matrix
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        int(img_width), int(img_height),  # Image dimensions
        int_mat[0, 0], int_mat[1, 1],     # Focal lengths (fx, fy)
        int_mat[0, 2], int_mat[1, 2]      # Principal point (cx, cy)
    )

    # Initialize an offscreen renderer for generating images without a display
    render = o3d.visualization.rendering.OffscreenRenderer(int(img_width), int(img_height))

    # Set a white background for the rendered scene
    render.scene.set_background([255.0, 255.0, 255.0, 255.0])  # RGBA

    # Define a material with default lighting for all parts
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # Default white color (overridden later)
    mtl.shader = "defaultLit"               # Use default lighting shader

    # Iterate over part IDs to load and render each mesh
    idx = 0
    for part in part_ids:
        idx += 1
        # Construct the full path to the OBJ file (e.g., "01.obj")
        curr_obj_path = os.path.join(obj_path, str(part).zfill(2) + '.obj')

        # Load the 3D mesh from the OBJ file
        mesh = o3d.io.read_triangle_mesh(curr_obj_path)
        
        # Assign a unique color to the part from the colors list (cycle if needed)
        mtl.base_color = colors[idx % len(colors)]
        
        # Add the mesh to the rendering scene with its assigned material
        render.scene.add_geometry("Part" + str(part), mesh, mtl)

    # Set up the camera using the pinhole intrinsic and extrinsic matrix
    render.setup_camera(pinhole, ext_mat)

    try:
        # Render the scene to an image (timeout not implemented here, but noted)
        img_o3d = render.render_to_image()
        print("Image rendered.")
    except Exception as e:
        # Handle any rendering errors and log them
        print(f"Exception occurred during rendering: {e}")

    # Convert the Open3D image to a PIL image and save it to the specified path
    Image.fromarray(np.array(img_o3d)).save(save_path)

    # Log the save operation and image details
    print("Rendered image saved to " + save_path)
    print("Rendered image: ", img_o3d)