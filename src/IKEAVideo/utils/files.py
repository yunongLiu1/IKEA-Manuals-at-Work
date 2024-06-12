import os


def get_checkpoint_path_from_dir(checkpoint_dir):
    checkpoint_path = None
    for file in os.listdir(checkpoint_dir):
        if "ckpt" in file:
            checkpoint_path = os.path.join(checkpoint_dir, file)
    assert checkpoint_path is not None
    return checkpoint_path

# given a path, check if the directory exists, if not, create it. If the path is a file path, create the directory
# containing the file
def create_dir_if_not_exists(path):
    if not os.path.isdir(path):
        dir = os.path.dirname(path)
    else:
        dir = path
    if not os.path.exists(dir):
        os.makedirs(dir)