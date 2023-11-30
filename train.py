from gaussian import GaussianModel
from data_reader import read_scene

def train():
    gaussian = GaussianModel()
    scene, cam_info = read_scene()

    # Get a random image from the training set

    # Render the scene from its POV

    # Compare rendered image and ground truth


if __name__ == '__main__':
    train()