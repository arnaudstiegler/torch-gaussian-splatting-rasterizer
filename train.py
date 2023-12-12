from gaussian import GaussianModel
from data_reader import read_scene

def train():
    gaussian = GaussianModel()
    scene, cam_info = read_scene()

    '''
    cam_info:
    Focal Length (f): The first two values, 1163.25472803 and 1156.28040499, represent the focal length of the camera in terms of pixels. In COLMAP, these are typically the focal lengths along the x-axis (fx) and y-axis (fy) respectively. These values determine the scale of the image projection and are crucial for depth perception in 3D reconstruction.
    Principal Point (cx, cy): The last two values, 978.5 and 545.5, denote the coordinates of the principal point. This point, typically near the center of the image sensor, is where the optical axis intersects the image plane. In COLMAP, these are given as cx and cy, which are the x and y coordinates of this principal point.
    
    scene: List[Image] where an image is:
    - qvec: array of size 4, rotation quaternion
    - tvec: array of size 3, translation vector
    - xys array of size N*2: provides the reconstructed coordinates of the points from SFM
    - point3D-ids array of size N*1: marks whether a point is visible from the image
    '''

    # Get a random image from the training set

    # Render the scene from its POV

    # Compare rendered image and ground truth
    print(cam_info)
    print(scene[1])
    import ipdb; ipdb.set_trace()

    # We want to render the image from 


if __name__ == '__main__':
    train()