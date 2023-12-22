import os
from data_reader_utils import read_extrinsics_binary, read_intrinsics_binary


'''
The coordinates of the projection/camera center are given by -R^t * T, 
where R^t is the inverse/transpose of the 3x3 rotation matrix composed from the quaternion 
and T is the translation vector
'''


def read_scene(path_to_scene: str):
    cameras_extrinsic_file = os.path.join(path_to_scene, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path_to_scene, "sparse/0", "cameras.bin")
    # This is the position for each image
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)

    # This is the properties of the camera itself
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    '''
    For the extrinsics, the content is:
    - image_id
    - qvec: rotation quaternion to go from world coordinate system to camera coordinate system
    - tvec: translation vector to go from world coordinate system to camera coordinate system
    - xys: array of size [N, 2] which represents the 2d coordinates of every point used for reconstruction
    - point3D_ids: array of size [N] where id == -1 if the point is not visible on the image, else it's the point id
    '''
    return cam_extrinsics, cam_intrinsics
