import numpy as np
from data_reader_utils import Camera

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    '''
    This is based on the formula to get from quaternion to rotation matrix, no tricks
    '''
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def get_world_to_camera_matrix(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation_matrix = quaternion_to_rotation_matrix(qvec)
    projection_matrix = np.zeros((4,4))
    projection_matrix[:3, :3] = rotation_matrix
    projection_matrix[3,:3] = tvec
    projection_matrix[3,3] = 1
    return projection_matrix


if __name__ == '__main__':
    qvec=np.array([ 0.94440356, -0.03458228,  0.32694849,  0.00326725])
    tvec=np.array([-1.233242  , -0.47090759,  3.94507725])

    projection_matrix = get_world_to_camera_matrix(qvec, tvec)
    print(projection_matrix)