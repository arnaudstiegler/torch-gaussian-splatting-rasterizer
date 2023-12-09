import numpy as np
from data_reader_utils import Camera

def quaternion_to_rotation_matrix(q):
    '''
    This is based on the formula to get from quaternion to rotation matrix, no tricks
    '''
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

triangle_coordinates = np.array([[0.0, 0.0, 0.0], [0.4, 0.7, 0.2], [0.3, 0.2, 0.4]])

qvec=np.array([ 0.94440356, -0.03458228,  0.32694849,  0.00326725])
tvec=np.array([-1.233242  , -0.47090759,  3.94507725])

cam_intrinsics = {1: Camera(id=1, model='PINHOLE', width=1957, height=1091, params=np.array([1163.25472803, 1156.28040499,  978.5       ,  545.5       ]))}
focal_x, focal_y, center_x, center_y = cam_intrinsics.params

# Project the triangle on the screen

rotation_matrix = quaternion_to_rotation_matrix(qvec)
projection_matrix = np.eye(4,4)

projection_matrix[:3, :3]


intrinsic_matrix = np.array([[focal_x, 0, center_x], [0, focal_y, center_y], [0,0,1]])
# extrinsic_matrix = 

