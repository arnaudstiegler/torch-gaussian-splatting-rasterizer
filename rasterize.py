import numpy as np
from data_reader_utils import Camera
from data_reader import read_scene
import meshio

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    '''
    This is based on the formula to get from quaternion to rotation matrix, no tricks
    '''
    w_q, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w_q, 2*x*z + 2*y*w_q],
        [2*x*y + 2*z*w_q, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w_q],
        [2*x*z - 2*y*w_q, 2*y*z + 2*x*w_q, 1 - 2*x**2 - 2*y**2]
    ])

def get_covariance_matrix_from_mesh(mesh: meshio.Mesh):
    scales = np.stack([mesh.point_data['scale_0'], mesh.point_data['scale_1'], mesh.point_data['scale_2']])
    rotations = np.stack([mesh.point_data['rot_0'], mesh.point_data['rot_1'], mesh.point_data['rot_2'], mesh.point_data['rot_3']])
    
    rotation_matrices = quaternion_to_rotation_matrix(rotations).reshape(rotations.shape[-1], 3, 3)
    scale_matrices = np.zeros((scales.shape[-1], 3, 3))
    indices = np.arange(3)
    
    scale_matrices[:, indices, indices] = scales.T
    import ipdb; ipdb.set_trace()
    return rotation_matrices @ scale_matrices @ scale_matrices.T @ rotation_matrices.T

def get_world_to_camera_matrix(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation_matrix = quaternion_to_rotation_matrix(qvec)
    projection_matrix = np.zeros((4,4))
    projection_matrix[:3, :3] = rotation_matrix
    projection_matrix[3,:3] = tvec
    projection_matrix[3,3] = 1
    return projection_matrix

def filter_view_frustum(gaussian_means: np.ndarray, gaussian_scales: np.ndarray, cam_info: Camera):
    # Apparently, we can't infer near/far clipping planes from the camera info alone

    # From the paper: "Specifically, we only keep Gaussians with a 99% confidence interval intersecting the view frustum"
    
    # cam_info.params[0] is the focal length on x-axis
    fov_x = 2*np.arctan(cam_info.width / (2*cam_info.params[0]))
    max_radius = gaussian_means[:,3]*np.tan(fov_x/2)

    import ipdb; ipdb.set_trace()

    # TODO: for now we approximate the viewing frustum filtering -> only keep gaussians for which the mean is within the radius
    # But we should take into account the spread as well
    gaussian = gaussian_means[np.absolute(gaussian_means[:, 0]) <= max_radius]


    pass

if __name__ == '__main__':
    # qvec=np.array([ 0.94440356, -0.03458228,  0.32694849,  0.00326725])
    # tvec=np.array([-1.233242  , -0.47090759,  3.94507725])

    # projection_matrix = get_world_to_camera_matrix(qvec, tvec)

    # scene, cam_info = read_scene()

    # filter_view_frustum(np.ndarray([0.0, 0.0, 0.0]), None, cam_info)

    mesh = meshio.read('data/trained_model/bonsai/point_cloud/iteration_7000/point_cloud.ply')

    gaussian_means = np.stack([mesh.point_data['nx'], mesh.point_data['ny'], mesh.point_data['nz']])
    cov_matrices = get_covariance_matrix_from_mesh(mesh)
    import ipdb; ipdb.set_trace()