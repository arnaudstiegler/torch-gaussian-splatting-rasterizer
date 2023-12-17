from os import read
import numpy as np
from data_reader_utils import Camera
from data_reader import read_scene
import meshio
from plyfile import PlyData, PlyElement
import logging
from data_reader import read_scene
import torch

logger = logging.Logger(__name__)

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

def project_to_camera_space(gaussian_means: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    # Note: @ is just a matmul
    return gaussian_means @ world_to_camera[:3, :3] + world_to_camera[-1, :3]

def get_covariance_matrix_from_mesh(mesh: PlyElement):
    scales = np.stack([mesh.elements[0]['scale_0'], mesh.elements[0]['scale_1'], mesh.elements[0]['scale_2']])
    rotations = np.stack([mesh.elements[0]['rot_0'], mesh.elements[0]['rot_1'], mesh.elements[0]['rot_2'], mesh.elements[0]['rot_3']])
    
    rotation_matrices = quaternion_to_rotation_matrix(rotations).reshape(rotations.shape[-1], 3, 3)
    scale_matrices = np.zeros((scales.shape[-1], 3, 3))
    indices = np.arange(3)
    
    scale_matrices[:, indices, indices] = scales.T
    return rotation_matrices @ scale_matrices @ np.transpose(scale_matrices, (0, 2, 1)) @ np.transpose(rotation_matrices, (0, 2, 1))

def get_world_to_camera_matrix(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation_matrix = quaternion_to_rotation_matrix(qvec)
    projection_matrix = np.zeros((4,4))
    projection_matrix[:3, :3] = rotation_matrix
    projection_matrix[:3, 3] = tvec
    projection_matrix[3,3] = 1
    return projection_matrix

def filter_view_frustum(gaussian_means: np.ndarray, cam_info: Camera):
    # From the paper: "Specifically, we only keep Gaussians with a 99% confidence interval intersecting the view frustum"
    # cam_info[1].params[0] is the focal length on x-axis
    fov_x = 2*np.arctan(cam_info[1].width / (2*cam_info[1].params[0]))
    max_radius_x = gaussian_means[:,2]*np.tan(fov_x/2)

    # cam_info[1].params[1] is the focal length on y-axis
    fov_y = 2*np.arctan(cam_info[1].height / (2*cam_info[1].params[1]))
    max_radius_y = gaussian_means[:,2]*np.tan(fov_y/2)

    # TODO: for now we approximate the viewing frustum filtering -> only keep gaussians for which the mean is within the radius
    # But we should take into account the spread as well
    fov_x_filtering = np.absolute(gaussian_means[:, 0]) <= max_radius_x
    fov_y_filtering = np.absolute(gaussian_means[:,1]) <= max_radius_y

    # TODO: should remove gaussians that are closer than the focal length
    # But there's something wrong with it, as all the gaussians have a mean
    # that's much smaller than the corresponding focal length
    # clip_plane_x_filtering = gaussian_means[:,0] < cam_info[1].params[0]
    # clip_plane_y_filtering = gaussian_means[:,1] < cam_info[1].params[1]

    return fov_x_filtering & fov_y_filtering


if __name__ == '__main__':
    
    scenes, cam_info = read_scene()
    fx, fy, cx, cy  = cam_info[1].params
    focals = np.array([fx, fy])
    width = cam_info[1].width
    height = cam_info[1].height
    
    scene = scenes[1]
    qvec = scene.qvec
    tvec = scene.tvec

    plydata = PlyData.read('data/trained_model/bonsai/point_cloud/iteration_30000/point_cloud.ply')
    gaussian_means = np.stack([plydata.elements[0]['x'], plydata.elements[0]['y'], plydata.elements[0]['z']]).T
    world_to_camera = get_world_to_camera_matrix(qvec, tvec)

    camera_space_gaussian_means = project_to_camera_space(gaussian_means, world_to_camera)

    gaussian_filtering = filter_view_frustum(camera_space_gaussian_means, cam_info)

    # Perspective project, i.e project on the screen
    # P'(x) = (P(x)/P(z))*fx
    projected_points = (camera_space_gaussian_means[:, :2] / camera_space_gaussian_means[:, -1][:, None])*focals  # The None allows to broadcast the division

    # Filter points outside of the screen (shouldn't this be done through the frustum culling???)
    # That's the viewport clipping
    screen_width_filtering = np.abs(projected_points[:,0])<= (width // 2)
    screen_height_filtering = np.abs(projected_points[:,1])<= (height // 2)

    projected_points = projected_points[gaussian_filtering & screen_width_filtering & screen_height_filtering]
    gaussian_depths = gaussian_means[:, -1][gaussian_filtering & screen_width_filtering & screen_height_filtering]

    # For the covariance, we only use the rotation/scale part of the transformation but not the translation
    # Also note that the perspective-divide does not apply in this scenario
    projected_covariances = get_covariance_matrix_from_mesh(plydata) @ world_to_camera[:3, :3]

    # # Project to NDC
    ndc_means = np.divide(projected_points, np.array([width/2, height/2])[None, :])
    # # Project to raster space
    # raster_triangle = np.floor(np.divide(ndc_triangle * np.array([width, height])[None, :], np.array([pixel_width, pixel_height])))
    # raster_triangle = raster_triangle.astype(int)
   
    import math
    num_x_tiles = math.floor(cam_info[1].width / 16)
    num_y_tiles = math.floor(cam_info[1].height / 16)

    # Here we're doing one tile per gaussian but we should do that based on overlap with their radii
    # So should probably be a [Num_gaussian, num_x_tiles, num_y_tiles]
    # tiles = np.divide(ndc_means*np.array([width//2, height//2]), np.array([num_x_tiles, num_y_tiles]))
    
    # We approximate by considering each gaussian only contributes to a single pixel
    pixels_attribution = (ndc_means*np.array([width//2, height//2]) + np.array([width//2, height//2])).astype(int)
    pixel_id = pixels_attribution[:, 0]*height + pixels_attribution[:, 1]
    sorting_tensor = torch.tensor(pixel_id + gaussian_depths)

    sorted_indices = torch.sort(sorting_tensor).indices

    # Create mask that only keeps the first of each pixel
    already_done = set()
    mask = []
    overlap = []
    for ind in sorted_indices:
        pixel = pixels_attribution[ind, 0]*height + pixels_attribution[ind, 1]
        if pixel in already_done:
            mask.append(0)
            overlap.append(1)
        else:
            overlap.append(0)
            mask.append(1)
            already_done.add(pixel)
    mask_tensor = torch.tensor(mask).bool()
    overlap_tensor = torch.tensor(overlap).bool()

    # Final depth-based filtering
    pixels_attribution = pixels_attribution[overlap_tensor]

    '''
    We then instantiate each Gaussian according to the number of tiles they overlap and assign each instance a 
    key that combines view space depth and tile ID.
    '''
    screen = np.ones((int(width / 1), int(height / 1), 3))*255

    # Size is N*3 (RGB)
    from spherical_harmonics import sh_to_rgb

    colors= np.stack([plydata.elements[0]['f_dc_0'], plydata.elements[0]['f_dc_1'], plydata.elements[0]['f_dc_2']]).T
    colors = colors[gaussian_filtering & screen_width_filtering & screen_height_filtering][overlap_tensor]
    rgb = sh_to_rgb(None, colors,0)
    rgb = np.clip(rgb, 0, 1)*255

    # For each pixel, we can combine color without taking into account opacity for now
    screen[pixels_attribution[:,0], pixels_attribution[:,1]] = rgb


    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10), dpi=100)  # Adjust the figure size and dpi as needed
    plt.imshow(screen.transpose(1,0,2))
    plt.show()
    # We then sort Gaussians based on these keys using a single fast GPU Radix sort