from collections import defaultdict
import numpy as np
from data_reader_utils import Camera
from data_reader import read_scene
from plyfile import PlyData, PlyElement
import logging
from data_reader import read_scene
import torch
from spherical_harmonics import sh_to_rgb
import tqdm
from utils import read_color_components
import matplotlib.pyplot as plt

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
    return torch.tensor(gaussian_means @ world_to_camera[:3, :3] + world_to_camera[-1, :3])

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

    
    colors = read_color_components(plydata)
    camera_space_gaussian_means = project_to_camera_space(gaussian_means, world_to_camera)

    rgb = sh_to_rgb(camera_space_gaussian_means, colors, 0)

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
    rgb = rgb[gaussian_filtering & screen_width_filtering & screen_height_filtering]

    # For the covariance, we only use the rotation/scale part of the transformation but not the translation
    # Also note that the perspective-divide does not apply in this scenario
    projected_covariances = get_covariance_matrix_from_mesh(plydata) @ world_to_camera[:3, :3]
    projected_covariances = projected_covariances[gaussian_filtering & screen_width_filtering & screen_height_filtering]

    # # Project to NDC
    ndc_means = torch.tensor(np.divide(projected_points, np.array([width/2, height/2])[None, :]))
   

    '''
    Solving for the spherical overlap issue: right now, we have a many:1 mapping between gaussians and pixels.
    This means a given gaussian has a single corresponding pixel while a given pixel can have multiple corresponding gaussians
    though for now we do not do alpha-blending (i.e only rasterize the closest)

    Now, we want to account for the ellipsoid shape of the gaussians: the envelope will extend beyond a single pixel and it makes
    the mapping a many:many mapping which complicates the operations from a torch perspective.

    To account for it, we can create multiple instances of the same gaussian, 1 for each pixel it overlaps. That allows us to keep it
    a fully tensor-based process while allowing for variable number of overlap per pixel
    '''

    det = torch.tensor(projected_covariances[:,0,0]*projected_covariances[:,1,1] - projected_covariances[:,1,0]*projected_covariances[:,0,1])
    trace = torch.tensor(projected_covariances[:,0,0] + projected_covariances[:,1,1])

    # Have to clamp to 0 in case lambda is negative (no guarantee it is not)
    lambda1 = torch.clamp((trace + torch.sqrt(trace**2 - 4*det)) / 2, 0)
    lambda2 = torch.clamp((trace - torch.sqrt(trace**2 - 4*det)) / 2, 0)

    sigma1 = torch.sqrt(lambda1)
    sigma2 = torch.sqrt(lambda2)

    screen_means = ndc_means*np.array([width//2, height//2]) + np.array([width//2, height//2])

    # Top-left, bottom right
    spread = 2
    bboxes = torch.stack([
        screen_means[:,0] - spread*sigma1, 
        screen_means[:,1] - spread*sigma2, 
        screen_means[:,0] + spread*sigma1,
        screen_means[:,1] + spread*sigma2
        ], dim=-1)

    rounded_bboxes = torch.floor(bboxes).to(int)
    
    sorted_indices = torch.sort(torch.tensor(gaussian_depths)).indices

    max_aggregation = 30
    # 4 -> (r, g, b, depth)
    screen = torch.zeros((int(width), int(height), 4, max_aggregation), dtype=float)
    last_pos = torch.zeros((int(width), int(height)))
    for bbox_index, bbox in enumerate(tqdm.tqdm(rounded_bboxes)):
        if (bbox[2] - bbox[0])*(bbox[3] - bbox[1]) == 0:
            # This means the gaussian doesn't cover any pixel
            continue
        
        # Clamping since it has to fit in the screen
        x_grid = torch.clamp(torch.arange(bbox[0], bbox[2]), 0, width-1)
        y_grid = torch.clamp(torch.arange(bbox[1], bbox[3]), 0, height-1)
        mesh_x, mesh_y = torch.meshgrid(x_grid, y_grid, indexing='ij')
        mesh = torch.stack([mesh_x, mesh_y], dim=-1).view(-1, 2)

        
        current_pos = last_pos[mesh[:,0], mesh[:,1]]
        valid = current_pos < max_aggregation

        if torch.all(valid == False):
            continue

        valid_mesh = mesh[valid, :]

        screen[valid_mesh[:, 0], valid_mesh[:,1], :, current_pos[valid].to(int)] = torch.concat([rgb[bbox_index], torch.ones((1))*gaussian_depths[bbox_index]]).to(float)
        
        last_pos[valid_mesh[:, 0], valid_mesh[:,1]] = last_pos[valid_mesh[:, 0], valid_mesh[:,1]] + 1
    

    # Very crude blending through average color across depth
    # We have to manually skip the 0.0 value
    # sum_masked = screen[:,:,:3,:].sum(dim=-1)
    # mask = torch.cat([torch.ones(screen[:,:,:3,0].shape).unsqueeze(-1), screen[:,:,:3,1:] != 0], dim=-1)
    # count_masked = mask.sum(dim=-1)

    # Calculate the mean
    # mean_masked = sum_masked / count_masked

    plt.figure(figsize=(10, 10))  # Adjust the figure size and dpi as needed
    plt.imshow(screen[:, :, :3, 0].transpose(1,0))
    '''
    Below is a color sample from the gaussians to do a color check
    '''
    # test = []
    # import random
    # for i in random.sample(list(range(rgb.shape[0])), 40):
    #     for _ in range(50):
    #         test.append(rgb[i,:].repeat(500,1))

    # plt.imshow(torch.stack(test, dim=0))
    plt.show()
    