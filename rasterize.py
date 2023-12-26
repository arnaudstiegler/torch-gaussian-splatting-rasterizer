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
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

logger = logging.Logger(__name__)

def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    '''
    This is based on the formula to get from quaternion to rotation matrix, no tricks
    '''

    w_q = quaternion[0, :]
    x = quaternion[1, :]
    y = quaternion[2, :]
    z = quaternion[3, :]
    return torch.stack([
        torch.stack([1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w_q, 2*x*z + 2*y*w_q]),
        torch.stack([2*x*y + 2*z*w_q, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w_q]),
        torch.stack([2*x*z - 2*y*w_q, 2*y*z + 2*x*w_q, 1 - 2*x**2 - 2*y**2])
    ]).float()

def project_to_camera_space(gaussian_means: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    # Note: @ is just a matmul
    # Here we project by; 1) applying the rotation, 2) adding the translation
    return gaussian_means @ world_to_camera[:3, :3] + world_to_camera[-1, :3]

def get_covariance_matrix_from_mesh(mesh: PlyElement):
    scales = torch.tensor(np.stack([mesh.elements[0]['scale_0'], mesh.elements[0]['scale_1'], mesh.elements[0]['scale_2']]))
    rotations = torch.tensor(np.stack([mesh.elements[0]['rot_0'], mesh.elements[0]['rot_1'], mesh.elements[0]['rot_2'], mesh.elements[0]['rot_3']]))
    
    # Because the quaternion is learned, we don't have the 
    unit_quaternions = torch.nn.functional.normalize(rotations, p=2.0, dim = 0)
    rotation_matrices = quaternion_to_rotation_matrix(unit_quaternions).reshape(rotations.shape[-1], 3, 3)
    scale_matrices = torch.zeros((scales.shape[-1], 3, 3))
    indices = torch.arange(3)
    
    scale_matrices[:, indices, indices] = scales.T
    return rotation_matrices @ scale_matrices @ torch.permute(scale_matrices, (0, 2, 1)) @ torch.permute(rotation_matrices, (0, 2, 1))

def get_world_to_camera_matrix(qvec: np.ndarray, tvec: np.ndarray) -> torch.Tensor:
    # TODO: should we normalize the quaternion first?
    rotation_matrix = quaternion_to_rotation_matrix(qvec.unsqueeze(1))
    projection_matrix = torch.zeros((4,4))
    projection_matrix[:3, :3] = torch.inverse(rotation_matrix.squeeze(-1))

    projection_matrix[3, :3] = tvec
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
    clip_plane_x_filtering = gaussian_means[:,0] < cam_info[1].params[0]
    clip_plane_y_filtering = gaussian_means[:,1] < cam_info[1].params[1]

    # return clip_plane_x_filtering
    # return torch.ones(gaussian_means.shape[0]).bool()
    return fov_x_filtering & fov_y_filtering


if __name__ == '__main__':
    
    scenes, cam_info = read_scene(path_to_scene='data/bonsai')
    fx, fy, cx, cy  = cam_info[1].params
    focals = np.array([fx, fy])
    width = cam_info[1].width
    height = cam_info[1].height
    
    scene = scenes[50]
    qvec = torch.tensor(scene.qvec)
    tvec = torch.tensor(scene.tvec)

    plydata = PlyData.read('data/trained_model/bonsai/point_cloud/iteration_30000/point_cloud.ply')
    gaussian_means = torch.tensor(np.stack([plydata.elements[0]['x'], plydata.elements[0]['y'], plydata.elements[0]['z']]).T).float()
    opacity = torch.sigmoid(torch.tensor(np.array(plydata.elements[0]['opacity'])))


    # The coordinates of the projection/camera center are given by -R^t * T, where R^t is the inverse/transpose of 
    # the 3x3 rotation matrix composed from the quaternion and T is the translation vector.
    world_to_camera = get_world_to_camera_matrix(qvec, tvec)

    
    colors = read_color_components(plydata)
    camera_space_gaussian_means = project_to_camera_space(gaussian_means, world_to_camera)

    # TODO: degree 2 buggy
    rgb = sh_to_rgb(gaussian_means, colors, degree=0)

    gaussian_filtering = filter_view_frustum(camera_space_gaussian_means, cam_info)

    # Perspective project, i.e project on the screen
    # P'(x) = (P(x)/P(z))*fx
    projected_points = (camera_space_gaussian_means[:, :2] / camera_space_gaussian_means[:, -1][:, None])*focals  # The None allows to broadcast the division

    # Filter points outside of the screen (shouldn't this be done through the frustum culling???)
    # That's the viewport clipping
    screen_width_filtering = np.abs(projected_points[:,0])<= (width // 2)
    screen_height_filtering = np.abs(projected_points[:,1])<= (height // 2)

    depth_filter = gaussian_means[:, -1] > 0.0
    full_filter = gaussian_filtering & screen_width_filtering & screen_height_filtering & depth_filter
    # TODO: bring this back
    full_filter = torch.ones(screen_width_filtering.shape).bool()

    projected_points = projected_points[full_filter]
    gaussian_depths = gaussian_means[:, -1][full_filter]
    rgb = rgb[full_filter]

    # For the covariance, we only use the rotation/scale part of the transformation but not the translation
    # Also note that the perspective-divide does not apply in this scenario
    projected_covariances = get_covariance_matrix_from_mesh(plydata).float() @ world_to_camera[:3, :3]
    projected_covariances = projected_covariances[full_filter]

    # # Project to NDC
    ndc_means = projected_points / torch.tensor([width/2, height/2])[None, :]
   

    '''
    Solving for the spherical overlap issue: right now, we have a many:1 mapping between gaussians and pixels.
    This means a given gaussian has a single corresponding pixel while a given pixel can have multiple corresponding gaussians
    though for now we do not do alpha-blending (i.e only rasterize the closest)

    Now, we want to account for the ellipsoid shape of the gaussians: the envelope will extend beyond a single pixel and it makes
    the mapping a many:many mapping which complicates the operations from a torch perspective.

    To account for it, we can create multiple instances of the same gaussian, 1 for each pixel it overlaps. That allows us to keep it
    a fully tensor-based process while allowing for variable number of overlap per pixel
    '''

    det = projected_covariances[:,0,0]*projected_covariances[:,1,1] - projected_covariances[:,1,0]*projected_covariances[:,0,1]
    trace = projected_covariances[:,0,0] + projected_covariances[:,1,1]

    # Have to clamp to 0 in case lambda is negative (no guarantee it is not)
    # To preven instabilities, we add the max
    # 0.1 is the value provided by the paper
    lambda1 = torch.clamp((trace + torch.sqrt(torch.clamp(trace**2 - 4*det, 0.1))) / 2, 0)
    lambda2 = torch.clamp((trace - torch.sqrt(torch.clamp(trace**2 - 4*det, 0.1))) / 2, 0)

    sigma1 = torch.sqrt(lambda1)
    sigma2 = torch.sqrt(lambda2)
    max_spread = torch.max(torch.stack([sigma1, sigma2], dim=-1), dim=-1).values

    screen_means = ndc_means*np.array([width//2, height//2]) + np.array([width//2, height//2])

    # Top-left, bottom right
    GAUSSIAN_SPREAD = 3
    bboxes = torch.stack([
        torch.clamp(screen_means[:,0] - GAUSSIAN_SPREAD*sigma1, 0, width),
        torch.clamp(screen_means[:,1] - GAUSSIAN_SPREAD*sigma2, 0, height),
        torch.clamp(screen_means[:,0] + GAUSSIAN_SPREAD*sigma1, 0, width),
        torch.clamp(screen_means[:,1] + GAUSSIAN_SPREAD*sigma2, 0, height)
        ], dim=-1)


    # Clamp again for gaussians that spread outside of the screen
    rounded_bboxes = torch.floor(torch.clamp(bboxes, 0)).to(int)

    # Start doing the tiling here
    bbox_center_x = rounded_bboxes[:, 0] + (rounded_bboxes[:, 2] - rounded_bboxes[:, 0]) / 2
    bbox_center_y = rounded_bboxes[:, 1] + (rounded_bboxes[:, 3] - rounded_bboxes[:, 1]) / 2

    bbox_tile_x = bbox_center_x // 16
    bbox_tile_y = bbox_center_y // 16

    tile_ids = bbox_tile_y * (height // 16) + bbox_tile_x

    # Max number of gaussians that overlap at a certain tile
    # max_buffer_needed = torch.max(torch.unique(tile_ids, return_counts=True)[1])

    # empty_tiles = torch.zeros((width // 16, height // 16, max_buffer_needed))

    # for tile_id in range(torch.max(tile_ids)):
    #     corresponding_gaussians = tile_ids == tile_id
    
    sorted_indices = torch.sort(gaussian_depths).indices

    # This is the max number of gaussians that we can aggregate from for a given pixel
    MAX_AGGREGATION = 150
    # 4 -> (r, g, b, depth)
    screen = torch.zeros((int(width), int(height), 3)).float()
    opacity_buffer = torch.ones((int(width), int(height), 3)).float()
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
        valid = torch.ones(current_pos.shape).bool()

        # if torch.all(valid == False):
        #     continue

        valid_mesh = mesh[valid, :]

        # TODO: not accounting for x/y when taking the opacity
        bbox_opacity = 1 - torch.exp(-opacity[bbox_index])
        screen[valid_mesh[:, 0], valid_mesh[:,1], :] += bbox_opacity*rgb[bbox_index]*opacity_buffer[valid_mesh[:, 0], valid_mesh[:,1]]
        
        # Update buffer and last position
        opacity_buffer[valid_mesh[:, 0], valid_mesh[:,1]] = opacity_buffer[valid_mesh[:, 0], valid_mesh[:,1]] * (1-bbox_opacity)
        last_pos[valid_mesh[:, 0], valid_mesh[:,1]] = last_pos[valid_mesh[:, 0], valid_mesh[:,1]] + 1
    

    # Create a figure
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)  # 2 rows, 1 column1, 1st subplot
    plt.imshow(screen[:, :, :3].transpose(1,0))
    plt.title('Reconstructed Image')

    # Display the second image
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
    plt.imshow(mpimg.imread(os.path.join('data/bonsai/images', scene.name)))
    plt.title('Reference Image')

    plt.show()
    