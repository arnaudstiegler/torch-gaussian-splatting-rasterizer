from tarfile import BLOCKSIZE
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
import math
from PIL import Image

logger = logging.Logger(__name__)


Z_FAR = 100.0
Z_NEAR = 0.01
GAUSSIAN_SPREAD = 3
BLOCK_SIZE = 16
MAX_GAUSSIAN_DENSITY = 0.99
MIN_ALPHA = 1/255

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
    	# 	matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		# matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		# matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
    return gaussian_means @ world_to_camera[:3, :3] + world_to_camera[-1, :3]

def get_covariance_matrix_from_mesh(mesh: PlyElement):
    scales = torch.exp(torch.tensor(np.stack([mesh.elements[0]['scale_0'], mesh.elements[0]['scale_1'], mesh.elements[0]['scale_2']])))
    rotations = torch.tensor(np.stack([mesh.elements[0]['rot_0'], mesh.elements[0]['rot_1'], mesh.elements[0]['rot_2'], mesh.elements[0]['rot_3']]))
    
    # Learned quaternions do not guarantee a unit norm
    unit_quaternions = torch.nn.functional.normalize(rotations, p=2.0, dim = 0)
    rotation_matrices = quaternion_to_rotation_matrix(unit_quaternions).permute(2,0,1)
    scale_matrices = torch.zeros((scales.shape[-1], 3, 3))
    indices = torch.arange(3)
    scale_matrices[:, indices, indices] = scales.T

    M = rotation_matrices @ scale_matrices

    return  M @ torch.permute(M, (0, 2, 1))

def get_world_to_camera_matrix(normalized_qvec: np.ndarray, tvec: np.ndarray) -> torch.Tensor:
    rotation_matrix = quaternion_to_rotation_matrix(normalized_qvec.unsqueeze(1))
    projection_matrix = torch.zeros((4,4))

    # For a rotation matrix, transpose <-> inverse
    projection_matrix[:3, :3] = rotation_matrix.squeeze(-1)

    projection_matrix[:3, 3] = tvec
    projection_matrix[3,3] = 1
    return projection_matrix

def get_projection_matrix(fov_x, fov_y):
    tan_half_fov_x = math.tan((fov_x / 2))
    tan_half_fov_y = math.tan((fov_y / 2))

    top = tan_half_fov_y * Z_NEAR
    bottom = -top
    right = tan_half_fov_x * Z_NEAR
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * Z_NEAR / (right - left)
    P[1, 1] = 2.0 * Z_NEAR / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * Z_FAR / (Z_FAR - Z_NEAR)
    P[2, 3] = -(Z_FAR * Z_NEAR) / (Z_FAR - Z_NEAR)
    return P

def filter_view_frustum(gaussian_means: np.ndarray, fov_x: float, fov_y: float):
    # From the paper: "Specifically, we only keep Gaussians with a 99% confidence interval intersecting the view frustum"
    
    max_radius_x = gaussian_means[:,2]*np.tan(fov_x/2)
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


def compute_covering_bbox(screen_means: torch.Tensor, projected_covariances: torch.Tensor, width: float, height: float) -> torch.Tensor:
    det = projected_covariances[:,0,0]*projected_covariances[:,1,1] - projected_covariances[:,1,0]*projected_covariances[:,0,1]
    trace = projected_covariances[:,0,0] + projected_covariances[:,1,1]

    # Have to clamp to 0 in case lambda is negative (no guarantee it is not)
    # To preven instabilities, we add the max
    # 0.1 is the value provided by the paper
    lambda1 = trace/2.0 + torch.sqrt(torch.max((trace**2)/4.0 - det, torch.tensor([0.1])))
    lambda2 = trace/2.0 - torch.sqrt(torch.max((trace**2)/4.0 - det, torch.tensor([0.1])))

    max_spread = torch.ceil(GAUSSIAN_SPREAD*torch.sqrt(torch.max(torch.stack([lambda1, lambda2], dim=-1), dim=-1).values))

    bboxes = torch.stack([
        torch.clamp((screen_means[:,0] - (max_spread)) / BLOCK_SIZE, 0, width-1),
        torch.clamp((screen_means[:,1] - (max_spread)) / BLOCK_SIZE, 0, height-1),
        torch.clamp((screen_means[:,0] + (max_spread + BLOCK_SIZE - 1)) / BLOCK_SIZE, 0, width-1),
        torch.clamp((screen_means[:,1] + (max_spread + BLOCK_SIZE - 1)) / BLOCK_SIZE, 0, height-1)
        ], dim=-1)

    # bboxes = torch.stack([
    #     torch.clamp(screen_means[:,0] - max_spread, 0, width-1),
    #     torch.clamp(screen_means[:,1] - max_spread, 0, height-1),
    #     torch.clamp(screen_means[:,0] + max_spread, 0, width-1),
    #     torch.clamp(screen_means[:,1] + max_spread, 0, height-1),
    # ], dim=-1)
    # Clamp again for gaussians that spread outside of the screen
    rounded_bboxes = torch.floor(bboxes).to(int)
    return rounded_bboxes


def compute_2d_covariance(cov_matrices, camera_space_points, tan_fov_x, tan_fov_y, focals, world_to_camera):
    limx = torch.tensor([1.3 * tan_fov_x])
    limy = torch.tensor([1.3 * tan_fov_y])

    # TODO: this should be fixed upstream
    # TODO: Somehow width and height are twice what they should be also
    focal_x, focal_y = focals / 2

    txtz = camera_space_points[:,0] / camera_space_points[:,2]
    tytz = camera_space_points[:,1] / camera_space_points[:,2]
    tx = torch.min(limx, torch.max(-limx, txtz)) * camera_space_points[:,2]
    ty = torch.min(limy, torch.max(-limy, tytz)) * camera_space_points[:,2]

    J = torch.zeros((camera_space_points.shape[0], 3,3))
    J[:,0,0] = focal_x / camera_space_points[:,2]
    J[:,0,2] = -(focal_x * tx) / (camera_space_points[:,2] * camera_space_points[:,2])
    J[:,1,1] = focal_y / camera_space_points[:,2]
    J[:,1,2] = -(focal_y * ty) / (camera_space_points[:,2] * camera_space_points[:,2])

    W = world_to_camera[:-1, :-1].T

    T = torch.bmm(W.expand(J.shape[0], 3,3).transpose(2,1), J.transpose(2,1)).transpose(2,1)

    vrk = torch.zeros((camera_space_points.shape[0], 3,3))
    vrk[:,0,0] = cov_matrices[:,0,0]
    vrk[:,0,1] = cov_matrices[:,0,1]
    vrk[:,0,2] = cov_matrices[:,0,2]
    vrk[:,1,0] = cov_matrices[:,0,1]
    vrk[:,1,1] = cov_matrices[:,1,1]
    vrk[:,1,2] = cov_matrices[:,1,2]
    vrk[:,2,0] = cov_matrices[:,0,2]
    vrk[:,2,1] = cov_matrices[:,1,2]
    vrk[:,2,2] = cov_matrices[:,2,2]

    proj_cov = T @ vrk  @ T.transpose(2,1)

    # Apply low-pass filter: every Gaussian should be at least
    # one pixel wide/high. Discard 3rd row and column.
    proj_cov[:,0,0] += 0.3
    proj_cov[:,1,1] += 0.3

    return proj_cov[:, :2,:2]

def blend_gaussian(bbox_index, x_max, x_min, y_max, y_min, screen, screen_means, sigma_x, sigma_y, sigma_x_y, rgb, opacity_buffer):
    x_grid = torch.arange(x_min[bbox_index], x_max[bbox_index])
    y_grid = torch.arange(y_min[bbox_index], y_max[bbox_index])

    mesh_x, mesh_y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    mesh = torch.stack([mesh_x, mesh_y], dim=-1).view(-1, 2).to(device)

    dist_to_mean = screen_means[bbox_index] - mesh
    gaussian_density = -0.5*(sigma_x[bbox_index]*(dist_to_mean[:,0]**2) + sigma_y[bbox_index]*(dist_to_mean[:,1]**2)) - sigma_x_y[bbox_index]*dist_to_mean[:,0]*dist_to_mean[:,1]

    only_pos = gaussian_density <= 0

    alpha = torch.min(opacity[bbox_index]*torch.exp(gaussian_density), torch.tensor([MAX_GAUSSIAN_DENSITY], device=device)).float()

    # For numerical stability, we ignore 
    valid = (alpha > MIN_ALPHA) & only_pos
    valid_mesh = mesh[valid, :]

    screen[valid_mesh[:, 0], valid_mesh[:,1], :] += alpha[valid, None]*rgb[bbox_index]*opacity_buffer[valid_mesh[:, 0], valid_mesh[:,1], None]
    
    # Update buffer
    opacity_buffer[valid_mesh[:, 0], valid_mesh[:,1]] = opacity_buffer[valid_mesh[:, 0], valid_mesh[:,1]] * (1-alpha[valid])

    return screen


if __name__ == '__main__':

    torch.set_num_threads(12)
    
    scenes, cam_info = read_scene(path_to_scene='data/bonsai')
    scene = scenes[2]

    # Loading the ground truth image
    # images_{scale_fraction} i.e if 2, image has been shrunk by a factor 2
    gt_img_path = os.path.join('data/bonsai/images_2', scene.name)
    img = Image.open(gt_img_path)

    fx, fy, cx, cy  = cam_info[1].params
    focals = np.array([fx, fy])
    width, height = img.size
    
    qvec = torch.tensor(scene.qvec)
    tvec = torch.tensor(scene.tvec)

    plydata = PlyData.read('data/trained_model/bonsai/point_cloud/iteration_30000/point_cloud.ply')
    gaussian_means = torch.tensor(np.stack([plydata.elements[0]['x'], plydata.elements[0]['y'], plydata.elements[0]['z']]).T).float()
    opacity = torch.sigmoid(torch.tensor(np.array(plydata.elements[0]['opacity'])))

    # The coordinates of the projection/camera center are given by -R^t * T, where R^t is the inverse/transpose of 
    # the 3x3 rotation matrix composed from the quaternion and T is the translation vector.
    world_to_camera = get_world_to_camera_matrix(qvec, tvec).transpose(0,1)
    fov_x = 2*np.arctan(cam_info[1].width / (2*fx))
    fov_y = 2*np.arctan(cam_info[1].height / (2*fy))
    tan_fov_x = math.tan(fov_x * 0.5)
    tan_fov_y = math.tan(fov_y * 0.5)

    projection_matrix = get_projection_matrix(fov_x, fov_y).transpose(0,1)

    full_proj_transform = (world_to_camera.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    colors = read_color_components(plydata)

    camera_space_gaussian_means = project_to_camera_space(gaussian_means, world_to_camera)

    # TODO: degree 2 and 3 buggy
    # TODO: redefine cam center using the camera projection matrix?
    rgb = sh_to_rgb(gaussian_means, colors, world_to_camera, degree=3)

    # Computing only with the view matrix
    p_view = gaussian_means @ world_to_camera[:3,:] + world_to_camera[-1,:]

    # This is new, and exactly based on what the paper is doing
    points = gaussian_means @ full_proj_transform[:3,:] + full_proj_transform[-1,:]

    # Frustum culling
    frustum_culling_filter = p_view[:,2] < 0.2
    points[frustum_culling_filter] = 0.0

    p_w = 1.0 / (points[:,-1]+ 0.0000001)
    p_proj = points[:,:-1] * p_w[:, None]
    
    # TODO: add depth filtering back
    screen_width_filtering = torch.abs(points[:,0]) < points[:,-1]
    screen_height_filtering = torch.abs(points[:,1]) < points[:,-1]

    new_filter = screen_width_filtering & screen_height_filtering

    projected_covariances = compute_2d_covariance(get_covariance_matrix_from_mesh(plydata).float(), camera_space_gaussian_means, tan_fov_x, tan_fov_y, focals, world_to_camera)
    projected_covariances[frustum_culling_filter] = 0.0

    screen_means = ((p_proj[:,:2] + 1.0) * torch.tensor([width, height]) - 1.0)/2

    rounded_bboxes = compute_covering_bbox(screen_means, projected_covariances, width, height)

    # 4 -> (r, g, b, depth)
    screen = torch.zeros((int(width), int(height), 3)).float()
    opacity_buffer = torch.ones((int(width), int(height))).float()

    # TODO: might wanna add back last_pos to limit the number of gaussians that get backpropagated
    last_pos = torch.zeros((int(width), int(height)))

    # Technically, could be a problem if equal to 0
    # Doesn't happen in practice
    det_inv = 1 / (projected_covariances[:,0,0]*projected_covariances[:,1,1] - projected_covariances[:,1,0]*projected_covariances[:,0,1])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rounded_bboxes = rounded_bboxes.to(device)
    screen_means = screen_means.to(device)
    projected_covariances = projected_covariances.to(device)
    det_inv = det_inv.to(device)
    screen = screen.to(device)
    rgb = rgb.to(device)
    opacity_buffer = opacity_buffer.to(device)

    sigma_x = projected_covariances[:,1,1] * det_inv[:]
    sigma_y = -projected_covariances[:,0,1] * det_inv[:]
    sigma_x_y = projected_covariances[:,0,0] * det_inv[:]

    x_min = torch.clamp(rounded_bboxes[:, 0]*BLOCK_SIZE, 0, width-1)
    y_min = torch.clamp(rounded_bboxes[:, 1]*BLOCK_SIZE, 0, height-1)
    x_max = torch.clamp(rounded_bboxes[:, 2]*BLOCK_SIZE, 0, width-1)
    y_max = torch.clamp(rounded_bboxes[:, 3]*BLOCK_SIZE, 0, height-1)

    depths = p_view[:,2]
    sorted_gaussians = torch.sort(depths).indices

    bbox_area = (x_max - x_min)*(y_max - y_min)

    for gaussian_index in tqdm.tqdm(sorted_gaussians):
        if bbox_area[gaussian_index] == 0:
            # This means the gaussian doesn't cover any pixel
            continue
        screen = blend_gaussian(gaussian_index, x_max, x_min, y_max, y_min, screen, screen_means, sigma_x, sigma_y, sigma_x_y, rgb, opacity_buffer)
 
    # Create a figure
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)  # 2 rows, 1 column1, 1st subplot
    plt.imshow(screen[:, :, :3].transpose(1,0).cpu())
    plt.title('Reconstructed Image')

    # Display the second image
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
    plt.imshow(mpimg.imread(gt_img_path))
    plt.title('Reference Image')

    plt.show()
    