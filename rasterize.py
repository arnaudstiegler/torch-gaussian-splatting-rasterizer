import logging
import math
import os
import subprocess
from typing import Optional, Tuple

import click
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from PIL import Image
from plyfile import PlyData, PlyElement

from spherical_harmonics import sh_to_rgb
from utils import read_color_components, read_scene

logger = logging.Logger(__name__)


# Z_FAR and Z_NEAR are computer graphics distance which mark the near sight and far sight limit
# i.e you cannot something closer than Z_NEAR or farther than Z_FAR
Z_FAR = 100.0
Z_NEAR = 0.01
# This is a scaling factor set in the original implementation. Not sure whether there's an actual reason to use this particular value
GAUSSIAN_SPREAD = 3
# Size of a processing block for a CUDA kernel (i.e a block processes a 16*16 set of pixels)
BLOCK_SIZE = 16
# Set maximum density to prevent overflow issues
MAX_GAUSSIAN_DENSITY = 0.99
# Minimum alpha before stopping to blend new gaussians (they will not be visible in any case)
MIN_ALPHA = 1 / 255


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Quaternion representation for rotation matrices is common, as it is a more efficient representation.
    The rotation matrix can be recovered with the formula below, no tricks just calculus.
    """
    w_q = quaternion[0, :]
    x = quaternion[1, :]
    y = quaternion[2, :]
    z = quaternion[3, :]
    return torch.stack(
        [
            torch.stack([1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w_q, 2 * x * z + 2 * y * w_q,]),
            torch.stack([2 * x * y + 2 * z * w_q, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w_q,]),
            torch.stack([2 * x * z - 2 * y * w_q, 2 * y * z + 2 * x * w_q, 1 - 2 * x ** 2 - 2 * y ** 2,]),
        ]
    ).float()


def get_world_to_camera_matrix(normalized_qvec: np.ndarray, tvec: np.ndarray) -> torch.Tensor:
    """
    We create the matrix that transforms coordinates from World space (i.e agnostic to your POV, or simply the reference coordinate space)
    to the Camera space which is the system of coordinates based on the camera POV

    Given a rotation matrix R and a translation vector T, the matrix is defined as:
    M = [[R11, R12, R13, -T_x],
         [R21, R22, R23, -T_y],
         [R31, R32, R33, -T_z],
         [  0,   0,   0,    1]]
    """
    rotation_matrix = quaternion_to_rotation_matrix(normalized_qvec.unsqueeze(1))
    projection_matrix = torch.zeros((4, 4))

    projection_matrix[:3, :3] = rotation_matrix.squeeze(-1)

    projection_matrix[:3, 3] = tvec
    projection_matrix[3, 3] = 1
    return projection_matrix


def project_to_camera_space(gaussian_means: torch.Tensor, world_to_camera: torch.Tensor) -> torch.Tensor:
    """
    This is just 3D geometry, the new coordinates are obtained by:
    - applying the rotation (i.e multiplying by the rotation matrix)
    - add the translation
    """
    return gaussian_means @ world_to_camera[:3, :3] + world_to_camera[-1, :3]


def get_covariance_matrix_from_mesh(mesh: PlyElement) -> torch.Tensor:
    """
    Covariance matrices are trained parameters. They will define the spread of each gaussian in the 3D space, and therefore
    the area of pixels covered by the gaussians once projected in 2D

    See paper: they formulate gaussian covariances using a scale matrix S and a rotation matrix R
    such that Cov = R * S * S_t * R_t
    """
    scales = torch.exp(
        torch.tensor(np.stack([mesh.elements[0]["scale_0"], mesh.elements[0]["scale_1"], mesh.elements[0]["scale_2"],]))
    )
    rotations = torch.tensor(
        np.stack(
            [
                mesh.elements[0]["rot_0"],
                mesh.elements[0]["rot_1"],
                mesh.elements[0]["rot_2"],
                mesh.elements[0]["rot_3"],
            ]
        )
    )

    # Learned quaternions do not guarantee a unit norm, therefore we have to normalize them
    unit_quaternions = torch.nn.functional.normalize(rotations, p=2.0, dim=0)
    rotation_matrices = quaternion_to_rotation_matrix(unit_quaternions).permute(2, 0, 1)
    scale_matrices = torch.zeros((scales.shape[-1], 3, 3))
    indices = torch.arange(3)
    scale_matrices[:, indices, indices] = scales.T

    M = rotation_matrices @ scale_matrices

    return M @ torch.permute(M, (0, 2, 1))


def get_projection_matrix(fov_x: float, fov_y: float) -> torch.Tensor:
    """
    The projection matrix models the transformation from the 3D camera space to
    the 2D screen space: effectively, you project all points onto that 2D plane
    using this matrix.

    It takes into account the field of view (what is visible from the camera for x-y axes)
    along with Z_FAR and Z_NEAR which accounts for points visible depth-wise
    """
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


def compute_covering_bbox(
    screen_means: torch.Tensor, projected_covariances: torch.Tensor, width: float, height: float,
) -> torch.Tensor:
    """
    For each 2D projected gaussian, we first compute its spread using its eigen values. And since
    we need to know the covered area in terms of pixels (ie. cannot model an ellipsoid), we approximate
    the spread with a bounding box centered on the gaussian.
    """

    det = (
        projected_covariances[:, 0, 0] * projected_covariances[:, 1, 1]
        - projected_covariances[:, 1, 0] * projected_covariances[:, 0, 1]
    )
    trace = projected_covariances[:, 0, 0] + projected_covariances[:, 1, 1]

    # Have to clamp to 0 in case lambda is negative (no guarantee it is not)
    # To preven instabilities, we set the max at 0.1 (value defined in the original implementation)
    lambda1 = trace / 2.0 + torch.sqrt(
        torch.max((trace ** 2) / 4.0 - det, torch.tensor([0.1], device=screen_means.device))
    )
    lambda2 = trace / 2.0 - torch.sqrt(
        torch.max((trace ** 2) / 4.0 - det, torch.tensor([0.1], device=screen_means.device))
    )

    max_spread = torch.ceil(
        GAUSSIAN_SPREAD * torch.sqrt(torch.max(torch.stack([lambda1, lambda2], dim=-1), dim=-1).values)
    )

    # The original implementation divides the screen space in blocks of size BLOCK_SIZE
    # We keep this paradigm here so that we can more easily map this step back to the original implementation
    # but for this simplified implementation, this is not required.
    bboxes = torch.stack(
        [
            torch.clamp((screen_means[:, 0] - (max_spread)) / BLOCK_SIZE, 0, width - 1),
            torch.clamp((screen_means[:, 1] - (max_spread)) / BLOCK_SIZE, 0, height - 1),
            torch.clamp((screen_means[:, 0] + (max_spread + BLOCK_SIZE - 1)) / BLOCK_SIZE, 0, width - 1,),
            torch.clamp((screen_means[:, 1] + (max_spread + BLOCK_SIZE - 1)) / BLOCK_SIZE, 0, height - 1,),
        ],
        dim=-1,
    )

    # Clamp again for gaussians that spread outside of the screen
    bboxes = torch.floor(bboxes).to(int)
    return bboxes


def compute_2d_covariance(
    cov_matrices, camera_space_points, tan_fov_x, tan_fov_y, focals, world_to_camera
) -> torch.Tensor:
    """
    The spread of each gaussian needs to be projected in screen space (similarly to each gaussian center).
    This is done by projecting the covariance matrices of each gaussian using the EWA Splatting technique.
    
    The original implementation is located at: https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d/cuda_rasterizer/forward.cu#L74
    """
    limx = torch.tensor([1.3 * tan_fov_x], device=cov_matrices.device)
    limy = torch.tensor([1.3 * tan_fov_y], device=cov_matrices.device)

    # TODO: this should be fixed upstream
    focal_x, focal_y = focals / 2

    txtz = camera_space_points[:, 0] / camera_space_points[:, 2]
    tytz = camera_space_points[:, 1] / camera_space_points[:, 2]
    tx = torch.min(limx, torch.max(-limx, txtz)) * camera_space_points[:, 2]
    ty = torch.min(limy, torch.max(-limy, tytz)) * camera_space_points[:, 2]

    # Compute the Jacobian matrix
    J = torch.zeros((camera_space_points.shape[0], 3, 3), device=cov_matrices.device)
    J[:, 0, 0] = focal_x / camera_space_points[:, 2]
    J[:, 0, 2] = -(focal_x * tx) / (camera_space_points[:, 2] * camera_space_points[:, 2])
    J[:, 1, 1] = focal_y / camera_space_points[:, 2]
    J[:, 1, 2] = -(focal_y * ty) / (camera_space_points[:, 2] * camera_space_points[:, 2])

    W = world_to_camera[:-1, :-1].T

    T = torch.bmm(W.expand(J.shape[0], 3, 3).transpose(2, 1), J.transpose(2, 1)).transpose(2, 1)

    vrk = torch.zeros((camera_space_points.shape[0], 3, 3), device=cov_matrices.device)
    vrk[:, 0, 0] = cov_matrices[:, 0, 0]
    vrk[:, 0, 1] = cov_matrices[:, 0, 1]
    vrk[:, 0, 2] = cov_matrices[:, 0, 2]
    vrk[:, 1, 0] = cov_matrices[:, 0, 1]
    vrk[:, 1, 1] = cov_matrices[:, 1, 1]
    vrk[:, 1, 2] = cov_matrices[:, 1, 2]
    vrk[:, 2, 0] = cov_matrices[:, 0, 2]
    vrk[:, 2, 1] = cov_matrices[:, 1, 2]
    vrk[:, 2, 2] = cov_matrices[:, 2, 2]

    proj_cov = T @ vrk @ T.transpose(2, 1)

    # Apply low-pass filter: every Gaussian should be at least
    # one pixel wide/high. Discard 3rd row and column.
    proj_cov[:, 0, 0] += 0.3
    proj_cov[:, 1, 1] += 0.3

    return proj_cov[:, :2, :2]


def rasterize_gaussian(
    gaussian_index: int,
    bboxes: torch.Tensor,
    screen: torch.Tensor,
    screen_means: torch.Tensor,
    sigmas: torch.Tensor,
    rgb: torch.Tensor,
    opacity_buffer: torch.Tensor,
    opacity: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Here we rasterize a gaussian, ie. compute what pixels are covered by the gaussian and its spread
    and "blend" the gaussian onto the existing screen (where previous gaussians have already been blended)
    """
    sigma_x, sigma_y, sigma_x_y = sigmas[gaussian_index]

    x_grid = torch.arange(bboxes[gaussian_index, 0], bboxes[gaussian_index, 2])
    y_grid = torch.arange(bboxes[gaussian_index, 1], bboxes[gaussian_index, 3])

    mesh_x, mesh_y = torch.meshgrid(x_grid, y_grid, indexing="ij")
    mesh = torch.stack([mesh_x, mesh_y], dim=-1).view(-1, 2).to(screen_means.device)

    # We compute the transmittance of the gaussian at each pixel covered which determines how much the new
    # gaussian contributes to the color of the resulting pixel
    dist_to_mean = screen_means[gaussian_index] - mesh
    gaussian_density = (
        -0.5 * (sigma_x * (dist_to_mean[:, 0] ** 2) + sigma_y * (dist_to_mean[:, 1] ** 2))
        - sigma_x_y * dist_to_mean[:, 0] * dist_to_mean[:, 1]
    )

    alpha = torch.min(
        opacity[gaussian_index] * torch.exp(gaussian_density),
        torch.tensor([MAX_GAUSSIAN_DENSITY], device=screen_means.device),
    ).float()

    # For numerical stability
    valid = (alpha > MIN_ALPHA) & (gaussian_density <= 0)
    valid_mesh = mesh[valid, :]

    # Update the screen pixels with the alpha blending values for each of the pixel
    screen[valid_mesh[:, 0], valid_mesh[:, 1], :] += (
        alpha[valid, None] * rgb[gaussian_index] * opacity_buffer[valid_mesh[:, 0], valid_mesh[:, 1], None]
    )

    # Update the opacity buffer to track how much transmittance is left before each pixel is "saturated"
    # i.e cannot transmit color from "deeper" gaussians
    opacity_buffer[valid_mesh[:, 0], valid_mesh[:, 1]] = opacity_buffer[valid_mesh[:, 0], valid_mesh[:, 1]] * (
        1 - alpha[valid]
    )

    return screen, opacity_buffer


@click.command()
@click.option("--input_dir", type=str, default="")
@click.option("--trained_model_path", type=str, default="")
@click.option("--output_path", type=str, default="")
@click.option("--generate_video", is_flag=True, type=bool, default=False)
def run_rasterization(
    input_dir: str, trained_model_path, output_path: Optional[str], generate_video: bool = False,
):
    torch.set_num_threads(os.cpu_count() - 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scenes, cam_info = read_scene(path_to_scene=input_dir)
    scene = scenes[2]

    # Loading the ground truth image
    # images_{scale_fraction} i.e if 2, image has been shrunk by a factor 2
    gt_img_path = os.path.join(os.path.join(input_dir, "images_2"), scene.name)
    img = Image.open(gt_img_path)

    fx, fy, _, _ = cam_info[1].params
    focals = np.array([fx, fy])
    width, height = img.size

    qvec = torch.tensor(scene.qvec)
    tvec = torch.tensor(scene.tvec)

    plydata = PlyData.read(os.path.join(trained_model_path, "point_cloud/iteration_30000/point_cloud.ply"))
    gaussian_means = torch.tensor(
        np.stack([plydata.elements[0]["x"], plydata.elements[0]["y"], plydata.elements[0]["z"],]).T, device=device
    ).float()
    opacity = torch.sigmoid(torch.tensor(np.array(plydata.elements[0]["opacity"]), device=device))

    # The coordinates of the projection/camera center are given by -R^t * T, where R^t is the inverse/transpose of
    # the 3x3 rotation matrix composed from the quaternion and T is the translation vector.
    world_to_camera = get_world_to_camera_matrix(qvec, tvec).transpose(0, 1).to(device)
    fov_x = 2 * np.arctan(cam_info[1].width / (2 * fx))
    fov_y = 2 * np.arctan(cam_info[1].height / (2 * fy))
    tan_fov_x = math.tan(fov_x * 0.5)
    tan_fov_y = math.tan(fov_y * 0.5)

    projection_matrix = get_projection_matrix(fov_x, fov_y).transpose(0, 1).to(device)

    full_proj_transform = (world_to_camera.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    colors = read_color_components(plydata).to(device)

    camera_space_gaussian_means = project_to_camera_space(gaussian_means, world_to_camera)

    rgb = sh_to_rgb(gaussian_means, colors, world_to_camera, degree=3)

    # Computing only with the view matrix
    p_view = gaussian_means @ world_to_camera[:3, :] + world_to_camera[-1, :]

    # This is new, and exactly based on what the paper is doing
    points = gaussian_means @ full_proj_transform[:3, :] + full_proj_transform[-1, :]

    # Frustum culling
    frustum_culling_filter = p_view[:, 2] < 0.2
    points[frustum_culling_filter] = 0.0

    p_w = 1.0 / (points[:, -1] + 0.0000001)
    p_proj = points[:, :-1] * p_w[:, None]

    covariance_matrices = get_covariance_matrix_from_mesh(plydata).float().to(device)

    projected_covariances = compute_2d_covariance(
        covariance_matrices, camera_space_gaussian_means, tan_fov_x, tan_fov_y, focals, world_to_camera,
    )
    projected_covariances[frustum_culling_filter] = 0.0

    screen_means = ((p_proj[:, :2] + 1.0) * torch.tensor([width, height], device=device) - 1.0) / 2

    rounded_bboxes = compute_covering_bbox(screen_means, projected_covariances, width, height)

    screen = torch.zeros((int(width), int(height), 3), device=device).float()
    opacity_buffer = torch.ones((int(width), int(height)), device=device).float()

    det = (
        projected_covariances[:, 0, 0] * projected_covariances[:, 1, 1]
        - projected_covariances[:, 1, 0] * projected_covariances[:, 0, 1]
    )
    # det can underflow into 0, so have to zero-out the inverse of det as well
    # More generally, if the determinant is 0 for a gaussian, it means that its density does not span a 3D space (ie could be a line or plane)
    det_inv = torch.where(det == 0, 0, 1 / det)

    # Computing the
    sigmas = torch.stack(
        [
            projected_covariances[:, 1, 1] * det_inv[:],
            projected_covariances[:, 0, 0] * det_inv[:],
            -projected_covariances[:, 0, 1] * det_inv[:],
        ],
        dim=-1,
    )

    x_min = torch.clamp(rounded_bboxes[:, 0] * BLOCK_SIZE, 0, width - 1)
    y_min = torch.clamp(rounded_bboxes[:, 1] * BLOCK_SIZE, 0, height - 1)
    x_max = torch.clamp(rounded_bboxes[:, 2] * BLOCK_SIZE, 0, width - 1)
    y_max = torch.clamp(rounded_bboxes[:, 3] * BLOCK_SIZE, 0, height - 1)

    bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    depths = p_view[:, 2]
    depth_sorted_gaussians = torch.sort(depths).indices

    bbox_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    """
    Here is where the inefficiency comes in compared to the official implementation:
    since we cannot parallelize the rasterization, we have to loop over the gaussians and
    rasterize them one by one (instead of distributing this process with a CUDA kernel)
    """
    iteration_step = 0
    for gaussian_index in tqdm.tqdm(depth_sorted_gaussians):
        if bbox_area[gaussian_index] == 0 or (torch.any(sigmas[gaussian_index] == 0)):
            continue
        screen, opacity_buffer = rasterize_gaussian(
            gaussian_index, bboxes, screen, screen_means, sigmas, rgb, opacity_buffer, opacity,
        )

        if iteration_step % 1000 == 0 and generate_video:
            img = Image.fromarray((screen[:, :, :3].transpose(1, 0).cpu().numpy() * 255.0).astype(np.uint8))
            img.save(os.path.join(output_path, f"image_iter_{str(iteration_step).zfill(7)}.png",))

        iteration_step += 1

    if generate_video:
        for i in range(1, 21):
            # We add 2 secs of video to let some time to see the fully recreated image before the video ends
            img.save(os.path.join(output_path, f"image_iter_{str(iteration_step + 1000*i).zfill(7)}.png",))

        video_path = os.path.join(output_path, "output.mp4")
        if os.path.exists(video_path):
            os.remove(video_path)
        cmd = f'ffmpeg -framerate 20 -pattern_type glob -i "/home/arnaud/splat_images/image_iter_*.png" -r 10 -vcodec libx264 -s {width - (width % 2)}x{height - (height % 2)} -pix_fmt yuv420p {video_path}'
        subprocess.run(cmd, shell=True, check=True)

    # Create a figure
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)  # 2 rows, 1 column1, 1st subplot
    plt.imshow(screen[:, :, :3].transpose(1, 0).cpu())
    plt.title("Reconstructed Image")

    # Display the second image
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
    plt.imshow(mpimg.imread(gt_img_path))
    plt.title("Reference Image")

    plt.show()


if __name__ == "__main__":

    run_rasterization()
