import os
from typing import Dict, List, Tuple

import torch
from plyfile import PlyElement

from data_reader import (BaseImage, Camera, read_extrinsics_binary,
                         read_intrinsics_binary)


def read_color_components(plydata: PlyElement) -> torch.Tensor:
    """
    Reads color attributes from the trained model.
    Since the color of a given gaussian enveloppe can differ based on the orientation/POV, we model gaussian colors using spherical harmonics
    This method is presented in: http://arxiv.org/abs/2205.14330

    f_dc is a `[N,3]` vector that corresponds to the base color in RGB format for the zero-th order spherical harmonic.
    f_rest are the remaining coefficients for the harmonics where the highest-order is 3, which means 15 coefficient per color, i.e 45 coefficients total.
    See: https://github.com/graphdeco-inria/gaussian-splatting/issues/485
    """

    dc_parameters = torch.stack([torch.tensor(plydata.elements[0][f"f_dc_{rgb_index}"]) for rgb_index in range(3)])

    rgb_tensors = []
    for rgb_index in range(3):
        components_to_stack = []
        for j in range(rgb_index * 15, (rgb_index + 1) * 15):
            components_to_stack.append(torch.tensor(plydata.elements[0][f"f_rest_{j}"]))
        rgb_tensors.append(torch.stack(components_to_stack))

    # Dimension: [N_gaussians, 3 (rgb), num sh coefficients]
    return torch.concatenate([dc_parameters.unsqueeze(1), torch.stack(rgb_tensors)], dim=1).transpose(2, 0)


def read_scene(path_to_scene: str) -> Tuple[List[BaseImage], Dict[int, Camera]]:
    """
    For the extrinsics, the content is:
    - image_id
    - qvec: rotation quaternion to go from world coordinate system to camera coordinate system
    - tvec: translation vector to go from world coordinate system to camera coordinate system
    - xys: array of size [N, 2] which represents the 2d coordinates of every point used for reconstruction
    - point3D_ids: array of size [N] where id == -1 if the point is not visible on the image, else it's the point id

    For the intrinsics, the content is:
    - camera model: how to approximate the camera lens (here we use PINHOLE which is the simplest model)
    - screen width and height
    - params:
        - Fx, Fy the focals for the camera which are used to compute the Field of View ("cone of observability" from the camera)
        - Cx, Cy the center of the camera on the screen
    """
    cameras_extrinsic_file = os.path.join(path_to_scene, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path_to_scene, "sparse/0", "cameras.bin")
    # This is the position for each image
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)

    # This is the properties of the camera itself
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    return cam_extrinsics, cam_intrinsics
