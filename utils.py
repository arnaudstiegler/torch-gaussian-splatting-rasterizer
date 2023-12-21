import torch
from plyfile import PlyElement

def read_color_components(plydata: PlyElement) -> torch.Tensor:
    
    rgb_tensors = []
    for rgb_index in range(3):
        components_to_stack = []
        components_to_stack.append(torch.tensor(plydata.elements[0][f'f_dc_{rgb_index}']))
        for j in range(rgb_index*15, (rgb_index+1)*15):
            components_to_stack.append(torch.tensor(plydata.elements[0][f'f_rest_{j}']))
        rgb_tensors.append(torch.stack(components_to_stack))
    
    # Dimension: [N_gaussians, 3 (rgb), num sh coefficients]
    return torch.stack(rgb_tensors, dim=-1).view(-1, 3, 16)