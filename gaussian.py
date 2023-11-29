from torch.nn import Parameter
import torch

class GaussianModel:
    def __init__(self):
        # All parameters for the Gaussian
        # position, covariance, 𝛼 and SH coefficients

        # Rotation parameters are represented as a quaternion
        self.quaternion = Parameter(torch.zeros(1,4))

        # a 3D vector 𝑠 for scaling
        self.scaling = Parameter(torch.zeros(3))