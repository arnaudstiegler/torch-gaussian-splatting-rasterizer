import torch

# Zero-th order coefficients
SH_0 = 0.28209479177387814  # this is sqrt(1/(4*PI))
# First order coefficients
SH_C1 = 0.4886025119029199  # this is 0.5*math.sqrt(3/math.pi)
# Second order coefficients
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
# Third order coefficients
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def sh_to_rgb(xyz: torch.Tensor, sh: torch.Tensor, world_view_transform: torch.Tensor, degree: int = 0) -> torch.Tensor:
    """
    See https://en.wikipedia.org/wiki/Table_of_spherical_harmonics for a good explanation on how spherical harmonics are
    represented.

    Note that here, we use the cartesian coordinates to represent them, which consists in expressing angles using x, y, z
    which explains the slight differences between the wikipedia table and the computation below.
    """
    cam_center = world_view_transform.inverse()[3, :3]
    dir = xyz - cam_center
    dir = dir / torch.norm(dir, dim=1).unsqueeze(-1).expand(-1, 3)

    x = dir[:, 0].view(-1, 1)
    y = dir[:, 1].view(-1, 1)
    z = dir[:, 2].view(-1, 1)

    colors = sh[:, 0, :] * SH_0

    if degree > 0:
        colors += -SH_C1 * y * sh[:, 1, :] + SH_C1 * z * sh[:, 2, :] - SH_C1 * x * sh[:, 3, :]

        if degree > 1:
            colors += (
                SH_C2[0] * x * y * sh[:, 4, :]
                + SH_C2[1] * y * z * sh[:, 5, :]
                + SH_C2[2] * (2 * z * z - x * x - y * y) * sh[:, 6, :]
                + SH_C2[3] * x * z * sh[:, 7, :]
                + SH_C2[4] * (x * x - y * y) * sh[:, 8, :]
            )
            if degree > 2:
                colors += (
                    SH_C3[0] * y * (3 * x * x - y * y) * sh[:, 9, :]
                    + SH_C3[1] * x * y * z * sh[:, 10, :]
                    + SH_C3[2] * y * (4 * z * z - x * x - y * y) * sh[:, 11, :]
                    + SH_C3[3] * z * (2 * z * z - 3 * x * x - 3 * y * y) * sh[:, 12, :]
                    + SH_C3[4] * x * (4 * z * z - x * x - y * y) * sh[:, 13, :]
                    + SH_C3[5] * z * (x * x - y * y) * sh[:, 14, :]
                    + SH_C3[6] * x * (x * x - 3 * y * y) * sh[:, 15, :]
                )

    # Colors are centered around 0
    # This offset recenters them around 0.5 (so in the [0,1] range)
    colors += 0.5
    # Since there's a trainable component in the mix, we have to clamp colors to ensure all values are positive
    colors = torch.clamp(colors, 0, 1)

    return colors
