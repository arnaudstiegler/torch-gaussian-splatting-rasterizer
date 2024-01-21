import torch

# See table of real spherical harmonics at: spherical_harmonics.md

# Zero-th order
SH_0 = 0.28209479177387814  # this is sqrt(1/(4*PI))
# First order (there should be 3 difference coefficients but all 3 are equal)
SH_C1 = 0.4886025119029199  # this is 0.5*math.sqrt(3/math.pi)
# Second order
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,  # Not sure where the - is coming from but this is how it's formatted on the official repo
    0.31539156525252005,
    -1.0925484305920792,  # Same comment
    0.5462742152960396,
]
# Third order
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def sh_to_rgb(xyz, sh, world_view_transform, degree=0):
    """
    Formula to retrieve RGB colors from polynomial spherical harmonics 
    """

    cam_center = world_view_transform.inverse()[3, :3]
    dir = xyz - cam_center
    dir = dir / torch.norm(dir, dim=1).unsqueeze(-1).expand(-1, 3)

    x = dir[:, 0].view(-1, 1)
    y = dir[:, 1].view(-1, 1)
    z = dir[:, 2].view(-1, 1)

    colors = sh[:, 0, :] * SH_0

    if degree > 0:
        colors += (
            -SH_C1 * y * sh[:, 1, :] + SH_C1 * z * sh[:, 2, :] - SH_C1 * x * sh[:, 3, :]
        )

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

    colors = colors + 0.5

    # Colors are centered around 0
    # This offset recenters them around 0.5 (so in the [0,1] range)
    # colors += 0.5
    # Since there's a trainable component in the mix, we have to clamp colors to ensure all values are positive
    colors = torch.clamp(colors, 0, 1)

    return colors
