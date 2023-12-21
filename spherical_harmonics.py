import torch

# See table of real spherical harmonics at: spherical_harmonics.md

# Zero-th order
SH_0 = 0.28209479177387814  # this is sqrt(1/(4*PI))
# First order (there should be 3 difference coefficients but all 3 are equal)
SH_C1 = 0.4886025119029199 # this is 0.5*math.sqrt(3/math.pi)
# Second order
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792, # Not sure where the - is coming from but this is how it's formatted on the official repo
    0.31539156525252005,
    -1.0925484305920792, # Same comment
    0.5462742152960396
]


def sh_to_rgb(xyz, sh, degree=0):
    '''
    Formula to retrieve RGB colors from polynomial spherical harmonics 
    '''

    # Gotta normalize the vector below

    normalized_xyz = xyz / torch.norm(xyz, dim=1)[:, None]

    x = normalized_xyz[:,0].view(-1, 1)
    y = normalized_xyz[:,1].view(-1, 1)
    z = normalized_xyz[:,2].view(-1, 1)

    colors = sh[:, :, 0]*SH_0

    if degree > 0:
        colors += - SH_C1*y*sh[:, :, 1] + SH_C1*z*sh[:, :, 2] - SH_C1*x*sh[:, :, 3]

        if degree > 1:
            colors += (
                SH_C2[0]*x*y*sh[:, :, 4] 
                + SH_C2[1] * y * z * sh[:, :, 5]
                + SH_C2[2] * (2 * z*z - x*x - y*y) * sh[:, :, 6]
                + SH_C2[3] * x * z * sh[:, :, 7]
                + SH_C2[4] * (x*x - z*z) * sh[:, :, 8]
            )

    # Colors are centered around 0
    # This offset recenters them around 0.5 (so in the [0,1] range)
    # colors += 0.5
    # Since there's a trainable component in the mix, we have to clamp colors to ensure all values are positive
    colors = torch.clamp(colors, 0, 1)*255

    return colors


	# 		if (deg > 2)
	# 		{
	# 			result = result +
	# 				SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
	# 				SH_C3[1] * xy * z * sh[10] +
	# 				SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
	# 				SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
	# 				SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
	# 				SH_C3[5] * z * (xx - yy) * sh[14] +
	# 				SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
	# 		}
	# 	}
	# }
	# result += 0.5f;

    # return colors