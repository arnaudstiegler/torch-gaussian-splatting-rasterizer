'''
Formula to retrieve RGB colors from polynomial spherical harmonics 
'''

# Zero-th order
SH_0 = 0.28209479177387814  # this is sqrt(1/(4*PI))

def sh_to_rgb(xyz, sh, degree=0):
    if degree > 0:
        raise ValueError('Not supported yet')
    # TODO: implement degree > 0

    return sh*SH_0