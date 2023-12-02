import numpy as np
import matplotlib.pyplot as plt
from rasterizer_playground.perspective_transform import perspective_project

def sign(p1, p2, p3) -> float:
    '''
    Function that returns the signed area of the parallelogram defined by p1p3 and p2p3
    '''
    return (p1[0] - p3[0])*(p2[1] - p3[1]) - (p2[0] - p3[0])*(p1[1] - p3[1])


def is_in_triangle(point, triangle):
    '''
    Compute the signed area for all parallelograms defined by the point and a pair of the triangle vertices
    If signs are all the same, it means the point is "on the same side" of every triangle edge, ie. within the triangle
    '''
    sign1 = sign(point, triangle[0], triangle[1])
    sign2 = sign(point, triangle[1], triangle[2])
    sign3 = sign(point, triangle[2], triangle[0])

    has_neg = sign1<0 or sign2<0 or sign3 < 0
    has_pos = sign1>0 or sign2>0 or sign3>0
    return not (has_neg and has_pos)

triangle = np.array([[0.0,0.0,0.0], [10.0,0.0,0.0], [0.0,5.0,0.0]])

camera_args = {
    # No rotation, just "stepping back" from the origin
    'camera_to_world': np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0, 1.0]
    ]),
    # We assume the distance between the camera and the canvas is one unit
    # This is an approximation, and canvas_dist is usually defined by FOV
    'canvas_dist': 1,
    # Screen properties
    'width': 100,
    'height': 100,
    'pixel_width': 1.0,
    'pixel_height': 1.0
}

raster_triangle = perspective_project(triangle, camera_args)

screen = np.zeros((int(camera_args['width'] / camera_args['pixel_width']), int(camera_args['height'] / camera_args['pixel_height'])))

# Find the bbox for the triangle so that we don't uselessly go over pixels
bbox = np.concatenate([np.min(raster_triangle[:, :2], axis=0), np.max(raster_triangle[:, :2], axis=0)]).astype(int)

for x_value in range(bbox[0], bbox[2]):
    for y_value in range(bbox[1], bbox[3]):
        # For each
        if is_in_triangle([x_value, y_value], raster_triangle):
            screen[x_value, y_value] = 1

# Marking the triangle vertices another color
screen[raster_triangle[:,0], raster_triangle[:,1]] = 0.5

plt.figure(figsize=(10, 10), dpi=100)  # Adjust the figure size and dpi as needed
plt.imshow(screen)
plt.show()