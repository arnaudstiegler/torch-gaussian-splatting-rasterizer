import numpy as np
import matplotlib.pyplot as plt
from rasterizer_playground.perspective_transform import perspective_project, project_to_camera_space

def edge_function(p1, p2, p3) -> float:
    '''
    Function that returns the signed area of the parallelogram defined by p1p3 and p2p3
    '''
    return (p1[0] - p3[0])*(p2[1] - p3[1]) - (p2[0] - p3[0])*(p1[1] - p3[1])


def is_in_triangle(point, triangle):
    '''
    Compute the signed area for all parallelograms defined by the point and a pair of the triangle vertices
    If signs are all the same, it means the point is "on the same side" of every triangle edge, ie. within the triangle
    '''
    e1 = edge_function(point, triangle[0], triangle[1])
    e2 = edge_function(point, triangle[1], triangle[2])
    e3 = edge_function(point, triangle[2], triangle[0])
    triangle_area = edge_function(*triangle)

    has_neg = e1<0 or e2<0 or e3 < 0
    has_pos = e1>0 or e2>0 or e3>0
    return not (has_neg and has_pos), [e1/triangle_area,e2/triangle_area,e3/triangle_area]

triangle = np.array([[0.0,0.0,0.0], [10.0,0.0,0.0], [0.0,5.0,0.0]])
triangle_colors = np.array([[255,0.0,0.0], [0.0,255,0.0], [0.0,0.0,255]])

camera_args = {
    # No rotation, just "stepping back" from the origin
    'camera_to_world': np.array([
        [1.0, 0.8, 0.0, 0.0],
        [0.2, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.9, 0.0],
        [0.0, 0.0, -1.0, 1.0]
    ]),
    # We assume the distance between the camera and the canvas is one unit
    # This is an approximation, and canvas_dist is usually defined by FOV
    'canvas_dist': 1,
    # Screen properties
    'width': 25,
    'height': 25,
    'pixel_width': 0.01,
    'pixel_height': 0.01
}

raster_triangle = perspective_project(triangle, camera_args)


screen = np.zeros((int(camera_args['width'] / camera_args['pixel_width']), int(camera_args['height'] / camera_args['pixel_height']), 3))

# Find the bbox for the triangle so that we don't uselessly go over pixels
bbox = np.concatenate([np.min(raster_triangle[:, :2], axis=0), np.max(raster_triangle[:, :2], axis=0)]).astype(int)

# Pre-computing for perspective correct interpolation
camera_triangle = project_to_camera_space(triangle, camera_args)
invert_depth = 1 / camera_triangle[:, 2]

for x_value in range(bbox[0], bbox[2]):
    for y_value in range(bbox[1], bbox[3]):
        is_in, barycentric_weights = is_in_triangle([x_value, y_value], raster_triangle)
        if is_in:
            # This is an incorrect interpolation (since the projection is non-linear)
            # screen[x_value, y_value, :] = (
            #     barycentric_weights[0]*triangle_colors[0] + 
            #     barycentric_weights[1]*triangle_colors[1] +
            #     barycentric_weights[2]*triangle_colors[2] 
            #     )
            # Assign color as a weighted sum of the colors of the 3 vertices
            # using perspective-correct interpolation
            depth = 1/((barycentric_weights[0]*invert_depth[0]) + (barycentric_weights[1]*invert_depth[1]) + (barycentric_weights[2]*invert_depth[2]))
            screen[x_value, y_value, :] = depth*(
                barycentric_weights[0]*(triangle_colors[0]*invert_depth[0]) + 
                barycentric_weights[1]*(triangle_colors[1]*invert_depth[1]) +
                barycentric_weights[2]*(triangle_colors[2]*invert_depth[2]) 
                )

# Marking the triangle vertices another color
screen[raster_triangle[:,0], raster_triangle[:,1]] = triangle_colors

# For matplotlib to not clip the color values, have to turn into integers
screen = screen.astype(int)

plt.figure(figsize=(10, 10), dpi=100)  # Adjust the figure size and dpi as needed
plt.imshow(screen)
plt.show()