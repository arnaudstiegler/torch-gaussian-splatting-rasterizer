import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict

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

def project_to_camera_space(triangle, camera_args):
    camera_to_world = camera_args['camera_to_world']

    # World space to camera space
    world_to_camera = np.linalg.inv(camera_to_world)
    triangle_camera = np.dot(triangle, world_to_camera[:3, :3]) + world_to_camera[3, :-1]
    return triangle_camera


def perspective_project(triangle: np.ndarray, camera_args: Dict[str, Any]) -> np.ndarray:
    width = camera_args['width']
    height = camera_args['height']
    pixel_width = camera_args['pixel_width']
    pixel_height = camera_args['pixel_height']
    triangle_camera = project_to_camera_space(triangle, camera_args)

    # Projection onto canvas by dividing by the z-coordinate (since canvas dist is unit)
    # Note that we also drop the z-axis, which would be needed to determine visibility
    projected_triangle = np.divide(triangle_camera[:, :2], triangle_camera[:, -1][:, None])  # The None allows to broadcast the division

    # Project to NDC
    ndc_triangle = np.divide(projected_triangle + np.array([width // 2, height // 2]), np.array([width, height])[None, :])
    # Project to raster space
    raster_triangle = np.floor(np.divide(ndc_triangle * np.array([width, height])[None, :], np.array([pixel_width, pixel_height])))
    raster_triangle = raster_triangle.astype(int)

    return raster_triangle


if __name__ == '__main__':
    raster_triangle  = perspective_project(triangle, camera_args)

    # Generate image
    screen = np.zeros((int(camera_args['width'] / camera_args['pixel_width']), int(camera_args['height'] / camera_args['pixel_height'])))
    screen[raster_triangle[:,0], raster_triangle[:,1]] = 1

    plt.figure(figsize=(10, 10), dpi=100)  # Adjust the figure size and dpi as needed
    plt.imshow(screen)
    plt.show()