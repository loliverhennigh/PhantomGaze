# Render functions for volumes

import math
import numba
from numba import cuda

from phantomgaze.utils.math import normalize, dot, cross

@cuda.jit(device=True)
def calculate_ray_direction(
        x,
        y,
        img_shape,
        camera_position,
        camera_focal,
        camera_up):
    """
    Calculate the direction of a ray from the camera to the image plane.

    Parameters
    ----------
    x : int
        The x coordinate of the pixel.
    y : int
        The y coordinate of the pixel.
    img_shape : tuple
        The shape of the image.
    camera_position : tuple
        The position of the camera.
    camera_focal : tuple
        The focal point of the camera.
    camera_up : tuple
        The up vector of the camera.

    Returns
    -------
    ray_direction : tuple
    """

    # Compute base vectors
    forward = (
        camera_focal[0] - camera_position[0],
        camera_focal[1] - camera_position[1],
        camera_focal[2] - camera_position[2],
    )
    forward = normalize(forward)
    right = cross(forward, camera_up)
    right = normalize(right)
    up = cross(right, forward)

    # Determine the center of the image
    center = (
        camera_position[0] + forward[0],
        camera_position[1] + forward[1],
        camera_position[2] + forward[2],
    )

    # Calculate the location on the image plane corresponding (x, y)
    aspect_ratio = img_shape[1] / img_shape[0]
    s = (x - img_shape[1] / 2) / img_shape[1]
    t = (y - img_shape[0] / 2) / img_shape[0]

    # Adjust for aspect ratio and field of view (assuming 90 degrees here)
    s *= aspect_ratio * math.tan(math.pi / 4.0)
    t *= math.tan(math.pi / 4.0)
    point_on_image_plane = (
        center[0] + s * right[0] + t * up[0],
        center[1] + s * right[1] + t * up[1],
        center[2] + s * right[2] + t * up[2],
    )

    # Calculate the ray direction
    ray_direction = (
        point_on_image_plane[0] - camera_position[0],
        point_on_image_plane[1] - camera_position[1],
        point_on_image_plane[2] - camera_position[2],
    )
    ray_direction = normalize(ray_direction)

    return ray_direction
