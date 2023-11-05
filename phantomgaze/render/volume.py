# Render functions for volumes 

import cupy as cp
import numba
from numba import cuda

from phantomgaze import Volume
from phantomgaze.colormap import Colormap
from phantomgaze.render.camera import calculate_ray_direction
from phantomgaze.render.utils import sample_array, sample_array_derivative
from phantomgaze.render.math import normalize, dot, cross
from phantomgaze.render.color import scalar_to_color, blend_colors
from phantomgaze.render.geometry import ray_intersect_box


@cuda.jit
def volume_kernel(
        volume_array,
        spacing,
        origin,
        camera_position,
        camera_focal,
        camera_up,
        color_map_array,
        vmin,
        vmax,
        opacity,
        img,
        depth):
    """Kernel for rendering a volume.

    Parameters
    ----------
    volume_array : ndarray
        The volume data.
    spacing : tuple
        The spacing of the volume data.
    origin : tuple
        The origin of the volume data.
    camera_position : tuple
        The position of the camera.
    camera_focal : tuple
        The focal point of the camera.
    camera_up : tuple
        The up vector of the camera.
    color_map_array : ndarray
        The color map array.
    vmin : float
        The minimum value of the scalar range.
    vmax : float
        The maximum value of the scalar range.
    opacity : float
        The opacity of the volume.
    img : ndarray
        The image to render to.
    depth : ndarray
        The depth buffer to render to.
    """

    # Get the x and y indices
    x, y = cuda.grid(2)

    # Make sure the indices are in bounds
    if x >= img.shape[1] or y >= img.shape[0]:
        return

    # Get ray direction
    ray_direction = calculate_ray_direction(
            x, y, img.shape,
            camera_position, camera_focal, camera_up)

    # Get volume upper bound
    volume_upper = (
        origin[0] + spacing[0] * volume_array.shape[0],
        origin[1] + spacing[1] * volume_array.shape[1],
        origin[2] + spacing[2] * volume_array.shape[2]
    )

    # Get the intersection of the ray with the volume
    t0, t1 = ray_intersect_box(
        origin, volume_upper, camera_position, ray_direction)

    # If there is no intersection, return
    if t0 > t1:
        return

    # Get the starting point of the ray
    ray_pos = (
        camera_position[0] + t0 * ray_direction[0],
        camera_position[1] + t0 * ray_direction[1],
        camera_position[2] + t0 * ray_direction[2]
    )

    # Get the step size
    step_size = min(spacing[0], min(spacing[1], spacing[2]))

    # Get depth
    current_depth = depth[y, x]
    if current_depth == cp.nan:
        current_depth = cp.inf

    # Start the ray marching
    distance = t0
    accum_color = (0.0, 0.0, 0.0, 0.0)
    for step in range(int((t1 - t0) / step_size)):
        # Break out of the kernel if the distance is greater than the depth
        if (distance > current_depth) or (accum_color[3] > 0.99):
            break

        # Get the value at the current position
        value = sample_array(volume_array, spacing, origin, ray_pos)

        # Get the color
        color = scalar_to_color(value, color_map_array, vmin, vmax)

        # Accumulate the color using the opacity
        accum_color = (
            accum_color[0] + color[0] * opacity * (1.0 - accum_color[3]),
            accum_color[1] + color[1] * opacity * (1.0 - accum_color[3]),
            accum_color[2] + color[2] * opacity * (1.0 - accum_color[3]),
            accum_color[3] + opacity * (1.0 - accum_color[3])
        )

        # Increment the distance
        ray_pos = (
            ray_pos[0] + step_size * ray_direction[0],
            ray_pos[1] + step_size * ray_direction[1],
            ray_pos[2] + step_size * ray_direction[2]
        )
        distance += step_size

    # Blend the color
    current_color = (
        img[y, x, 0],
        img[y, x, 1],
        img[y, x, 2],
        img[y, x, 3]
    )
    color = blend_colors(accum_color, current_color)

    # Set the color
    img[y, x, 0] = color[0]
    img[y, x, 1] = color[1]
    img[y, x, 2] = color[2]
    img[y, x, 3] = color[3]


def volume(volume, camera, colormap=None, opacity=0.001, img=None, depth=None):
    """Render a volume

    Parameters
    ----------
    volume : Volume
        The volume to render.
    camera : Camera
        The camera to render with.
    colormap : ColorMap
        The colormap to use.
    opacity : float
        The opacity of the volume.
    img : ndarray
        The image to render to.
    depth : ndarray
        The depth buffer to render to.
    """

    # Create the image and depth buffer if necessary
    if img is None:
        img = cp.zeros((camera.height, camera.width, 4), dtype=cp.float32)
    if depth is None:
        depth = cp.zeros((camera.height, camera.width), dtype=cp.float32) + cp.nan # nan is the closest thing to infinity

    # Set up thread blocks
    threads_per_block = (16, 16)
    blocks = (
        (img.shape[1] + threads_per_block[0] - 1) // threads_per_block[0],
        (img.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
    )

    # Get colormap if necessary
    if colormap is None:
        colormap = Colormap('jet', float(color_array.min()), float(color_array.max()))

    # Run kernel
    volume_kernel[blocks, threads_per_block](
        volume.array,
        volume.spacing,
        volume.origin,
        camera.position,
        camera.focal_point,
        camera.view_up,
        colormap.color_map_array,
        colormap.vmin,
        colormap.vmax,
        opacity,
        img,
        depth
    )

    return img, depth
