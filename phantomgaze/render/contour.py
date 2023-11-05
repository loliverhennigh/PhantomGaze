# Render functions for rendering a contour of a volume.

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
def contour_kernel(
        volume_array,
        spacing,
        origin,
        camera_position,
        camera_focal,
        camera_up,
        threshold,
        color_array,
        color_map_array,
        vmin,
        vmax,
        opacity,
        img,
        depth):
    """Kernel for rendering a contour of a volume.

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
    threshold : float
        The threshold to use for the contour.
    color_array : ndarray
        The color data.
    color_map_array : ndarray
        The color map array.
    vmin : float
        The minimum value of the scalar range.
    vmax : float
        The maximum value of the scalar range.
    opacity : float
        The opacity of the contour.
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

    # Set starting value to lowest possible value
    value = sample_array(volume_array, spacing, origin, ray_pos)

    # Inside-outside stored in the sign
    sign = 1 if value > threshold else -1

    # Start the ray marching
    distance = t0
    for step in range(int((t1 - t0) / step_size)):
        # Check if distance is greater then current depth or 
        if (distance > depth[y, x]):
            return

        # Get next step position
        next_ray_pos = (
            ray_pos[0] + step_size * ray_direction[0],
            ray_pos[1] + step_size * ray_direction[1],
            ray_pos[2] + step_size * ray_direction[2]
        )

        # Get the value in the next step
        next_value = sample_array(volume_array, spacing, origin, next_ray_pos)

        # If contour is crossed, set the color and depth
        if (next_value - threshold) * sign < 0:
            # Update the sign
            sign = -sign

            # Linearly interpolate the position
            t = (threshold - value) / (next_value - value)
            pos_contour = (
                ray_pos[0] + t * step_size * ray_direction[0],
                ray_pos[1] + t * step_size * ray_direction[1],
                ray_pos[2] + t * step_size * ray_direction[2]
            )

            # Get gradient
            gradient = sample_array_derivative(
                volume_array, spacing, origin, pos_contour)

            # Normalize gradient
            gradient = normalize(gradient)

            # Calculate intensity
            intensity = dot(gradient, ray_direction)
            intensity = abs(intensity)

            # Get the color
            scalar = sample_array(color_array, spacing, origin, pos_contour)
            color = scalar_to_color(
                scalar, color_map_array, vmin, vmax)

            # Blend the color
            cur_color = (
                img[y, x, 0],
                img[y, x, 1],
                img[y, x, 2],
                img[y, x, 3],
            )
            new_color = (
                color[0] * intensity,
                color[1] * intensity,
                color[2] * intensity,
                opacity
            )
            color = blend_colors(new_color, cur_color)

            # Set the color
            img[y, x, 0] = color[0]
            img[y, x, 1] = color[1]
            img[y, x, 2] = color[2]
            img[y, x, 3] = color[3]

            # Set the depth
            if opacity == 1.0:
                depth[y, x] = distance + t * step_size

        # Update the value and position
        value = next_value
        ray_pos = next_ray_pos
        distance += step_size


def contour(volume, camera, threshold, color=None, colormap=None, opacity=1.0, img=None, depth=None):
    """Render a contour of a volume.

    Parameters
    ----------
    volume : Volume
        The volume to render.
    camera : Camera
        The camera to render from.
    threshold : float
        The threshold to use for the contour.
    color : Volume, optional
        The color of the contour. If None, the color will be grey/white.
    colormap : Colormap, optional
        The colormap to use for the volume. If None, the colormap will be
        jet.
    opacity : float, optional
        The opacity of the contour. 1.0 is opaque and 0.0 is transparent.
    img : ndarray, optional
        The image to render to. If None, a new image will be created.
    depth : ndarray, optional
        The depth buffer to render to. If None, a new depth buffer will be
    """

    # Create the image and depth buffer if necessary
    if img is None:
        img = cp.zeros((camera.height, camera.width, 4), dtype=cp.float32)
        img[:, :, 3] = 1.0
    if depth is None:
        depth = cp.zeros((camera.height, camera.width), dtype=cp.float32) + cp.nan # nan is the closest thing to infinity

    # Set up thread blocks
    threads_per_block = (16, 16)
    blocks = (
        (img.shape[1] + threads_per_block[0] - 1) // threads_per_block[0],
        (img.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
    )

    # Get color data if necessary
    if color is None:
        color_array = cp.zeros(volume.shape, dtype=cp.float32)
    else:
        color_array = color.array

    # Get colormap if necessary
    if colormap is None:
        colormap = Colormap('jet', float(color_array.min()), float(color_array.max()))

    # Run kernel
    contour_kernel[blocks, threads_per_block](
        volume.array,
        volume.spacing,
        volume.origin,
        camera.position,
        camera.focal_point,
        camera.view_up,
        threshold,
        color_array,
        colormap.color_map_array,
        colormap.vmin,
        colormap.vmax,
        opacity,
        img,
        depth
    )

    return img, depth
