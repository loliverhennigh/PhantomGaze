# Render functions for wireframes

import cupy as cp
import numba
from numba import cuda

from phantomgaze import Volume
from phantomgaze.colormap import Colormap
from phantomgaze.render.utils import sample_array, sample_array_derivative
from phantomgaze.render.math import normalize, dot, cross
from phantomgaze.render.color import scalar_to_color, blend_colors


@cuda.jit
def wireframe_kernel(
        lines_array,
        camera_position,
        camera_focal,
        camera_up,
        img,
        depth):

    """
    Kernel for rendering a wireframe of lines

    Parameters
    ----------
    lines_array : cupy.ndarray
        The array of lines to render. The array should have shape (nr_lines, 3).
    camera_position : tuple of float
        The position of the camera.
    camera_focal : tuple of float
        The focal point of the camera.
    camera_up : tuple of float
        The up vector of the camera.
    img : cupy.ndarray
        The image to render to.
    depth : cupy.ndarray
        The depth buffer to render to.
    """

    # Get the x and y indices
    x, y = cuda.grid(2)

    # Make sure the indices are in bounds
    if x >= img.shape[1] or y >= img.shape[0]:
        return

    # Get depth
    current_depth = depth[y, x]
    if current_depth == cp.nan:
        current_depth = cp.inf

    # Get current pixel color
    current_color = (
        img[y, x, 0],
        img[y, x, 1],
        img[y, x, 2],
        img[y, x, 3]
    )

    # Loop over all lines
    for i in range(lines_array.shape[0]):







def wireframe_box(origin, upper, camera, img=None, depth=None):
    """Render a wireframe of a box.

    Parameters
    ----------
    origin : tuple of float
        The origin of the box.
    upper : tuple of float
        The upper corner of the box.
    camera : Camera
        The camera to render the box from.
    img : cupy.ndarray, optional
        The image to render to. If None, a new image will be created.
    depth : cupy.ndarray, optional
        The depth buffer to render to. If None, a new depth buffer will be created.
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

    # We assume that 'origin' is the lower corner and 'upper' is the opposite upper corner of the box.
    lines_array = cp.array([
        [[origin[0], origin[1], origin[2]], [upper[0], origin[1], origin[2]]],
        [[origin[0], origin[1], origin[2]], [origin[0], upper[1], origin[2]]],
        [[origin[0], origin[1], origin[2]], [origin[0], origin[1], upper[2]]],
        [[upper[0], upper[1], upper[2]], [origin[0], upper[1], upper[2]]],
        [[upper[0], upper[1], upper[2]], [upper[0], origin[1], upper[2]]],
        [[upper[0], upper[1], upper[2]], [upper[0], upper[1], origin[2]]],
        [[origin[0], upper[1], origin[2]], [upper[0], upper[1], origin[2]]],
        [[origin[0], upper[1], origin[2]], [origin[0], upper[1], upper[2]]],
        [[origin[0], origin[1], upper[2]], [upper[0], origin[1], upper[2]]],
        [[origin[0], origin[1], upper[2]], [origin[0], upper[1], upper[2]]],
        [[upper[0], origin[1], origin[2]], [upper[0], origin[1], upper[2]]],
        [[upper[0], origin[1], origin[2]], [upper[0], upper[1], origin[2]]]
    ], dtype=cp.float32)
 
    # Run kernel
    wireframe_kernel[blocks, threads_per_block](
        lines_array,
        camera.position,
        camera.focal,
        camera.up,
        img,
        depth
    )

    return img, depth


