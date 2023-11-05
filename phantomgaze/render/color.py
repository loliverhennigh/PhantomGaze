# Color helper functions

import numba
from numba import cuda

@cuda.jit
def scalar_to_color(value, color_map_array, vmin, vmax):
    """Convert a scalar value to a color.

    Parameters
    ----------
    value : float
        The scalar value to convert.
    color_map_array : ndarray
        The color map array.
    vmin : float
        The minimum value of the scalar range.
    vmax : float
        The maximum value of the scalar range.
    """

    # Bound the value
    value = min(max(value, vmin), vmax)

    # Get the index
    index = int((value - vmin) / (vmax - vmin) * (color_map_array.shape[0] - 1))

    # Set the color
    color = (
        color_map_array[index, 0],
        color_map_array[index, 1],
        color_map_array[index, 2],
        color_map_array[index, 3],
    )
    return color

@cuda.jit
def blend_colors(cur_color, pre_color):
    """Blend two colors.

    Parameters
    ----------
    cur_color : tuple
        The first color. Forth value is alpha.
    pre_color : tuple
        The second color. Forth value is alpha.
    """

    # Blend the colors
    color = (
        cur_color[0] * cur_color[3] + pre_color[0] * pre_color[3] * (1 - cur_color[3]),
        cur_color[1] * cur_color[3] + pre_color[1] * pre_color[3] * (1 - cur_color[3]),
        cur_color[2] * cur_color[3] + pre_color[2] * pre_color[3] * (1 - cur_color[3]),
        cur_color[3] + pre_color[3] * (1 - cur_color[3]),
    )
    return color
