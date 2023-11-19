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
