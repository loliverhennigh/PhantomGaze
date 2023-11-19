# Colormap class for plots

import cupy as cp 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass

@dataclass
class Coloring:
    """
    Basic coloring class for plots.

    Parameters
    ----------
    vmin : float
        The minimum value of the colormap.
    vmax : float
        The maximum value of the colormap.
    color_map_array : cp.array
        The array of colors for the colormap.
    nan_color : tuple
        The color for NaN values.
    nan_opacity : float
        The opacity for NaN values.
    opaque : bool
        Whether the geometry is opaque or not, by default True
    """
    vmin: float
    vmax: float
    color_map_array: cp.array
    nan_color: tuple
    nan_opacity: float = 1.0
    opaque: bool = True


class Colormap(Coloring):
    """A colormap class for plots.

    Parameters
    ----------
    name : str
        The name of the colormap using matplotlib's naming convention.
    vmin : float
        The minimum value of the colormap.
    vmax : float
        The maximum value of the colormap.
    num_table_values : int
        The number of values in the colormap table.
    opacity : cp.array, float, optional
        The opacity array for the colormap. If None is given, then the
        colormap is opaque. If an array is given, then the colormap uses
        the array as the opacity.
    nan_color : tuple
        The color for NaN values.
    """

    def __init__(
            self,
            name='jet',
            vmin=0.0,
            vmax=1.0,
            num_table_values=256,
            opacity=None,
            nan_color=(1.0, 1.0, 0.0),
            nan_opacity=1.0,
            ):

        """Initialize the colormap."""
        self.name = name
        self.vmin = vmin
        self.vmax = vmax
        self.num_table_values = num_table_values
        self.nan_color = nan_color
        self.nan_opacity = nan_opacity

        # Get the colormap
        self.cmap = cm.get_cmap(name, num_table_values)
        self.color_map_array = cp.array([self.cmap(i) for i in range(num_table_values)])

        # Set the opacity
        if (opacity is None):
            self.opaque = True
        elif isinstance(opacity, float) and opacity == 1.0:
            self.opaque = True
        elif isinstance(opacity, float) and opacity < 1.0:
            self.opaque = False
            self.color_map_array[:, 3] = opacity
        elif isinstance(opacity, (list, tuple, cp.ndarray, np.ndarray)):
            self.opaque = False
            self.color_map_array[:, 3] = cp.array(opacity)
        else:
            raise TypeError('Invalid opacity type.')

class SolidColor(Coloring):
    """A coloring class for solid colors.
    TODO: Find a better abstraction for this.

    Parameters
    ----------
    color : tuple
        The color for the solid color.
    opacity : float
        The opacity for the solid color.
    """

    def __init__(
            self,
            color=(1.0, 1.0, 1.0),
            opacity=1.0,
            ):
        self.vmin = 0.0 # Not used
        self.vmax = 1.0
        self.color_map_array = cp.array([[color[0], color[1], color[2], opacity]])
        self.nan_color = color # Not used
        self.nan_opacity = 1.0
        if opacity == 1.0:
            self.opaque = True
        else:
            self.opaque = False
