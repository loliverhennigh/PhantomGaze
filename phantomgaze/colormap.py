# Colormap class for plots

import numpy as cp 
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Colormap:
    """A colormap class for plots.

    Parameters
    ----------
    name : str
        The name of the colormap using matplotlib's naming convention.
    range : tuple
        The range of the colormap.
    """

    def __init__(self, name, vmin, vmax, num_colors=256):
        """Initialize the colormap."""
        self.name = name
        self.vmin = vmin
        self.vmax = vmax
        self.num_colors = num_colors

        # Get the colormap
        self.cmap = cm.get_cmap(name, num_colors)
        self.color_map_array = cp.array([self.cmap(i) for i in range(num_colors)])
