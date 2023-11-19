# Renders Axis

import math

from phantomgaze import ScreenBuffer
from phantomgaze import SolidColor
from phantomgaze.objects import Arrow 
from phantomgaze.render import geometry

# Store the axes Geometries for later use
_axes = {}

def axes(size, center, camera, screen_buffer=None):
    """
    Renders an axes with the given size and center
    x-axes is red
    y-axes is yellow
    z-axes is green

    Parameters
    ----------
    size : float
        Size of the axes
    center : tuple
        Center of the axes
    camera : Camera
        Camera to render from
    screen_buffer : ScreenBuffer
        ScreenBuffer to render the axes to. If None, a new ScreenBuffer is created
    """

    # Get the screen buffer
    if screen_buffer is None:
        screen_buffer = ScreenBuffer.from_camera(camera)

    # Make the axes, reusing the geometry if possible
    if (size, center) not in _axes:
        x_axes = Arrow(height=size)
        x_axes = x_axes.rotate(-math.pi / 2, (0, 0, 1))
        x_axes = x_axes.translate((center[0] + size, center[1], center[2]))
        x_color = SolidColor(color=(1.0, 0.0, 0.0))
        y_axes = Arrow(height=size)
        y_axes = y_axes.rotate(math.pi, (0, 0, 1))
        y_axes = y_axes.translate((center[0], center[1] + size, center[2]))
        y_color = SolidColor(color=(1.0, 1.0, 0.0))
        z_axes = Arrow(height=size)
        z_axes = z_axes.rotate(math.pi / 2, (1, 0, 0))
        z_axes = z_axes.translate((center[0], center[1], center[2] + size))
        z_color = SolidColor(color=(0.0, 1.0, 0.0))
        _axes[(size, center)] = (x_axes, x_color, y_axes, y_color, z_axes, z_color)
    else:
        x_axes, x_color, y_axes, y_color, z_axes, z_color = _axes[(size, center)]

    # Render the axes
    geometry(x_axes, camera, x_color, screen_buffer)
    geometry(y_axes, camera, y_color, screen_buffer)
    geometry(z_axes, camera, z_color, screen_buffer)

    return screen_buffer
