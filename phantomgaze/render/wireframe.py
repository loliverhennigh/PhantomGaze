# Renders Wireframe

from phantomgaze import ScreenBuffer
from phantomgaze import SolidColor
from phantomgaze.objects import BoxFrame
from phantomgaze.render import geometry

# Store the wireframe geometry
_axes = {}

def wireframe(
        lower_bound,
        upper_bound,
        thickness,
        camera,
        color=SolidColor(),
        screen_buffer=None):
    """
    Renders a wireframe.

    Parameters
    ----------
    lower_bound : tuple
        The lower bound of the wireframe.
    upper_bound : tuple
        The upper bound of the wireframe.
    thickness : float
        The thickness of the wireframe.
    camera : phantomgaze.Camera
        The camera object.
    color : phantomgaze.SolidColor
        The color of the wireframe.
    screen_buffer : phantomgaze.ScreenBuffer
        The screen buffer to render the wireframe to.
    """

    # Get the screen buffer
    if screen_buffer is None:
        screen_buffer = ScreenBuffer.from_camera(camera)

    # Get the wireframe geometry
    if (lower_bound, upper_bound, thickness) not in _axes:
        box_frame = BoxFrame(
            lower_bound,
            upper_bound,
            thickness,
            )
        _axes[(lower_bound, upper_bound, thickness)] = box_frame
    else:
        box_frame = _axes[(lower_bound, upper_bound, thickness)]

    # Render the wireframe
    geometry(box_frame, camera, color, screen_buffer)

    return screen_buffer


