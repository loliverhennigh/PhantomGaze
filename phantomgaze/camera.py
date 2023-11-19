# Camera class

from phantomgaze.background import SolidBackground

class Camera:
    """
    This class represents a camera in a 3D scene.

    Parameters
    ----------
    position : tuple
        The position of the camera in the scene.
    focal_point : tuple
        The point that the camera is looking at.
    view_up : tuple
        The up vector of the camera.
    height : int
        The height of the camera image.
    width : int
        The width of the camera image.
    max_depth : float
        The maximum depth of the camera.
    background : phantomgaze.background.Background
        The background of the camera.
    """

    def __init__(
            self,
            position=(0, 0, 6.69),
            focal_point=(0, 0, 0),
            view_up=(0, 1, 0),
            height=960,
            width=1280,
            max_depth=10.0,
            background=SolidBackground(color=(0.4, 0.4, 0.55)) # Paraview default
            ):

        self.position = position
        self.focal_point = focal_point
        self.view_up = view_up
        self.height = height
        self.width = width
        self.max_depth = max_depth
        self.background = background
