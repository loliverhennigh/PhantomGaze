# Camera class

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
    """

    def __init__(
            self,
            position=(0, 0, 6.69),
            focal_point=(0, 0, 0),
            view_up=(0, 1, 0),
            height=480,
            width=640,
            ): # TODO Add more parameters if needed
        self.position = position
        self.focal_point = focal_point
        self.view_up = view_up
        self.height = height
        self.width = width
