# Background class for rendering
# TODO: Add gradient background

class Background:
    """ Base class for background """

class SolidBackground(Background):
    """ Solid background """

    def __init__(self, color=(0.0, 0.0, 0.0)):
        """ Initialize the background

        Parameters
        ----------
        color : tuple
            RGB color of the background
        """

        self.color = color
