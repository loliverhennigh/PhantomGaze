# Simple example of rendering a sphere and a box frame with phantomgaze

import matplotlib.pyplot as plt
import math
from tqdm import tqdm

import phantomgaze as pg

if __name__ == "__main__":

    # Create camera object
    camera = pg.Camera(position=(0.0, 1.0, 6.69), focal_point=(0.0, 0.0, 0.0), view_up=(0.0, 1.0, 0.0))

    # Render the axes
    screen_buffer = pg.render.axes(size=1.0, center=(1.0, 0.0, 0.0), camera=camera)

    # Plot the result
    plt.imshow(screen_buffer.image.get())
    plt.show()
