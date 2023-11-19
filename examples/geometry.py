# Simple example of rendering a sphere and a box frame with phantomgaze

import matplotlib.pyplot as plt
import math

import phantomgaze as pg

if __name__ == "__main__":

    # Create sphere
    sphere = pg.objects.Sphere(radius=1.0)

    # Create box frame
    box_frame = pg.objects.BoxFrame(lower_bound=(-1.0, -1.0, -1.0), upper_bound=(1.0, 1.0, 1.0), thickness=0.1)

    # Create camera object
    camera = pg.Camera(position=(2.0, 1.0, -4.0), focal_point=(0.0, 0.0, 0.0), view_up=(0.0, 1.0, 0.0))

    # Render the sphere
    color = pg.SolidColor(color=(0.0, 1.0, 0.0), opacity=1.0)
    screen_buffer = pg.render.geometry(sphere, camera, color=color)

    # Render the box frame as transparent
    color = pg.SolidColor(color=(1.0, 0.0, 1.0), opacity=0.5)
    screen_buffer = pg.render.geometry(box_frame, camera, color=color, screen_buffer=screen_buffer)

    # Plot the result
    plt.imshow(screen_buffer.image.get())
    plt.show()
