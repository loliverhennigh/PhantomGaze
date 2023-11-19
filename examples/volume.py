# Simple example of rendering a volume using phantomgaze

import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm

import phantomgaze as pg

if __name__ == "__main__":
    # Create sin feild to make volume plot from
    X = cp.linspace(-1, 1, 256)
    Y = cp.linspace(-1, 1, 256)
    Z = cp.linspace(-1, 1, 256)
    X, Y, Z = cp.meshgrid(X, Y, Z, indexing="ij")
    sin_volume = pg.objects.Volume(
        cp.sin(X * 2 * cp.pi) * cp.sin(Y * 2 * cp.pi) * cp.sin(Z * 2 * cp.pi),
        spacing=(2 / 256, 2 / 256, 2 / 256),
        origin=(-1.0, -1.0, -1.0),
    )

    # Create color map
    colormap = pg.Colormap("jet", vmin=0.0, vmax=1.0, opacity=0.5)

    # Create camera object
    camera = pg.Camera(position=(0.0, 1.5, 6.0), focal_point=(0.0, 0.0, 0.0), view_up=(0.0, 1.0, 0.0))

    # Create screen buffer
    screen_buffer = pg.ScreenBuffer.from_camera(camera)
   
    for _ in tqdm(range(100)):
        # Render the axes
        screen_buffer = pg.render.axes(size=0.1, center=(-1.5, -1.5, 1.0), camera=camera, screen_buffer=screen_buffer)

        # Render wireframe of the volume
        screen_buffer = pg.render.wireframe(lower_bound=(-1.0, -1.0, -1.0), upper_bound=(1.0, 1.0, 1.0), thickness=2 / 256, camera=camera, screen_buffer=screen_buffer)

        # Render the volume plot
        screen_buffer = pg.render.volume(sin_volume, camera, colormap=colormap, screen_buffer=screen_buffer)

    # Render the axes
    screen_buffer = pg.render.axes(size=0.1, center=(-1.5, -1.5, 1.0), camera=camera)

    # Render wireframe of the volume
    screen_buffer = pg.render.wireframe(lower_bound=(-1.0, -1.0, -1.0), upper_bound=(1.0, 1.0, 1.0), thickness=2 / 256, camera=camera, screen_buffer=screen_buffer)

    # Render the volume plot
    screen_buffer = pg.render.volume(sin_volume, camera, colormap=colormap, screen_buffer=screen_buffer)

    # Show the rendered image
    plt.imshow(screen_buffer.image.get())
    plt.show()
