# Simple example of rendering a contour of a sphere using phantomgaze

import cupy as cp
import matplotlib.pyplot as plt

import phantomgaze as pg

if __name__ == "__main__":
    # Create SDF feild of a sphere using cupy
    X = cp.linspace(-1, 1, 256)
    Y = cp.linspace(-1, 1, 256)
    Z = cp.linspace(-1, 1, 256)
    X, Y, Z = cp.meshgrid(X, Y, Z, indexing="ij")
    sphere = -(cp.sqrt(X**2 + Y**2 + Z**2) - 0.5)
    sphere_volume = pg.objects.Volume(
        sphere, spacing=(2 / 256, 2 / 256, 2 / 256), origin=(-1.0, -1.0, -1.0)
    )

    # Create color volume
    color_volume = pg.objects.Volume(
        cp.sin(X * 2 * cp.pi) * cp.sin(Y * 2 * cp.pi) * cp.sin(Z * 2 * cp.pi),
        spacing=(2 / 256, 2 / 256, 2 / 256),
        origin=(-1.0, -1.0, -1.0),
    )

    # Create color map
    colormap = pg.Colormap("Accent", vmin=-1.0, vmax=1.0, opacity=cp.linspace(0.0, 1.0, 256))

    # Create camera object
    camera = pg.Camera(position=(2.0, 1.0, -4.0), focal_point=(0.0, 0.0, 0.0), view_up=(0.0, 1.0, 0.0))

    # Render the contour of the inner sphere
    screen_buffer = pg.render.contour(sphere_volume, camera, threshold=0.25, color=color_volume)

    # Render the outer sphere with transparent color map
    screen_buffer = pg.render.contour(sphere_volume, camera, threshold=-0.75, color=color_volume, colormap=colormap, screen_buffer=screen_buffer)

    # Show the rendered image
    plt.imshow(screen_buffer.image.get())
    plt.show()
