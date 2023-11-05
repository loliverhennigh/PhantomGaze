# Simple example of rendering a contour of a sphere using phantomgaze

import phantomgaze as pg
import phantomgaze.render
import cupy as cp
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == "__main__":
    # Create SDF feild of a sphere using cupy
    X = cp.linspace(-1, 1, 256)
    Y = cp.linspace(-1, 1, 256)
    Z = cp.linspace(-1, 1, 256)
    X, Y, Z = cp.meshgrid(X, Y, Z, indexing="ij")
    sphere = -(cp.sqrt(X**2 + Y**2 + Z**2) - 0.5)

    # Create volume object from the SDF field (zero copy)
    sphere_volume = pg.Volume(
        sphere, spacing=(2 / 256, 2 / 256, 2 / 256), origin=(-1.0, -1.0, -1.0)
    )

    # Creat color volume
    color_volume = pg.Volume(
        cp.sin(X * 2 * cp.pi) * cp.sin(Y * 2 * cp.pi) * cp.sin(Z * 2 * cp.pi),
        spacing=(2 / 256, 2 / 256, 2 / 256),
        origin=(-1.0, -1.0, -1.0),
    )

    # Create camera object
    camera = pg.Camera(position=(0.0, 0.0, -4.0), focal_point=(0.0, 0.0, 0.0), view_up=(0.0, 1.0, 0.0))

    # Colormap
    cmap = pg.Colormap("copper_r", vmin=-1.0, vmax=1.0)

    # Render the volume using the raymarching algorithm in phantomgaze
    def render_frame(angle, sphere_volume, color_volume):
        # Calculate the camera's x and z positions for the given angle
        radius = 4.0  # distance of the camera from the focal point
        x = radius * np.cos(np.radians(angle))
        z = radius * np.sin(np.radians(angle))
    
        # Update camera position
        camera.position = (x, 0.0, z)
    
        # Render the contour of the sphere
        img, depth = phantomgaze.render.contour(sphere_volume, camera, threshold=-0.75, color=color_volume, opacity=1.0)

        # Render wireframe of the bounding box
        img, depth = phantomgaze.render.wireframe_box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), camera, img=img, depth=depth)

        # Render the volume of the color volume
        img, depth = phantomgaze.render.volume(color_volume, camera, opacity=0.01, colormap=cmap, img=img, depth=depth)

        return img.get(), depth.get()
    
    # Create a figure for plotting
    fig, ax = plt.subplots(1, 2)
    
    # Function to update the frames in the animation
    def update_frame(i):
        angle = i * 10  # Change the multiplier for faster/slower rotation
        img, depth = render_frame(angle, sphere_volume, color_volume)
        ax[0].cla()
        l = ax[0].imshow(img)
        ax[1].cla()
        l = ax[1].imshow(depth)
        return ax
    
    ani = animation.FuncAnimation(fig, update_frame, frames=np.arange(0, 2.0 * np.pi, 0.1))
    plt.show()
