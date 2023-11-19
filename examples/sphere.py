# Simple example of rendering a contour of a sphere using phantomgaze


import cupy as cp
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

    # Creat color volume
    color_volume = pg.objects.Volume(
        cp.sin(X * 2 * cp.pi) * cp.sin(Y * 2 * cp.pi) * cp.sin(Z * 2 * cp.pi),
        spacing=(2 / 256, 2 / 256, 2 / 256),
        origin=(-1.0, -1.0, -1.0),
    )
    sphere_volume_color = pg.Colormap("jet", vmin=-1.0, vmax=1.0, opacity=np.linspace(0.0, 1.0, 256))

    # Create solid sphere object
    solid_sphere = pg.objects.Sphere(radius=0.2)
    solid_sphere_color = pg.SolidColor(color=(1.0, 0.0, 1.0), opacity=1.0)

    # Create box frame object
    box_frame = pg.objects.BoxFrame(lower_bound=(-1.0, -1.0, -1.0), upper_bound=(1.0, 1.0, 1.0), thickness=0.05)
    box_frame_color = pg.SolidColor(color=(0.0, 1.0, 0.0), opacity=0.5)

    # Colormap
    volume_cmap = pg.Colormap("copper_r", vmin=-1.0, vmax=1.0)

    # Create camera object
    camera = pg.Camera(position=(0.0, 0.0, -4.0), focal_point=(0.0, 0.0, 0.0), view_up=(0.0, 1.0, 0.0))

    # Render the volume using the raymarching algorithm in phantomgaze
    def render_frame(angle, sphere_volume, color_volume):
        # Calculate the camera's x and z positions for the given angle
        radius = 4.0  # distance of the camera from the focal point
        x = radius * np.cos(np.radians(angle))
        z = radius * np.sin(np.radians(angle))
    
        # Update camera position
        camera.position = (x, 0.0, z)
    
        # Render the contour of the sphere SDF
        screen_buffer = pg.render.contour(sphere_volume, camera, threshold=-0.75, color=color_volume, colormap=sphere_volume_color)

        # Render the solid sphere
        screen_buffer = pg.render.geometry(solid_sphere, camera, color=solid_sphere_color, screen_buffer=screen_buffer)

        # Render the box frame
        screen_buffer = pg.render.geometry(box_frame, camera, color=box_frame_color, screen_buffer=screen_buffer)

        return screen_buffer.image.get(), screen_buffer.depth_buffer.get()
    
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
