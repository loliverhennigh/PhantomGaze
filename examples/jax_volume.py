# Simple example of rendering a volume given a jax array

import jax.numpy as jnp
import matplotlib.pyplot as plt

import phantomgaze as pg

if __name__ == "__main__":
    # Create sin feild to make volume plot from
    X = jnp.linspace(-1, 1, 256)
    Y = jnp.linspace(-1, 1, 256)
    Z = jnp.linspace(-1, 1, 256)
    X, Y, Z = jnp.meshgrid(X, Y, Z, indexing="ij")
    sin_volume = pg.objects.Volume(
        jnp.sin(X * 2 * jnp.pi) * jnp.sin(Y * 2 * jnp.pi) * jnp.sin(Z * 2 * jnp.pi),
        spacing=(2 / 256, 2 / 256, 2 / 256),
        origin=(-1.0, -1.0, -1.0),
    )

    # Create color map
    colormap = pg.Colormap("jet", vmin=0.0, vmax=1.0, opacity=0.5)

    # Create camera object
    camera = pg.Camera(position=(0.0, 1.5, 6.0), focal_point=(0.0, 0.0, 0.0), view_up=(0.0, 1.0, 0.0))
    
    # Render the volume plot
    screen_buffer = pg.render.volume(sin_volume, camera, colormap=colormap, screen_buffer=screen_buffer)

    # Show the rendered image
    plt.imshow(screen_buffer.image.get())
    plt.show()
