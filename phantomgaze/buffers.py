# Class that stores fragment information for rendering

import cupy as cp
import numba
from numba import cuda

# Make kernel for combining buffers
@cuda.jit
def _combine_buffers_kernel(
        opaque_pixel_buffer,
        depth_buffer,
        normal_buffer,
        transparent_pixel_buffer,
        revealage_buffer,
        background_buffer,
        image_buffer):

    # Get the x and y indices
    x, y = cuda.grid(2)

    # Make sure the indices are in bounds
    if x >= image_buffer.shape[1] or y >= image_buffer.shape[0]:
        return

    # If the depth is not infinite, then the pixel is opaque
    if depth_buffer[y, x] != cp.inf:
        # Get the color from the opaque pixel buffer
        final_color = (
            opaque_pixel_buffer[y, x, 0],
            opaque_pixel_buffer[y, x, 1],
            opaque_pixel_buffer[y, x, 2]
        )
    else:
        # Initialize from background
        final_color = (
            background_buffer[y, x, 0],
            background_buffer[y, x, 1],
            background_buffer[y, x, 2]
        )

    # Blend with revealage
    alpha = (1.0 - revealage_buffer[y, x])
    transparent_color = (
        transparent_pixel_buffer[y, x, 0],
        transparent_pixel_buffer[y, x, 1],
        transparent_pixel_buffer[y, x, 2]
    )
    final_color = (
        final_color[0] * (1.0 - alpha) + transparent_color[0] * alpha,
        final_color[1] * (1.0 - alpha) + transparent_color[1] * alpha,
        final_color[2] * (1.0 - alpha) + transparent_color[2] * alpha
    )

    # Write to the image buffer (flip y axis TODO: fix this)
    flip_y = image_buffer.shape[0] - y - 1
    image_buffer[flip_y, x, 0] = final_color[0]
    image_buffer[flip_y, x, 1] = final_color[1]
    image_buffer[flip_y, x, 2] = final_color[2]
    image_buffer[flip_y, x, 3] = 1.0 


class ScreenBuffer:
    """
    Create a screen buffer.
    The screen buffer stores fragment information for rendering.

    Parameters
    ----------
    height : int
        The height of the screen buffer
    width : int
        The width of the screen buffer
    """

    def __init__(self, height, width):
        # Store height and width
        self.height = height
        self.width = width

        # Create buffers for opaque rendering
        self.opaque_pixel_buffer = cp.zeros((height, width, 3), dtype=cp.float32)
        self.depth_buffer = cp.zeros((height, width), dtype=cp.float32) + cp.inf
        self.normal_buffer = cp.zeros((height, width, 3), dtype=cp.float32)

        # Create buffer transparent rendering
        self.transparent_pixel_buffer = cp.zeros((height, width, 3), dtype=cp.float32)
        self.revealage_buffer = cp.ones((height, width), dtype=cp.float32)

        # Create buffer for background
        self.background_buffer = cp.zeros((height, width, 3), dtype=cp.float32)

        # Create buffer for final image
        self.image_buffer = cp.zeros((height, width, 4), dtype=cp.float32)

    @staticmethod
    def from_camera(camera):
        """ Create a screen buffer from a camera

        Parameters
        ----------
        camera : Camera
            The camera to create the screen buffer for, uses the camera's height and width
        """

        # Make screen buffer
        screen_buffer = ScreenBuffer(camera.height, camera.width)

        # Set background
        screen_buffer.background_buffer[:, :, 0] = camera.background.color[0]
        screen_buffer.background_buffer[:, :, 1] = camera.background.color[1]
        screen_buffer.background_buffer[:, :, 2] = camera.background.color[2]

        return screen_buffer

    @property
    def image(self):
        """ Get the image buffer """

        # Run the kernel
        threads_per_block = (16, 16)
        blocks_per_grid = (
            (self.width + threads_per_block[0] - 1) // threads_per_block[0],
            (self.height + threads_per_block[1] - 1) // threads_per_block[1]
        )
        _combine_buffers_kernel[blocks_per_grid, threads_per_block](
            self.opaque_pixel_buffer,
            self.depth_buffer,
            self.normal_buffer,
            self.transparent_pixel_buffer,
            self.revealage_buffer,
            self.background_buffer,
            self.image_buffer
        )
        return self.image_buffer

    def clear(self):
        """ Clear the screen buffer """

        # Clear opaque buffers
        self.opaque_pixel_buffer.fill(0.0)
        self.depth_buffer.fill(cp.inf)
        self.normal_buffer.fill(0.0)

        # Clear transparent buffers
        self.transparent_pixel_buffer.fill(0.0)
        self.revealage_buffer.fill(1.0)

        # Clear background buffer
        self.background_buffer.fill(0.0)

        # Clear image buffer
        self.image_buffer.fill(0.0)
