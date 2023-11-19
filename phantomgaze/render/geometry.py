# functions for rendering geometry

import cupy as cp
import numba
from numba import cuda

from phantomgaze import ScreenBuffer
from phantomgaze import SolidColor
from phantomgaze.objects import Geometry
from phantomgaze.utils.math import normalize, dot
from phantomgaze.render.camera import calculate_ray_direction

_geometry_render_kernels = {}

def kernel_constructor_render_geometry(sdf, sdf_derivative, opaque=True):
    """
    Constructs a kernel the renders a signed distance function.
    TODO: probably make decorator

    Parameters
    ----------
    sdf : function
        The signed distance function to render. Must be a numba.jit(device=True) function.
    sdf_derivative : function
        The signed distance function derivative. Must be a numba.jit(device=True) function.
    opaque : bool, optional
        Whether the geometry is opaque or not, by default True

    Returns
    -------
    function
        The kernel that renders the signed distance function.
        Outputs: (img, depth)
    """

    # Check if the kernel has already been constructed
    if (sdf, sdf_derivative, opaque) in _geometry_render_kernels:
        return _geometry_render_kernels[(sdf, sdf_derivative, opaque)]

    # Define the kernel
    @cuda.jit
    def render_kernel(
            distance_threshold,
            camera_position,
            camera_focal,
            camera_up,
            max_depth,
            color_map_array,
            opaque_pixel_buffer,
            depth_buffer,
            normal_buffer,
            transparency_pixel_buffer,
            revealage_buffer,
            ):
        """
        Renders a signed distance function.

        Parameters
        ----------
        distance_threshold : float, optional
            The distance to check for intersection
        camera_position : tuple
            The position of the camera.
        camera_focal : tuple
            The focal point of the camera.
        camera_up : tuple
            The up vector of the camera.
        max_depth : float
            The maximum depth to render.
        color_map_array : cp.array
            The array of colors for the colormap. [r, g, b, a]
        opaque_pixel_buffer : ndarray
            The opaque pixel buffer.
        depth_buffer : ndarray
            The depth buffer.
        normal_buffer : ndarray
            The normal buffer.
        transparency_pixel_buffer : ndarray
            The transparency pixel buffer.
        revealage_buffer : ndarray
            The revealage buffer.
        """

        # Get the x and y indices
        x, y = cuda.grid(2)
    
        # Make sure the indices are in bounds
        if x >= opaque_pixel_buffer.shape[1] or y >= opaque_pixel_buffer.shape[0]:
            return
    
        # Get ray direction
        ray_direction = calculate_ray_direction(
                x, y, opaque_pixel_buffer.shape,
                camera_position, camera_focal, camera_up)
    
        # Get the starting point of the ray
        ray_pos = (
            camera_position[0],
            camera_position[1],
            camera_position[2]
        )

        # Distance traveled by the ray
        distance_traveled = 0

        # Ray march
        while distance_traveled < max_depth:
            # Get the distance to the object
            distance = abs(sdf(ray_pos))

            # Update the distance traveled
            distance_traveled += distance

            # Check if distance is greater than current depth
            if distance_traveled > depth_buffer[y, x]:
                return

            # Update the ray position
            ray_pos = (
                ray_pos[0] + ray_direction[0] * distance,
                ray_pos[1] + ray_direction[1] * distance,
                ray_pos[2] + ray_direction[2] * distance
            )

            # Check if the ray is close enough to the object
            if abs(distance) < distance_threshold:
                # Get intensity TODO: Maybe change this
                gradient = sdf_derivative(ray_pos)
                gradient = normalize(gradient)
                intensity = dot(gradient, ray_direction)
                intensity = abs(intensity)

                # Get the color and opacity
                color = (
                    color_map_array[0, 0],
                    color_map_array[0, 1],
                    color_map_array[0, 2]
                )
                opacity = color_map_array[0, 3]

                # If solid, set the pixel buffer (meta programming)
                if opaque:
                    # Set the opaque pixel buffer
                    opaque_pixel_buffer[y, x, 0] = color[0] * intensity
                    opaque_pixel_buffer[y, x, 1] = color[1] * intensity
                    opaque_pixel_buffer[y, x, 2] = color[2] * intensity

                    # Set the depth buffer
                    depth_buffer[y, x] = distance_traveled

                    # Set the normal buffer
                    normal_buffer[y, x, 0] = gradient[0]
                    normal_buffer[y, x, 1] = gradient[1]
                    normal_buffer[y, x, 2] = gradient[2]

                    # Exit the loop
                    return

                # If transparent, set the pixel buffer (meta programming)
                else:
                    # Calculate weight following Weighted Blended OIT
                    normalized_distance = distance_traveled / max_depth
                    weight = 1.0 / (normalized_distance**2.0 + 1.0)
    
                    # Accumulate the color
                    transparency_pixel_buffer[y, x, 0] += color[0] * weight * opacity * intensity
                    transparency_pixel_buffer[y, x, 1] += color[1] * weight * opacity * intensity
                    transparency_pixel_buffer[y, x, 2] += color[2] * weight * opacity * intensity
    
                    # Set the revealage buffer
                    revealage_buffer[y, x] *= (1.0 - opacity * weight)
    
                    # Take small steps to pass through the object
                    while distance < distance_threshold:
    
                        # Take a small step
                        ray_pos = (
                            ray_pos[0] + ray_direction[0] * distance_threshold,
                            ray_pos[1] + ray_direction[1] * distance_threshold,
                            ray_pos[2] + ray_direction[2] * distance_threshold
                        )
    
                        # Update the distance traveled
                        distance_traveled += distance_threshold
    
                        # Get the new signed distance
                        distance = abs(sdf(ray_pos))
            
    # Add the kernel to the dictionary
    _geometry_render_kernels[(sdf, sdf_derivative, opaque)] = render_kernel

    return render_kernel

def geometry(
        geometry,
        camera,
        color=SolidColor(color=(1.0, 1.0, 1.0), opacity=1.0),
        screen_buffer=None):
    """
    Renders a geometry object

    Parameters
    ----------
    geometry : Geometry
        The geometry object to render
    camera : Camera
        The camera object to render with
    color : SolidColor, optional
        The color of the geometry
    screen_buffer : ScreenBuffer, optional
        The screen buffer to render to. If None, a new screen buffer will be created.
    """

    # Get the screen buffer
    if screen_buffer is None:
        screen_buffer = ScreenBuffer.from_camera(camera)

    # Set the block size
    threads_per_block = (16, 16)
    blocks = (
        (screen_buffer.width + threads_per_block[0] - 1) // threads_per_block[0],
        (screen_buffer.height + threads_per_block[1] - 1) // threads_per_block[1]
    )

    # Construct the kernel
    render_kernel = kernel_constructor_render_geometry(geometry.sdf, geometry.derivative, color.opaque)

    # Run the kernel
    render_kernel[blocks, threads_per_block](
        geometry.distance_threshold,
        camera.position,
        camera.focal_point,
        camera.view_up,
        camera.max_depth,
        color.color_map_array,
        screen_buffer.opaque_pixel_buffer,
        screen_buffer.depth_buffer,
        screen_buffer.normal_buffer,
        screen_buffer.transparent_pixel_buffer,
        screen_buffer.revealage_buffer,
    )

    return screen_buffer
