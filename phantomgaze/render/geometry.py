# Helper functions for geometry

import cupy as cp
import numba
from numba import cuda

from phantomgaze.render.math import normalize, dot, cross


@cuda.jit(device=True)
def ray_intersect_box(
        box_origin,
        box_upper,
        ray_origin,
        ray_direction):
    """Compute the intersection of a ray with a box.

    Parameters
    ----------
    box_origin : tuple
        The origin of the box
    box_upper : tuple
        The upper bounds of the box.
    ray_origin : tuple
        The origin of the ray.
    ray_direction : tuple
        The direction of the ray.
    """

    # Get tmix and tmax
    tmin_x = (box_origin[0] - ray_origin[0]) / ray_direction[0]
    tmax_x = (box_upper[0] - ray_origin[0]) / ray_direction[0]
    tmin_y = (box_origin[1] - ray_origin[1]) / ray_direction[1]
    tmax_y = (box_upper[1] - ray_origin[1]) / ray_direction[1]
    tmin_z = (box_origin[2] - ray_origin[2]) / ray_direction[2]
    tmax_z = (box_upper[2] - ray_origin[2]) / ray_direction[2]

    # Get tmin and tmax
    tmmin_x = min(tmin_x, tmax_x)
    tmmax_x = max(tmin_x, tmax_x)
    tmmin_y = min(tmin_y, tmax_y)
    tmmax_y = max(tmin_y, tmax_y)
    tmmin_z = min(tmin_z, tmax_z)
    tmmax_z = max(tmin_z, tmax_z)

    # Get t0 and t1
    t0 = max(0.0, max(tmmin_x, max(tmmin_y, tmmin_z)))
    t1 = min(tmmax_x, min(tmmax_y, tmmax_z))

    # Return the intersection
    return t0, t1


@cuda.jit(device=True)
def intersect_ray_with_line(ray_origin, ray_direction, line_start, line_end):
    """
    Check if the ray intersects with the line segment and return the distance
    if it does.

    Parameters
    ----------
    ray_origin : tuple
        The origin point of the ray.
    ray_direction : tuple
        The normalized direction vector of the ray.
    line_start : tuple
        The start point of the line segment.
    line_end : tuple
        The end point of the line segment.

    Returns
    -------
    tuple
        (bool, float) indicating if there is an intersection and the distance.
    """
    # Calculate vectors
    v1 = (
        ray_origin[0] - line_start[0],
        ray_origin[1] - line_start[1],
        ray_origin[2] - line_start[2]
    )
    v2 = (
        line_end[0] - line_start[0],
        line_end[1] - line_start[1],
        line_end[2] - line_start[2]
    )
    v3 = (
        -ray_direction[1],
        ray_direction[0],
        0.0
    )

    # Calculate the cross product
    v2_cross_v3 = cross(v2, v3)

    # Compute normal
    v2_cross_v3_norm = (v2_cross_v3[0] ** 2 + v2_cross_v3[1] ** 2 + v2_cross_v3[2] ** 2) ** 0.5

    # Check if the line and the ray are parallel
    if v2_cross_v3_norm < 1e-5:
        return False, cp.inf

    # Calculate the t and u parameters for intersection
    t = dot(v1, v3) / dot(v2, v3)
    u = cp.dot(v1, v2_cross_v3) / cp.linalg.norm(v2_cross_v3)

    # Check if the intersection is within the line segment and in the ray's direction
    if 0 <= t <= 1 and u >= 0:
        # Calculate the intersection distance
        intersection_point = line_start + t * v2
        distance = cp.linalg.norm(ray_origin - intersection_point)
        return True, distance

    return False, cp.inf
