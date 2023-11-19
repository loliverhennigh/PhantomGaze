# Simple math utilities

import numba
from numba import cuda

@cuda.jit(device=True)
def clamp(value, min_value, max_value):
    """Clamp a value between a minimum and maximum value.

    Parameters
    ----------
    value : float
        The value to clamp.
    min_value : float
        The minimum value.
    max_value : float
        The maximum value.

    Returns
    -------
    float
        The clamped value.
    """

    # Clamp the value
    return max(min(value, max_value), min_value)

@cuda.jit(device=True)
def sign(value):
    """Get the sign of a value.

    Parameters
    ----------
    value : float
        The value to get the sign of.

    Returns
    -------
    float
        The sign of the value.
    """

    # Get the sign of the value
    if value < 0:
        return -1
    else:
        return 1

@cuda.jit(device=True)
def length(vector):
    """Compute the length of a vector.

    Parameters
    ----------
    vector : tuple
        The vector to compute the length of.

    Returns
    -------
    float
        The length of the vector.
    """

    # Compute the length of the vector
    if len(vector) == 2:
        return (vector[0] ** 2 + vector[1] ** 2) ** 0.5
    else:
        return (vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2) ** 0.5

@cuda.jit(device=True)
def normalize(vector):
    """Normalize a vector.

    Parameters
    ----------
    vector : tuple
        The vector to normalize.

    Returns
    -------
    tuple
        The normalized vector.
    """

    # Get the length of the vector
    length = (vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2) ** 0.5

    # Normalize the vector
    return vector[0] / length, vector[1] / length, vector[2] / length

@cuda.jit(device=True)
def dot(vector1, vector2):
    """Compute the dot product of two vectors.

    Parameters
    ----------
    vector1 : tuple
        The first vector.
    vector2 : tuple
        The second vector.

    Returns
    -------
    float
        The dot product of the two vectors.
    """

    # Compute the dot product
    if len(vector1) == 2:
        return vector1[0] * vector2[0] + vector1[1] * vector2[1]
    else:
        return vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]

@cuda.jit(device=True)
def cross(vector1, vector2):
    """Compute the cross product of two vectors.

    Parameters
    ----------
    vector1 : tuple
        The first vector.
    vector2 : tuple
        The second vector.

    Returns
    -------
    tuple
        The cross product of the two vectors.
    """

    # Compute the cross product
    return (vector1[1] * vector2[2] - vector1[2] * vector2[1],
            vector1[2] * vector2[0] - vector1[0] * vector2[2],
            vector1[0] * vector2[1] - vector1[1] * vector2[0])


@cuda.jit(device=True)
def quaternion_multiply(q1, q2):
    """Multiply two quaternions.

    Parameters
    ----------
    q1 : tuple
        The first quaternion.
    q2 : tuple
        The second quaternion.
    """

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    )
