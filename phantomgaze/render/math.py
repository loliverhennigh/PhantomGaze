# Simple math utilities

import numba
from numba import cuda

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
