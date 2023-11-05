# Sign distance function of different geometries

def sphere(x, y, z, radius):
    """
    Sign distance function of a sphere

    Parameters
    ----------
    x : float
        x-coordinate of the point
    y : float
        y-coordinate of the point
    z : float
        z-coordinate of the point
    radius : float
        radius of the sphere
    """

    return np.sqrt(x**2 + y**2 + z**2) - radius

def cube(x, y, z, length):
    """
    Sign distance function of a cube

    Parameters
    ----------
    x : float
        x-coordinate of the point
    y : float
        y-coordinate of the point
    z : float
        z-coordinate of the point
    length : float
        length of the cube
    """

    return np.max(np.abs([x, y, z])) - length

def cylinder(x, y, z, radius, length):
    """
    Sign distance function of a cylinder

    Parameters
    ----------
    x : float
        x-coordinate of the point
    y : float
        y-coordinate of the point
    z : float
        z-coordinate of the point
    radius : float
        radius of the cylinder
    length : float
        length of the cylinder
    """

    return np.sqrt(x**2 + y**2) - radius




