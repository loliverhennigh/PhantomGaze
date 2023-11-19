# Geometries

from numba import cuda
import math

from phantomgaze.utils.math import length, clamp, dot, sign, quaternion_multiply

class Geometry:
    """
    Class that represents a geometry to be rendered
    The geometry is defined by a signed distance function
    This allows for boolean operations to be performed on the geometry

    Parameters
    ----------
    sdf : cuda.jit function
        The signed distance function
    lower_bound : tuple
        Lower bound of the signed distance function
    upper_bound : tuple
        Upper bound of the signed distance function
    distance_threshold : float
        The distance threshold for the signed distance function when rendering
    """

    def __init__(self, sdf, lower_bound, upper_bound, distance_threshold=0.001):
        # Store the parameters
        self.sdf = sdf
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.distance_threshold = distance_threshold

        # Define the kernel for the derivative
        @cuda.jit(device=True)
        def sdf_derivative(pos):
            dx = sdf((pos[0] + 0.001, pos[1], pos[2])) - sdf(
                (pos[0] - 0.001, pos[1], pos[2])
            )
            dy = sdf((pos[0], pos[1] + 0.001, pos[2])) - sdf(
                (pos[0], pos[1] - 0.001, pos[2])
            )
            dz = sdf((pos[0], pos[1], pos[2] + 0.001)) - sdf(
                (pos[0], pos[1], pos[2] - 0.001)
            )
            return (dx, dy, dz)
        self.derivative = sdf_derivative

    def __add__(self, other):
        """
        Adds two signed distance functions together (union)

        Parameters
        ----------
        other : Geometry 
            The other signed distance function

        Returns
        -------
        Geometry
            The new signed distance function
        """

        # Get the signed distance functions
        sdf = self.sdf
        other_sdf = other.sdf

        # Define the new signed distance function
        @cuda.jit(device=True)
        def new_sdf(pos):
            return min(sdf(pos), other_sdf(pos))

        # Expand the bounds
        lower_bound = (
            min(self.lower_bound[0], other.lower_bound[0]),
            min(self.lower_bound[1], other.lower_bound[1]),
            min(self.lower_bound[2], other.lower_bound[2]),
        )
        upper_bound = (
            max(self.upper_bound[0], other.upper_bound[0]),
            max(self.upper_bound[1], other.upper_bound[1]),
            max(self.upper_bound[2], other.upper_bound[2]),
        )

        # Return the new signed distance function
        return Geometry(new_sdf, lower_bound, upper_bound, min(self.distance_threshold, other.distance_threshold))

    def __sub__(self, other):
        """
        Subtracts two signed distance functions together (difference)

        Parameters
        ----------
        other : Geometry
            The other signed distance function

        Returns
        -------
        Geometry
            The new signed distance function
        """

        # Define the new signed distance function
        @cuda.jit(device=True)
        def new_sdf(pos):
            return max(self.sdf(pos), -other.sdf(pos))

        # Return the new signed distance function
        return Geometry(new_sdf, self.lower_bound, self.upper_bound, self.distance_threshold)

    def __and__(self, other):
        """
        Intersects two signed distance functions together (intersection)

        Parameters
        ----------
        other : Geometry
            The other signed distance function

        Returns
        -------
        Geometry
            The new signed distance function
        """

        # Define the new signed distance function
        @cuda.jit(device=True)
        def new_sdf(pos):
            return max(self.sdf(pos), other.sdf(pos))

        # Shrink the bounds
        lower_bound = (
            max(self.lower_bound[0], other.lower_bound[0]),
            max(self.lower_bound[1], other.lower_bound[1]),
            max(self.lower_bound[2], other.lower_bound[2]),
        )
        upper_bound = (
            min(self.upper_bound[0], other.upper_bound[0]),
            min(self.upper_bound[1], other.upper_bound[1]),
            min(self.upper_bound[2], other.upper_bound[2]),
        )

        # Return the new signed distance function
        return Geometry(new_sdf, lower_bound, upper_bound, min(self.distance_threshold, other.distance_threshold))

    def translate(self, translation):
        """
        Translates the signed distance function

        Parameters
        ----------
        translation : tuple
            The translation vector

        Returns
        -------
        Geometry
            The new signed distance function
        """

        # Get the signed distance function
        sdf = self.sdf

        # Define the new signed distance function
        @cuda.jit(device=True)
        def new_sdf(pos):
            return sdf((pos[0] - translation[0], pos[1] - translation[1], pos[2] - translation[2]))

        # Get new bounds
        lower_bound = (
            self.lower_bound[0] + translation[0],
            self.lower_bound[1] + translation[1],
            self.lower_bound[2] + translation[2],
        )
        upper_bound = (
            self.upper_bound[0] + translation[0],
            self.upper_bound[1] + translation[1],
            self.upper_bound[2] + translation[2],
        )

        # Return the new signed distance function
        return Geometry(new_sdf, lower_bound, upper_bound, self.distance_threshold)


    def rotate(self, angle, axis):
        """
        Rotates the signed distance function

        Parameters
        ----------
        angle : float
            The angle to rotate by
        axis : tuple
            The axis to rotate around

        Returns
        -------
        Geometry
            The new signed distance function
        """


        # Compute the rotation quaternion
        half_angle = angle / 2.0
        c = math.cos(half_angle)
        s = math.sin(half_angle)
        q = (c, axis[0] * s, axis[1] * s, axis[2] * s) # rotation quaternion
        q_inv = (c, -axis[0] * s, -axis[1] * s, -axis[2] * s) # inverse of rotation quaternion

        # Get the signed distance function
        sdf = self.sdf

        # Define the new signed distance function
        @cuda.jit(device=True)
        def new_sdf(pos):
            # Convert pos to quaternion form
            pos_q = (0, pos[0], pos[1], pos[2])
    
            # Perform quaternion multiplication: q * pos * q_inv
            pos_rotated = quaternion_multiply(
                quaternion_multiply(q, pos_q),
                q_inv
            )
    
            # Extract the rotated position vector
            return sdf((pos_rotated[1], pos_rotated[2], pos_rotated[3]))

        # Return the new signed distance function
        # TODO: Fix the bounds
        return Geometry(new_sdf, self.lower_bound, self.upper_bound, self.distance_threshold)


class Sphere(Geometry):
    """
    Class that represents a sphere

    Parameters
    ----------
    radius : float
        The radius of the sphere
    center : tuple
        The center of the sphere
    """

    def __init__(self, radius, center=(0.0, 0.0, 0.0)):
        # Define the signed distance function
        @cuda.jit(device=True)
        def sdf(pos):
            return length((pos[0] - center[0], pos[1] - center[1], pos[2] - center[2])) - radius

        # Define the bounds
        lower_bound = (-radius+center[0], -radius+center[1], -radius+center[2])
        upper_bound = (radius+center[0], radius+center[1], radius+center[2])

        # Call the parent constructor
        super().__init__(sdf, lower_bound, upper_bound, distance_threshold=radius / 100.0)


class BoxFrame(Geometry):
    """
    Class that represents a box frame
    Reference: https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

    Parameters
    ----------
    lower_bound : tuple
        lower bound of the box frame
    upper_bound : tuple
        upper bound of the box frame
    thickness : float
        thickness of the box frame
    """

    def __init__(self, lower_bound, upper_bound, thickness):
        # Define the signed distance function
        @cuda.jit(device=True)
        def sdf(pos):
            # Compute size of the box frame
            size = (upper_bound[0] - lower_bound[0], upper_bound[1] - lower_bound[1], upper_bound[2] - lower_bound[2])
        
            # Compute the center of the box frame
            center = (lower_bound[0] + size[0] / 2, lower_bound[1] + size[1] / 2, lower_bound[2] + size[2] / 2)
        
            # Compute the point position relative to the center
            p = (pos[0] - center[0], pos[1] - center[1], pos[2] - center[2])
        
            # Compute the SDF
            p = (abs(p[0]) - size[0] / 2, abs(p[1]) - size[1] / 2, abs(p[2]) - size[2] / 2)
            q = (
                abs(p[0] + thickness) - thickness,
                abs(p[1] + thickness) - thickness,
                abs(p[2] + thickness) - thickness,
            )
            e_x = length((max(p[0], 0), max(q[1], 0), max(q[2], 0))) + min(
                max(p[0], max(q[1], q[2])), 0
            )
            e_y = length((max(q[0], 0), max(p[1], 0), max(q[2], 0))) + min(
                max(q[0], max(p[1], q[2])), 0
            )
            e_z = length((max(q[0], 0), max(q[1], 0), max(p[2], 0))) + min(
                max(q[0], max(q[1], p[2])), 0
            )
        
            return min(min(e_x, e_y), e_z)

        # Call the parent constructor
        super().__init__(sdf, lower_bound, upper_bound, distance_threshold=thickness/100.0)

class Cone(Geometry):
    """
    Class that represents a cone
    Reference: https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

    Parameters
    ----------
    c: tuple
        The (sin, cos) of the angle
    h: float
        The height of the cone
    center: tuple
        The center of the cone
    """

    def __init__(self, c, h, center=(0.0, 0.0, 0.0)):
        # Define the signed distance function
        @cuda.jit(device=True)
        def sdf(pos):
            # Compute the point position relative to the center
            p = (pos[0] - center[0], pos[1] - center[1], pos[2] - center[2])

            # Compute the SDF
            q = (h * c[0] / c[1], -h)
            w = (length((p[0], p[2])), p[1])
            a = (w[0] - q[0] * clamp(dot(w, q) / dot(q, q), 0.0, 1.0), w[1] - q[1] * clamp(dot(w, q) / dot(q, q), 0.0, 1.0))
            b = (w[0] - q[0] * clamp(w[0] / q[0], 0.0, 1.0), w[1] - q[1] * 1.0)
            k = sign(q[1])
            d = min(dot(a, a), dot(b, b))
            s = max(k * (w[0] * q[1] - w[1] * q[0]), k * (w[1] - q[1]))
            return d**0.5 * sign(s)

        # Define the bounds
        lower_bound = (-h + center[0], -h + center[1], -h + center[2])
        upper_bound = (h + center[0], h + center[1], h + center[2])

        # Call the parent constructor
        super().__init__(sdf, lower_bound, upper_bound, distance_threshold=h / 100.0)

class Cylinder(Geometry):
    """
    Class that represents a cylinder
    Reference: https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

    Parameters
    ----------
    radius: float
        The radius of the cylinder
    height: float
        The height of the cylinder
    center: tuple
        The center of the cylinder
    """

    def __init__(self, radius, height, center=(0.0, 0.0, 0.0)):
        # Define the signed distance function
        @cuda.jit(device=True)
        def sdf(pos):
            # Compute the point position relative to the center
            p = (pos[0] - center[0], pos[1] - center[1], pos[2] - center[2])

            # Compute the SDF
            d = length((p[0], p[2])) - radius
            h = abs(p[1]) - height
            return min(max(d, h), 0.0) + length((max(d, 0.0), max(h, 0.0)))

        # Define the bounds
        lower_bound = (-radius + center[0], -height + center[1], -radius + center[2])
        upper_bound = (radius + center[0], height + center[1], radius + center[2])

        # Call the parent constructor
        super().__init__(sdf, lower_bound, upper_bound, distance_threshold=radius / 100.0)

class Arrow(Geometry):
    """
    Class that represents an arrow

    Parameters
    ----------
    height: float
        The height of the arrow
    center: tuple
        The center of the arrow
    """

    def __init__(self, height, center=(0.0, 0.0, 0.0)):
        # Define the radius
        radius = height / 10.0

        # Make the cylinder
        cylinder = Cylinder(radius, height)

        # Make the cone
        angle = math.atan2(radius, height/3.0)
        cone = Cone((math.sin(angle), math.cos(angle)), height, (0.0 , 3.0 * height / 2.0, 0.0))
        cone = cone.rotate(math.pi, (1.0, 0.0, 0.0))

        # Union the two geometries
        arrow = cylinder + cone
        arrow = arrow.translate(center)

        # Call the parent constructor
        super().__init__(arrow.sdf, arrow.lower_bound, arrow.upper_bound, distance_threshold=radius / 100.0)

