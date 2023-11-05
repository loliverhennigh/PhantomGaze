# Volume class for 3D volume data

#from phantomgaze.utils import _cupy_to_backend, _backend_to_cupy
from phantomgaze.slice import Slice

class Volume:
    """
    Volume class for 3D volume data

    Parameters
    ----------
    array : ndarray
        3D volume data, this can be any of the following:
        - numpy.ndarray
        - cupy.ndarray
        - torch.Tensor
        - jax.numpy.ndarray
        - warp.Array
    spacing : tuple
        Spacing between voxels in the volume
    origin : tuple
        Origin of the volume
    """

    def __init__(self, array, spacing, origin):
        self.array = array # TODO Add support for other backends
        self.spacing = spacing
        self.origin = origin
        self.shape = array.shape

    def slice(self, origin, normal):
        """
        Slice the volume with a plane

        Parameters
        ----------
        origin : tuple
            Origin of the plane
        normal : tuple
            Normal of the plane

        Returns
        -------
        slice : Slice
        """

        # Assert normal is in x, y, or z direction
        assert np.sum(np.abs(normal)) == 1, "Normal must be in x, y, or z direction"

        # Determine which axis to slice along
        axis = None
        for i, n in enumerate(normal):
            if abs(n) == 1:
                axis = i
                break

        # Calculate the index of the slice
        slice_idx = int((origin[axis] - self.origin[axis]) / self.spacing[axis])

        # Extract the slice from the array
        if axis == 0:
            slice_data = self.array[slice_idx, :, :]
        elif axis == 1:
            slice_data = self.array[:, slice_idx, :]
        else:
            slice_data = self.array[:, :, slice_idx]

        # Create and return a new Slice instance
        return Slice(slice_data, self.spacing, origin, normal)
