# Volume class for 3D volume data

from phantomgaze.utils.backends import backend_to_cupy

class Volume:
    """
    Volume class for 3D volume data

    Parameters
    ----------
    array : ndarray
        3D volume data, this can be any of the following:
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
        self.array = backend_to_cupy(array)
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
        raise NotImplementedError("Volume.slice() is not implemented yet")
