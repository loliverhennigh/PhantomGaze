# Slice class for 2D slice data of a 3D volume

class Slice:
    """
    Slice class for 2D slice data of a 3D volume

    Parameters
    ----------
    array : ndarray
        2D slice data, this can be any of the following:
        - numpy.ndarray
        - cupy.ndarray
        - torch.Tensor
        - jax.numpy.ndarray
        - warp.Array
    spacing : tuple
        Spacing between voxels in the slice
    origin : tuple
        Origin of the slice
    normal : tuple
        Normal of the slice
    """

    def __init__(self, array, spacing, origin, normal):
        self.array = array
        self.spacing = spacing
        self.origin = origin
        self.normal = normal
        self.shape = array.shape

        # Assert normal is in x, y, or z direction
        assert np.sum(np.abs(normal)) == 1, "Normal must be in x, y, or z direction"
