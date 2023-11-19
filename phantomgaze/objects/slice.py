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
        raise NotImplementedError("Slice class is not implemented yet")
