import logging
import cupy as cp

# Import backends
# jax
try:
    import jax.dlpack as jdlpack
    import jax.numpy as jnp
except ImportError:
    logging.warning("JAX not installed, JAX backend not available")

# warp
try:
    import warp as wp
except ImportError:
    logging.warning("Warp not installed, Warp backend not available")

# torch
try:
    import torch
except ImportError:
    logging.warning("Torch not installed, Torch backend not available")


def backend_to_cupy(backend_array):
    """
    Convert backend array to cupy array

    Parameters
    ----------
    backend_array : array
        Array from backend, can be jax, warp, or torch
    """

    # Perform zero-copy conversion to cupy array
    if isinstance(backend_array, cp.ndarray):
        return backend_array
    elif isinstance(backend_array, jnp.ndarray):
        dl_array = jdlpack.to_dlpack(backend_array)
        cupy_array = cp.fromDlpack(dl_array)
    elif isinstance(backend_array, wp.array):
        dl_array = wp.to_dlpack(backend_array)
        cupy_array = cp.fromDlpack(dl_array)
    elif isinstance(backend_array, torch.Tensor):
        cupy_array = cp.fromDlpack(backend_array.toDlpack())
    else:
        raise ValueError(f"Backend {type(backend_array)} not supported")
    return cupy_array
