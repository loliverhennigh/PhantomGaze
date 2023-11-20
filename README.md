# PhantomGaze: A GPU-Accelerated Rendering Engine for Scientific Visualization

## Introduction

PhantomGaze is a Python package that provides a basic rendering engine optimized for scientific computing visualizations. Utilizing the power of Numba, it enables efficient operations directly on the GPU, enhancing the visualization of complex scientific data. This package is particularly useful for rendering volumetric data, geometric shapes, and contour visualizations with ease.

## Features

- GPU-accelerated rendering engine
- Volumetric rendering
- Geometric rendering
- Contour rendering
- Customizable color maps
- Integration with GPU libraries such as CuPy, JAX, PyTorch, and Warp.

## Installation

PhantomGaze can be installed using pip:

```bash
pip install .
```

(TODO: Make package available on PyPI)

## Usage

There are several examples in the `examples` directory. A minimal example of a contour plot is shown below:

```python
import cupy as cp
import matplotlib.pyplot as plt

import phantomgaze as pg

# Create SDF feild of a sphere using cupy
lin = cp.linspace(-1, 1, 256)
X, Y, Z = cp.meshgrid(lin, lin, lin, indexing="ij")
sphere = -(cp.sqrt(X**2 + Y**2 + Z**2) - 1.0)
sphere_volume = pg.objects.Volume(
    sphere, spacing=(2 / 256, 2 / 256, 2 / 256), origin=(-1.0, -1.0, -1.0)
)
color_volume = pg.objects.Volume(
    cp.sin(X * 2 * cp.pi) * cp.sin(Y * 2 * cp.pi) * cp.sin(Z * 2 * cp.pi),
    spacing=(2 / 256, 2 / 256, 2 / 256),
    origin=(-1.0, -1.0, -1.0),
)

# Create camera object
camera = pg.Camera(position=(2.0, 1.0, -4.0), focal_point=(0.0, 0.0, 0.0), view_up=(0.0, 1.0, 0.0))

# Render the contour of the inner sphere
screen_buffer = pg.render.contour(sphere_volume, camera, threshold=0.0, color=color_volume)

# Show the rendered image
plt.imshow(screen_buffer.image.get())
plt.show()
```

This produces the following image:

![Contour](https://github.com/loliverhennigh/PhantomGaze/blob/main/assets/readme_example.png)

## Gallery

The following images were generated using the examples in the `examples` directory.

![Axes](https://github.com/loliverhennigh/PhantomGaze/blob/main/assets/axes.png)
![Geometry](https://github.com/loliverhennigh/PhantomGaze/blob/main/assets/geometry.png)
![Volume](https://github.com/loliverhennigh/PhantomGaze/blob/main/assets/volume.png)

