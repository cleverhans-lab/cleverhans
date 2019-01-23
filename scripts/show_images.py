#!/usr/bin/env python3
"""
Loads an ndarray containing a batch of images and displays it.
Usage:
show_images.py file.npy
"""
import sys
import numpy as np
from cleverhans.plot.image import show, make_grid
# pylint has a bug here, thinks sys.argv is empty
_, path = sys.argv # pylint: disable=E0632

image_batch = np.load(path)

grid = make_grid(image_batch)

show(grid)
