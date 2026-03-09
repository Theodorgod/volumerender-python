import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import h5py as h5
from scipy.interpolate import interpn 
import time
import statistics

"""
Create Your Own Volume Rendering (With Python)
Philip Mocz (2020) Princeton University, @PMocz

Simulate the Schrodinger-Poisson system with the Spectral method
"""

def transferFunction(x):
    r = (
        1.0 * cp.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * cp.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.1 * cp.exp(-((x - -3.0) ** 2) / 0.5)
    )
    g = (
        1.0 * cp.exp(-((x - 9.0) ** 2) / 1.0)
        + 1.0 * cp.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.1 * cp.exp(-((x - -3.0) ** 2) / 0.5)
    )
    b = (
        0.1 * cp.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * cp.exp(-((x - 3.0) ** 2) / 0.1)
        + 1.0 * cp.exp(-((x - -3.0) ** 2) / 0.5)
    )
    a = (
        0.6 * cp.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * cp.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.01 * cp.exp(-((x - -3.0) ** 2) / 0.5)
    )

    return r, g, b, a


def main(path):
    """Volume Rendering"""

    # Load Datacube
    f = h5.File(path, "r")
    datacube = cp.array(f["density"])

    # Datacube Grid
    Nx, Ny, Nz = datacube.shape
    x = cp.linspace(-Nx / 2, Nx / 2, Nx)
    y = cp.linspace(-Ny / 2, Ny / 2, Ny)
    z = cp.linspace(-Nz / 2, Nz / 2, Nz)
    points = (x, y, z)

    # Do Volume Rendering at Different Viewing Angles
    Nangles = 10
    for i in range(Nangles):
        print("Rendering Scene " + str(i + 1) + " of " + str(Nangles) + ".\n")

        # Camera Grid / Query Points -- rotate camera view
        angle = cp.pi / 2 * i / Nangles
        N = 180
        c = cp.linspace(-N / 2, N / 2, N)
        qx, qy, qz = cp.meshgrid(c, c, c)
        qxR = qx
        qyR = qy * cp.cos(angle) - qz * cp.sin(angle)
        qzR = qy * cp.sin(angle) + qz * cp.cos(angle)
        qi = cp.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T

        # Interpolate onto Camera Grid
        camera_grid = interpn(points, datacube, qi, method="linear").reshape((N, N, N))

        # Do Volume Rendering
        image = cp.zeros((camera_grid.shape[1], camera_grid.shape[2], 3))

        for dataslice in camera_grid:
            r, g, b, a = transferFunction(cp.log(dataslice))
            image[:, :, 0] = a * r + (1 - a) * image[:, :, 0]
            image[:, :, 1] = a * g + (1 - a) * image[:, :, 1]
            image[:, :, 2] = a * b + (1 - a) * image[:, :, 2]

        image = cp.clip(image, 0.0, 1.0)



    return 0