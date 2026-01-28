import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.interpolate import interpn
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import os

"""
Create Your Own Volume Rendering (With Python)
Philip Mocz (2020) Princeton University, @PMocz

Simulate the Schrodinger-Poisson system with the Spectral method
"""


def transferFunction(x):
    r = (
        1.0 * np.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * np.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.1 * np.exp(-((x - -3.0) ** 2) / 0.5)
    )
    g = (
        1.0 * np.exp(-((x - 9.0) ** 2) / 1.0)
        + 1.0 * np.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.1 * np.exp(-((x - -3.0) ** 2) / 0.5)
    )
    b = (
        0.1 * np.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * np.exp(-((x - 3.0) ** 2) / 0.1)
        + 1.0 * np.exp(-((x - -3.0) ** 2) / 0.5)
    )
    a = (
        0.6 * np.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * np.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.01 * np.exp(-((x - -3.0) ** 2) / 0.5)
    )

    return r, g, b, a


def main():
    """Volume Rendering"""

    # Load Datacube
    f = h5.File("datacube.hdf5", "r")
    datacube = np.array(f["density"])

    # Datacube Grid
    Nx, Ny, Nz = datacube.shape
    x = np.linspace(-Nx / 2, Nx / 2, Nx)
    y = np.linspace(-Ny / 2, Ny / 2, Ny)
    z = np.linspace(-Nz / 2, Nz / 2, Nz)
    points = (x, y, z)

    # Store images for GIF creation
    images = []
    
    # Do Volume Rendering at Different Viewing Angles
    Nangles = 50
    for i in range(Nangles):
        print("Rendering Scene " + str(i + 1) + " of " + str(Nangles) + ".\n")

        # Camera Grid / Query Points -- rotate camera view
        angle = np.pi / 2 * i / Nangles
        N = 180
        c = np.linspace(-N / 2, N / 2, N)
        qx, qy, qz = np.meshgrid(c, c, c)
        qxR = qx
        qyR = qy * np.cos(angle) - qz * np.sin(angle)
        qzR = qy * np.sin(angle) + qz * np.cos(angle)
        qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T

        # Interpolate onto Camera Grid
        camera_grid = interpn(points, datacube, qi, method="linear").reshape((N, N, N))

        # Do Volume Rendering
        image = np.zeros((camera_grid.shape[1], camera_grid.shape[2], 3))

        for dataslice in camera_grid:
            r, g, b, a = transferFunction(np.log(dataslice))
            image[:, :, 0] = a * r + (1 - a) * image[:, :, 0]
            image[:, :, 1] = a * g + (1 - a) * image[:, :, 1]
            image[:, :, 2] = a * b + (1 - a) * image[:, :, 2]

        image = np.clip(image, 0.0, 1.0)
        
        # Store image and save as PNG
        images.append((image * 255).astype(np.uint8))
        plt.imsave(f"volumerender{i}.png", image)

    # Create GIF from the rendered images
    print("Creating GIF from rendered images...\n")
    clip = ImageSequenceClip(images, fps=60)
    clip.write_gif("volumerender.gif")
    print("GIF saved as volumerender.gif\n")
    
    # Open the GIF with default viewer
    os.startfile("volumerender.gif")

    return 0


if __name__ == "__main__":
    main()
