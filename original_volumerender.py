import numpy as np
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


def main(path):
    """Volume Rendering"""

    # Load Datacube
    f = h5.File(path, "r")
    datacube = np.array(f["density"])

    # Datacube Grid
    Nx, Ny, Nz = datacube.shape
    x = np.linspace(-Nx / 2, Nx / 2, Nx)
    y = np.linspace(-Ny / 2, Ny / 2, Ny)
    z = np.linspace(-Nz / 2, Nz / 2, Nz)
    points = (x, y, z)

    # Do Volume Rendering at Different Viewing Angles
    Nangles = 10
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

        # Plot Volume Rendering
        # plt.figure(figsize=(4, 4), dpi=80)

        # plt.imshow(image)
        # plt.axis("off")

        # # Save figure
        # plt.savefig(
        #     "volumerender" + str(i) + ".png", dpi=240, bbox_inches="tight", pad_inches=0
        # )

    # Plot Simple Projection -- for Comparison
    # plt.figure(figsize=(4, 4), dpi=80)

    # plt.imshow(np.log(np.mean(datacube, 0)), cmap="viridis")
    # plt.clim(-5, 5)
    # plt.axis("off")

    # # Save figure
    # plt.savefig("projection.png", dpi=240, bbox_inches="tight", pad_inches=0)
    # plt.show()

    return 0



def run():

    # paths = ["data/data_128.hdf5", "data/datacube.hdf5", "data/data_512.hdf5", "data/data_1024.hdf5"]
    #paths = ["data/data_128.hdf5", "data/datacube.hdf5", "data/data_512.hdf5"]#, "data/data_1024.hdf5"]

    paths = ["data/datacube.hdf5", "data/data_384.hdf5", "data/data_512.hdf5", "data/data_640.hdf5", "data/data_768.hdf5"]
        
    iterations = 5

    run_times = [0] * iterations

    for path in paths:
        for i in range(iterations):
            t1 = time.time()
            main(path)
            t2 = time.time()
            run_times[i] = t2-t1

        mean = statistics.mean(run_times)
        print(f"Average run time over {iterations} iterations: {mean}")


if __name__ == "__main__":
    run()