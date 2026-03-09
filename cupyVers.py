import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import h5py as h5
#from scipy.interpolate import interpn
from cupyx.scipy.interpolate import interpn
import time
from cupyx.profiler import benchmark

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

#@profile
def main(file):
    """Volume Rendering"""

    # Load Datacube
    f = h5.File(file, "r")
    datacube = cp.array(f["density"])

    # Datacube Grid
    Nx, Ny, Nz = datacube.shape
    x = cp.linspace(-Nx / 2, Nx / 2, Nx)
    y = cp.linspace(-Ny / 2, Ny / 2, Ny)
    z = cp.linspace(-Nz / 2, Nz / 2, Nz)
    points = (x, y, z)

    # Do Volume Rendering at Different Viewing Angles
    Nangles = 10
    pre = time.time()
    for i in range(Nangles):
        #print("Rendering Scene " + str(i + 1) + " of " + str(Nangles) + ".\n")

        

        # Camera Grid / Query Points -- rotate camera view
        angle = np.pi / 2 * i / Nangles
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

        cp.cuda.Stream.null.synchronize()

    post = time.time()
    print(post - pre)

        #image = cp.clip(image, 0.0, 1.0)

        # Plot Volume Rendering
        #plt.figure(figsize=(4, 4), dpi=80)

        #plt.imshow(image.get())
        #plt.axis("off")

        # Save figure
        #plt.savefig(
           # "volumerender" + str(i) + ".png", dpi=240, bbox_inches="tight", pad_inches=0
        #)

    #print(tot_time)

    # Plot Simple Projection -- for Comparison
   # plt.figure(figsize=(4, 4), dpi=80)

    #plt.imshow(cp.log(cp.mean(datacube, 0)).get(), cmap="viridis")
   # plt.clim(-5, 5)
   # plt.axis("off")

    # Save figure
  #  plt.savefig("projection.png", dpi=240, bbox_inches="tight", pad_inches=0)
    #plt.show()

    return post - pre


if __name__ == "__main__":
    files = ["datacube.hdf5", "data_384.hdf5", "data_512.hdf5", "data_640.hdf5", "data_768.hdf5"]
    print(files)
    times = []
    iterations = 1
    
    for idx, f in enumerate(files[:1]):  # remove [:1] to run all
        file_times = []
        for _ in range(iterations):
            elapsed, images = main(f)
            file_times.append(elapsed)
            benchmark(main, f)  # CuPy GPU profiling
        avg_time = np.mean(file_times)
        times.append(file_times)
        print(f"{f} average over {iterations} iterations: {avg_time:.4f} s")

    x = np.arange(iterations)

    for size, t in zip([256, 384, 512, 640, 768], times):
        plt.plot(x, t, label=f"Size {size}")

    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig("output.png", dpi=300)

# Time for 256 file: 5.38
# Time for 384 file: 4.87
# Time for 512 file: 5.21
# Time for 640 file: 5.52
# Time for 768 file: 5.89