import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import h5py as h5
from scipy.interpolate import interpn 
import time
import statistics
from numexpr import evaluate
from cythonfn import cython_transferFunction


def original_transferFunction(x):
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

def inplace_transferFunction(x, rgba):
    rgba[0] += (
        1.0 * np.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * np.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.1 * np.exp(-((x - -3.0) ** 2) / 0.5)
    )
    rgba[1] += (
        1.0 * np.exp(-((x - 9.0) ** 2) / 1.0)
        + 1.0 * np.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.1 * np.exp(-((x - -3.0) ** 2) / 0.5)
    )
    rgba[2] += (
        0.1 * np.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * np.exp(-((x - 3.0) ** 2) / 0.1)
        + 1.0 * np.exp(-((x - -3.0) ** 2) / 0.5)
    )
    rgba[3] += (
        0.6 * np.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * np.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.01 * np.exp(-((x - -3.0) ** 2) / 0.5)
    )

def numexpr_transferFunction(x, rgba):
    e1 = np.exp(-((x - 9.0) ** 2) / 1.0)
    e2 = np.exp(-((x - 3.0) ** 2) / 0.1)
    e3 = np.exp(-((x - -3.0) ** 2) / 0.5)

    evaluate("1.0 * e1 + 0.1 * e2 + 0.1 * e3", out=rgba[0])
    evaluate("1.0 * e1 + 1.0 * e2 + 0.1 * e3", out=rgba[1])
    evaluate("0.1 * e1 + 0.1 * e2 + 1.0 * e3", out=rgba[2])
    evaluate("0.6 * e1 + 0.1 * e2 + 0.01 * e3", out=rgba[3])

    return rgba[0], rgba[1], rgba[2], rgba[3]

def cupy_transferFunction(x):
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


iterations=100


def benchmark_function(func, log_input):
    times = []

    # Warmup
    for _ in range(10):
        func(log_input)

    for _ in range(iterations):
        t1 = time.perf_counter()
        func(log_input)
        t2 = time.perf_counter()
        times.append(t2 - t1)

    return statistics.mean(times), statistics.stdev(times)

def wrap_inplace(func, shape):
    rgba = np.zeros((4,) + shape)

    def wrapped(x):
        func(x, rgba)
        return rgba

    return wrapped

def wrap_numexpr(func, shape):
    rgba = np.zeros((4,) + shape)

    def wrapped(x):
        func(x, rgba)
        return rgba

    return wrapped

def run():

    N = 180
    fake_dataslice = np.random.rand(N, N)
    log_input = np.log(fake_dataslice)

    # Wrap special functions
    tf_inplace_wrapped = wrap_inplace(inplace_transferFunction, log_input.shape)
    tf_numexpr_wrapped = wrap_numexpr(numexpr_transferFunction, log_input.shape)

    functions = [
        ("Original", original_transferFunction),
        ("Cython", cython_transferFunction),
        ("NumExpr", tf_numexpr_wrapped),
        ("Inplace", tf_inplace_wrapped)
        # ("CuPy", cupy_transferFunction)
    ]

    means = []
    stds = []
    labels = []

    print(f"\nBenchmarking size {N}x{N}\n")

    for name, func in functions:
        mean, std = benchmark_function(func, log_input)
        means.append(mean)
        stds.append(std)
        labels.append(name)
        print(f"{name}: {mean:.6f}s ± {std:.6f}")

    plt.figure()
    plt.bar(labels, means)
    plt.ylabel("Time per call (seconds)")
    plt.title("transferFunction() Benchmark (N=180)")
    plt.grid(axis="y")
    plt.show()

if __name__ == "__main__":
    print("HEJ THEO, glöm inte att du äger :heart:")
    print("To run: ")
    print("python3 setup.py build_ext --inplace")
    print("python3 transfer_function.py")
    run()

