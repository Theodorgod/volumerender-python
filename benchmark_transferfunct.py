import time
import numpy as np
import matplotlib.pyplot as plt
import statistics

from original_volumerender import transferFunction as tf_original
from cython_volumerender import transferFunction as tf_cython
from numexpr_volumerender import transferFunction as tf_numexpr
from inplace_volumerender import transferFunction as tf_inplace
from cupy_volumerender import transferFunction as tf_cupy



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
    tf_inplace_wrapped = wrap_inplace(tf_inplace, log_input.shape)
    tf_numexpr_wrapped = wrap_numexpr(tf_numexpr, log_input.shape)

    functions = [
        ("Original", tf_original),
        ("Cython", tf_cython),
        ("NumExpr", tf_numexpr_wrapped),
        ("Inplace", tf_inplace_wrapped),
        ("CuPy", tf_cupy)
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
    run()
