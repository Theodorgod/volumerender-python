import time
import numpy as np
import volumerender
import volumerender_cython

import statistics

paths = ["data/data_128.hdf5", "data/datacube.hdf5", "data/data_512.hdf5", "data/data_1024.hdf5"]

def run():

    iterations = 10

    run_times = [0] * iterations

    for i in range(iterations):
        t1 = time.time()
        volumerender_cython.main(paths[1])
        t2 = time.time()
        run_times[i] = t2-t1

    mean = statistics.mean(run_times)
    stdev = statistics.stdev(run_times)
    print(f"Average run time over {iterations} iterations: {mean}")
    print(f"Standard deviation over {iterations} iterations: {stdev}")


run()

