# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport exp

def cython_transferFunction(np.ndarray[np.double_t, ndim=2] x):

    cdef np.ndarray[np.double_t, ndim=2] r = np.empty((x.shape[0], x.shape[1]), dtype=np.float64)
    cdef np.ndarray[np.double_t, ndim=2] g = np.empty((x.shape[0], x.shape[1]), dtype=np.float64)
    cdef np.ndarray[np.double_t, ndim=2] b = np.empty((x.shape[0], x.shape[1]), dtype=np.float64)
    cdef np.ndarray[np.double_t, ndim=2] a = np.empty((x.shape[0], x.shape[1]), dtype=np.float64)

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

    return r,g,b,a