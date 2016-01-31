# compile with: python setup.py build_ext --inplace

import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX, srand

DTYPE = np.float
ctypedef np.float_t DTYPE_t

# scale weights and reset by popsize (or something else, e.g. not-all-to all)
# before calling this function


def runpop_js(np.ndarray[DTYPE_t, ndim=2] W, double step, int pop_size, int rate=100,
              int T=500, int tm=20, int ts=2, double reset=0, int seed=0):
    srand(seed)
    cdef int K = len(W)
    cdef np.ndarray[DTYPE_t, ndim = 2, negative_indices = False, mode = 'c']\
        spikes = np.zeros((T / step, K))
    spikes[0, K - pop_size:] = 1
    cdef np.ndarray[DTYPE_t, ndim = 1, negative_indices = False, mode = 'c']\
        u = np.zeros(K)
    cdef double fs = np.exp(-step / ts)  # decay factor synapse
    cdef double fm = np.exp(-step / tm)  # decay factor membrane
    # combine factor to save computing time
    cdef double f = 1. / ts * np.sqrt(fs) * step / tm * np.sqrt(fm)
    cdef np.ndarray[DTYPE_t, ndim = 1, negative_indices = False, mode = 'c']\
        c = np.zeros(K)  # current
    cdef int t
    cdef int k
    cdef double cr  # constant current into reward neurons
    cdef double f2  # another factor
    if reset == 0:
        for t in range(1, int(T / step)):
            c *= fs
            if 1 in spikes[t - 1]:
                c += np.dot(W, spikes[t - 1]) * f
            u = u * fm + c
            u[K - pop_size:] = rate / 1000.
            for k in range(K):
                spikes[t, k] = rand() < step * u[k] * RAND_MAX
    else:
        u[K - pop_size:] = rate / 1000.
        f *= (1 + reset)
        cr = rate / 1000. * step / tm * (1 + reset)  # constant current into reward neuron
        f2 = 1. * reset / tm
        for t in range(1, int(T / step)):
            c *= fs
            if 1 in spikes[t - 1]:
                c += np.dot(W, spikes[t - 1]) * f
            u = u * fm + c - f2 * spikes[t - 1]
            u[K - pop_size:] += cr
            for k in range(K):
                spikes[t, k] = rand() < step * u[k] * RAND_MAX
    return spikes


def runpopU_js(np.ndarray[DTYPE_t, ndim=2] W, np.ndarray[DTYPE_t, ndim=1] uinit,
               double step, int pop_size, int rate=100, int T=500,
               int tm=20, int ts=2, double reset=0, int seed=0):
    srand(seed)
    cdef int K = len(W)
    cdef np.ndarray[DTYPE_t, ndim = 2, negative_indices = False, mode = 'c']\
        spikes = np.zeros((T / step, K))
    spikes[0, K - pop_size:] = 1
    cdef np.ndarray[DTYPE_t, ndim = 1, negative_indices = False, mode = 'c']\
        u = np.zeros(K)
    cdef double fs = np.exp(-step / ts)  # decay factor synapse
    cdef double fm = np.exp(-step / tm)  # decay factor membrane
    # combine factor to save computing time
    cdef double f = 1. / ts * np.sqrt(fs) * step / tm * np.sqrt(fm)
    cdef np.ndarray[DTYPE_t, ndim = 1, negative_indices = False, mode = 'c']\
        c = np.zeros(K)  # current
    cdef int t
    cdef int k
    cdef double cr  # constant current into reward neurons
    cdef double f2  # another factor
    u = uinit
    if reset == 0:
        for t in range(1, int(T / step)):
            c *= fs
            if 1 in spikes[t - 1]:
                c += np.dot(W, spikes[t - 1]) * f
            u = u * fm + c
            u[K - pop_size:] = rate / 1000.
            for k in range(K):
                spikes[t, k] = rand() < step * u[k] * RAND_MAX
    else:
        f *= (1 + reset)
        cr = rate / 1000. * step / tm * (1 + reset)  # constant current into reward neuron
        f2 = 1. * reset / tm
        for t in range(1, int(T / step)):
            c *= fs
            if 1 in spikes[t - 1]:
                c += np.dot(W, spikes[t - 1]) * f
            u = u * fm + c - f2 * spikes[t - 1]
            u[K - pop_size:] += cr
            for k in range(K):
                spikes[t, k] = rand() < step * u[k] * RAND_MAX
    return (spikes, u)
