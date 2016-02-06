# compile with: python setup.py build_ext --inplace

import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.math cimport exp, sqrt

DTYPE = np.float
ctypedef np.float_t DTYPE_t

# scale weights and reset by popsize (or something else, e.g. not-all-to all)
# before calling this function


def run(np.ndarray[DTYPE_t, ndim=2] W, np.ndarray[DTYPE_t, ndim=1] uinit, double step,
        int pop_size, int rate=100, int T=500, int tm=20, int ts=2, double reset=0,
        int seed=1, double r0=0., double r1=1., int tdec=300, int trise=50, int tdly=0, rb=0., V0=0., gamma=1.):
    srand(seed + 1)
    W[:pop_size, -pop_size] = r0
    W[pop_size:2 * pop_size, -pop_size] = r1
    cdef int K = len(W)
    cdef np.ndarray[DTYPE_t, ndim = 2, negative_indices = False, mode = 'c']\
        spikes = np.zeros((int(T / step), K))
    cdef np.ndarray[DTYPE_t, ndim = 1, negative_indices = False, mode = 'c']\
        u = np.zeros(K)
    cdef double fs = exp(-step / ts)  # decay factor synapse
    cdef double fm = exp(-step / tm)  # decay factor membrane
    # combine factor to save computing time
    cdef double f = 1. / ts * sqrt(fs) * step / tm * sqrt(fm)
    cdef np.ndarray[DTYPE_t, ndim = 1, negative_indices = False, mode = 'c']\
        c = np.zeros(K)  # current
    cdef int t
    cdef int k
    cdef double f2  # another factor
    cdef double fV0
    cdef double th
    u = uinit
    if reset == 0:
        fV0 = (1 - fm) * V0 * gamma * rate / 1000.
        th = V0 * (gamma - 1) * rate / 1000.
        for t in range(1, int(T / step)):
            c *= fs
            if 1 in spikes[t - 1]:
                c += np.dot(W, spikes[t - 1]) * f
            u = u * fm + c + fV0
            u[K - pop_size:] = rate / 1000. * (0 + (rb if t * step < tdly else
                                                    exp(-(t * step - tdly) / tdec) - exp(-(t * step - tdly) / trise)))
            for k in range(K - pop_size):
                spikes[t, k] = rand() < step * (u[k] - th) * RAND_MAX
            for k in range(K - pop_size, K):
                spikes[t, k] = rand() < step * u[k] * RAND_MAX
    else:
        f *= (1 + reset)
        f2 = 1. * reset / tm
        fV0 = (1 - fm) * (1 + reset) * V0 * gamma * rate / 1000.
        th = (1 + reset) * V0 * (gamma - 1) * rate / 1000.
        for t in range(1, int(T / step)):
            c *= fs
            if 1 in spikes[t - 1]:
                c += np.dot(W, spikes[t - 1]) * f
            u = u * fm + c - f2 * spikes[t - 1] + fV0
            u[K - pop_size:] = rate / 1000. * (0 + (rb if t * step < tdly
                                                    else exp(-(t * step - tdly) / tdec) - exp(-(t * step - tdly) / trise)))
            for k in range(K - pop_size):
                spikes[t, k] = rand() < step * (u[k] - th) * RAND_MAX
            for k in range(K - pop_size, K):
                spikes[t, k] = rand() < step * u[k] * RAND_MAX
    return np.array(spikes)


def runRoesch(np.ndarray[DTYPE_t, ndim=2] W, np.ndarray[DTYPE_t, ndim=1] uinit, double step,
        int pop_size, int rate=100, int T=500, int tm=20, int ts=2, double reset=0,
        int seed=0, double th=0., double rbase=0., double r0=0., double r1=1., double rp=5.,
        int tmean=70, int tsigma=50):
    srand(seed + 1)
    W[:pop_size, -pop_size] = r0
    W[pop_size:2 * pop_size, -pop_size] = r1
    cdef int K = len(W)
    cdef np.ndarray[DTYPE_t, ndim = 2, negative_indices = False, mode = 'c']\
        spikes = np.zeros((int(T / step), K))
    spikes[0, K - pop_size:] = 1
    cdef np.ndarray[DTYPE_t, ndim = 1, negative_indices = False, mode = 'c']\
        u = np.zeros(K)
    cdef double fs = exp(-step / ts)  # decay factor synapse
    cdef double fm = exp(-step / tm)  # decay factor membrane
    # combine factor to save computing time
    cdef double f = 1. / ts * sqrt(fs) * step / tm * sqrt(fm)
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
            u = u * fm + c + (1 - fm) * (2 * th + rbase)
            u[K - pop_size:] = rate / 1000. * (1 + rp * exp(-(t * step - tmean)**2 / tsigma**2))
            for k in range(K - pop_size):
                spikes[t, k] = rand() < step * (th + (u[k] - th) * (u[k] > th)) * RAND_MAX
            for k in range(K - pop_size, K):
                spikes[t, k] = rand() < step * u[k] * RAND_MAX
    else:
        f *= (1 + reset)
        cr = rate / 1000. * step / tm * (1 + reset)  # constant current into reward neuron
        f2 = 1. * reset / tm
        for t in range(1, int(T / step)):
            c *= fs
            if 1 in spikes[t - 1]:
                c += np.dot(W, spikes[t - 1]) * f
            u = u * fm + c - f2 * spikes[t - 1] + (1. + reset) * (1 - fm) * (2 * th + rbase)
            u[K - pop_size:] = rate / 1000. * (1 + rp * exp(-(t * step - tmean)**2 / tsigma**2))
            for k in range(K - pop_size):
                spikes[t, k] = rand() < step * (th + (u[k] - th) * (u[k] > th)) * RAND_MAX
            for k in range(K - pop_size, K):
                spikes[t, k] = rand() < step * u[k] * RAND_MAX
    return np.array(spikes)
