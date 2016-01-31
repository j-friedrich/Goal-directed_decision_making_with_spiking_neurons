import numpy as np
cimport numpy as np
from libc.math cimport sin, exp

DTYPE = np.float
ctypedef np.float_t DTYPE_t


def trapz2d(int i, int n, int a, int bar=3, double h=.1, int steps=3):
    cdef double sum = 0.
    cdef double p
    cdef double v
    cdef double tmp1
    cdef double tmp2
    for p in np.arange(i / 33 + bar - h / 2, i / 33 - bar, -h):
        for v in np.arange(i % 33 - bar, i % 33 + bar, h):
            tmp1 = abs(n / 33 - get_next_pvs(p, v, a, steps)[0])
            tmp2 = n % 33 - get_next_pvs(p, v, a, steps)[1]
            sum += phi(i / 33 - p) * phi(min(tmp1, 16 - tmp1)) *\
                phi(i % 33 - v) * phi(tmp2)
    return sum * h * h


def phi(double i):
    cdef double pi = 3.14159265
    return 0 if abs(i) > 5 else exp(-2 * i**2) * (2 / pi)**(1. / 2)


def get_next_pv(double pp, double vv, int action):
    cdef double pi = 3.14159265
    cdef double v = vv + .02 * 8 / pi * (-.05 * (vv - 16) * pi / 8
                                         + 9.81 * sin(pp * pi / 8) + 5 * (action - 1))
    cdef double p = (pp + .02 * (v - 16)) % 16
    return [p, v]


def get_next_pvs(double pp, double vv, int action, int steps):
    cdef int i
    cdef double p = pp
    cdef double v = vv
    for i in range(steps):
        [p, v] = get_next_pv(p, v, action)
    return [p, v]


def get_R(double p, double v):
    cdef double pi = 3.14159265
    return exp(-(min(p, 16 - p) * pi / 8)**2 - .2 * ((v - 16) * pi / 8)**2)


def iterate(np.ndarray[DTYPE_t, ndim=2] V, double gamma, double h):
    cdef np.ndarray[DTYPE_t, ndim = 2, negative_indices = False, mode = 'c'] VV = np.copy(V)
    cdef int i, j
    cdef double p, v, pp, vv
    cdef np.ndarray[DTYPE_t, ndim = 1, negative_indices = False, mode = 'c'] Vtmp = np.zeros(3)
    cdef int a, i0, j0
    cdef double i1, j1
    for i, p in enumerate(np.arange(h / 2., 16, h)):
        for j, v in enumerate(np.arange(h / 2., 32, h)):
            for a in range(3):
                pp, vv = get_next_pv(p, v, a)
                i0 = int(pp / h + .5) - 1
                i1 = pp / h - .5 - i0
                j0 = int(vv / h + .5) - 1
                j1 = vv / h - .5 - j0
                if j0 < 0:
                    j0 = 0
                    j1 = 0
                if j0 > len(V[0]) - 2:
                    j0 = len(V[0]) - 2
                    j1 = 1
                Vtmp[a] = (1 - i1) * (1 - j1) * VV[i0 % len(V), j0]\
                    + (1 - i1) * j1 * VV[i0 % len(V), j0 + 1] \
                    + i1 * (1 - j1) * VV[(i0 + 1) % len(V), j0] \
                    + i1 * j1 * VV[(i0 + 1) % len(V), j0 + 1]
            V[i, j] = get_R(p, v) + gamma * max(Vtmp)
