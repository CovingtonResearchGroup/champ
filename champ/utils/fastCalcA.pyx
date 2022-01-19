import numpy as np
cimport numpy as np
cimport cython

from libc.stdlib cimport malloc, free
from libc.math cimport fabs

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t findYMin(DTYPE_t [:] y, int nump) nogil:
  cdef DTYPE_t ymin = y[0]
  cdef Py_ssize_t i
  for i in xrange(nump):
    if y[i] < ymin:
      ymin = y[i]

  return(ymin)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t findYMax(DTYPE_t [:] y, int nump) nogil:
  cdef DTYPE_t ymax = y[0]
  cdef Py_ssize_t i
  for i in xrange(nump):
    if y[i] > ymax:
      ymax = y[i]

  return(ymax)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t _calcA(double* x, double* xm , double* y, double* ym, int nump) nogil:
  cdef:
    Py_ssize_t i
    DTYPE_t sA, A
    DTYPE_t sum = 0

  for i in xrange(nump):
    sum += xm[i]*y[i] - x[i]*ym[i]

  sA = sum * 0.5
  A = fabs(sA)
  return(A)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calcA(DTYPE_t [:] x, DTYPE_t [:] xm, DTYPE_t [:] y, DTYPE_t [:] ym, DTYPE_t depth = -1.0):
  cdef int nump = <int>x.size #number of points

  cdef DTYPE_t ymin = findYMin(y, nump)
  cdef DTYPE_t ymax = findYMax(y, nump)

  if ymax - ymin <= depth or depth == -1.0: #use full xs
    return(_calcA(&x[0], &xm[0], &y[0], &ym[0], nump)) #&x[0] mem addr of first element

  # Allocate mem for subset arrays
  cdef double *xs = <double *>malloc(nump*sizeof(double))
  cdef double *ys = <double *>malloc(nump*sizeof(double))
  cdef double *xms = <double *>malloc(nump*sizeof(double))
  cdef double *yms = <double *>malloc(nump*sizeof(double))

  cdef Py_ssize_t i
  cdef int n = 0

  for i in xrange(nump):
    if y[i] - ymin < depth:
      xs[<Py_ssize_t>n] = <double>x[i]
      ys[<Py_ssize_t>n] = <double>y[i]
      xms[<Py_ssize_t>n+1] = xs[<Py_ssize_t>n]
      yms[<Py_ssize_t>n+1] = ys[<Py_ssize_t>n]
      n += 1

  xms[0] = xs[<Py_ssize_t>n-1]
  yms[0] = ys[<Py_ssize_t>n-1]

  nump = n # new number of pts

  cdef DTYPE_t A = _calcA(xs, xms, ys, yms, nump)
  free(xs); free(ys); free(xms); free(yms); #Free mem

  return(A)
