import numpy as np
cimport numpy as np
cimport cython

from libc.stdlib cimport malloc, free
from libc.math cimport fabs, hypot

ctypedef np.float64_t DTYPE_t

np.import_array()

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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t _calcP(double* x, double* xp , double* y, double* yp, int nump) nogil:
  cdef:
    Py_ssize_t i
    DTYPE_t sum = 0

  for i in xrange(nump):
    sum += hypot(x[i] - xp[i], y[i] - yp[i])

  return(sum)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calcP(DTYPE_t [:] x, DTYPE_t [:] xp, DTYPE_t [:] y, DTYPE_t [:] yp, DTYPE_t depth = -1.0):
  cdef int nump = <int>x.size #Number of points in xs

  cdef DTYPE_t ymin = findYMin(y, nump)
  cdef DTYPE_t ymax = findYMax(y, nump)

  if ymax - ymin <= depth or depth == -1.0: # Calculate perimeter for whole xs
    return(_calcP(&x[0], &xp[0], &y[0], &yp[0], nump)) #&x[0] mem addr of first element

  # Allocte memory for depth x, y subsets below depth
  cdef double *xs = <double *>malloc(nump*sizeof(double))
  cdef double *ys = <double *>malloc(nump*sizeof(double))
  cdef double *xps = <double *>malloc(nump*sizeof(double))
  cdef double *yps = <double *>malloc(nump*sizeof(double))

  cdef Py_ssize_t i
  cdef int n = 0

  for i in xrange(nump):
    if y[i] - ymin < depth:
      xs[<Py_ssize_t>n] = <double>x[i]
      ys[<Py_ssize_t>n] = <double>y[i]
      xps[<Py_ssize_t>n] = xp[<Py_ssize_t>i]
      yps[<Py_ssize_t>n] = yp[<Py_ssize_t>i]
      n += 1

  if n == 0: # Subset is empty array
    free(xs); free(ys); free(xps); free(yps); #free mem
    return(0.0)

  xps[n-1] = xs[0] # Last element of plus array is first of reg array
  yps[n-1] = ys[0]

  nump = n # Number of pts in subset

  cdef DTYPE_t P = _calcP(xs, xps, ys, yps, nump-1) #nump-1 as full produces false ceiling
  free(xs); free(ys); free(xps); free(yps); #free mem

  return(P)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rollm(double [:] a):
  cdef:
    int n = <int>a.size
    Py_ssize_t i
    np.ndarray[np.float64_t, ndim=1] b = np.PyArray_EMPTY(1, [n], np.NPY_FLOAT64, 0)

  for i in xrange(n):
    b[i] = a[ ((i-1) % n + n) % n]

  return b

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rollp(double [:] a):
  cdef:
    int n = <int>a.size
    Py_ssize_t i
    np.ndarray[np.float64_t, ndim=1] b = np.PyArray_EMPTY(1, [n], np.NPY_FLOAT64, 0)

  for i in xrange(n):
    b[i] = a[ ((i+1) % n + n) % n]

  return b
