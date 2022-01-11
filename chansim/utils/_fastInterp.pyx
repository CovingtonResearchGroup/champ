import cython
from libc.math cimport fmin, fabs

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def interp(double [:] x, double [:] ret, double [:] coeffs, double a, double b, (double, double) f_v):
    cdef int nump = <int>x.size
    cdef int n = <int>coeffs.size - 3
    cdef Py_ssize_t i

    for i in xrange(nump):

        if x[i] < a:
            ret[i] = f_v[0]
        elif x[i] > b:
            ret[i] = f_v[1]
        else:
            ret[i] = _interp(x[i], coeffs, n, a, b)

    return(True)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _interp(double x, double [:] coeffs, int n, double a, double b) nogil:
    cdef double h = (b-a)/<double>n
    cdef int i
    cdef int l = <int>( (x-a)//h ) + 1
    cdef int m = <int>fmin( l+3, n+3 )
    cdef double t, u
    cdef double s = 0

    for i in xrange(l, m+1):
        t = fabs( (x-a)/h - (i-2) )

        if t <= 1:
            u = 4 - 6*t**2 + 3*t**3
        elif t <= 2:
            u = (2-t)**3
        else:
            u = 0

        s += coeffs[i-1] * u

    return(s)
