#
# Implements fast cubic spline method of Habermann and Kindermann (2007)
#
import numpy as np
from scipy.linalg import solve_banded
from ._fastInterp import interp, interpf

class fast1DCubicSpline(object):
    """
    Interpolate a 1-D function with cubic splines.

    Parameters
    ----------
    x : (N,) numpy.ndarray
        A 1-D array of floats.
    y : (N,) numpy.ndarray
        A 1-D array of floats, the same length as 'x'.
    bounds_error : bool, optional
        If True an error is raised if value to be interpolated is out of the
        original bounds. Default is true.
    fill_value : None or tuple length 2, optional
        - if a tuple is provided, and bounds_error is False, set values to be
        interpolated out of range to fill_value[0] when less than original
        minimum x, and fill_value[1] when larger than the original maximum x.

    Methods
    -------
    __call__
    """
    def __init__(self, x, y, bounds_error=True, fill_value=None):
        """Initialize interpolator."""

        assert x.size == y.size, "x and y must be same length"

        assert type(bounds_error) == type(True), "bounds_error must be True/False"

        self.b_e = bounds_error

        self.f_v = fill_value
        if type(self.f_v) != type(None):
            assert len(fill_value) == 2, "fill_value must have 2 elements"

        self.x = x
        self.y = y

        self.n = x.size - 1
        self.xmin = self.x[0]
        self.xmax = self.x[-1]


        self.coeffs = self.calcCoeffs()

    def calcCoeffs(self):
        """Calculates interpolation coeffecients"""
        coeffs = np.zeros(self.n+3)
        coeffs[1] = 1./6 * self.y[0]
        coeffs[self.n+1] = 1./6 * self.y[self.n]

        mat = np.ones( (3, self.n - 1) )
        mat[0, 0] = 0
        mat[1, :] = 4
        mat[-1, -1] = 0

        b = self.y[1:-1].copy()
        b[0] -= coeffs[1]
        b[-1] -= coeffs[self.n+1]

        coeffs[2:-2] = solve_banded((1,1), mat, b)

        coeffs[0] = 2*coeffs[1] - coeffs[2]
        coeffs[-1] = 2*coeffs[-2] - coeffs[-3]

        return(coeffs)

    def __call__(self, x):
        """
        Returns interpolated values.

        Parameters
        ----------
        x : numpy.ndarray, float, or np.float64
            Value or values to evaluate in interpolated function.
        """
        assert type(x) == np.ndarray or type(x) == np.float64 or type(x) == float

        if self.b_e:
            assert np.any(np.logical_or(x<self.xmin,x>self.xmax)) == True, "Variable out of bounds"

        ret = 0

        if type(x) == np.ndarray:
            ret = np.zeros_like(x) #array to store results
            interp(x, ret, self.coeffs, self.xmin, self.xmax, self.f_v)

        elif type(x) == float or type(x) == np.float64:
            if x < self.xmin:
                ret = self.f_v[0]
            elif x > self.xmax:
                ret = self.f_v[1]
            else:
                ret = interpf(x, self.coeffs, self.n, self.xmin, self.xmax)

        return(ret)
