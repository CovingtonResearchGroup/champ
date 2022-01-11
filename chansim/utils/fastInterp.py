import numpy as np
from scipy.linalg import solve_banded
from _fastInterp import interp

class fast1DCubicSpline(object):
    def __init__(self, x, y, bounds_error=True, fill_value=np.nan):

        assert type(bounds_error) == type(True), "bounds_error must be True/False"

        self.b_e = bounds_error

        self.f_v = fill_value
        if self.f_v != np.nan:
            assert len(fill_value) == 2, "fill_value must have 2 elements"

        self.x = x
        self.y = y

        self.n = x.size - 1
        self.xmin = self.x[0]
        self.xmax = self.x[-1]


        self.coeffs = self.calcCoeffs()

    def calcCoeffs(self):
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

        if self.b_e:
            assert np.any(np.logical_or(x<self.xmin,x>self.xmax)) == True, "Interpolant out of bounds"

        ret = np.zeros_like(x)
        interp(x, ret, self.coeffs, self.xmin, self.xmax, self.f_v)
        return(ret)
