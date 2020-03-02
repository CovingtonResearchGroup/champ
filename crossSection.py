from numpy import sin, cos, pi, fabs, sign, roll, arctan2, diff, sum, hypot,\
                    logical_and, where, linspace, sqrt
from scipy import interpolate
from scipy.optimize import brentq
import matplotlib.pyplot as plt


g=9.8 #m/s^2

class CrossSection:

    def __init__(self, x, y):
        self.n = len(x)
        self.x = x
        self.y = y
        self.ymin = min(y)
        self.ymax = max(y)
        self.create_pm()

    # Create arrays of x+1, y+1, x-1, x+1
    def create_pm(self):
        self.xm = roll(self.x, 1)
        self.ym = roll(self.y, 1)
        self.xp = roll(self.x, self.x.size-1)
        self.yp = roll(self.y, self.y.size-1)

    def calcP(self, wantidx=None):
        if type(wantidx) != type(None):
            l = hypot(self.x[wantidx] - self.xp[wantidx], self.y[wantidx] - self.yp[wantidx])
        else:
            l = hypot(self.x - self.xp, self.y - self.yp)

        P = sum(l)
        return P
        #self.pp = cumsum(self.l)
		#self.P = self.pp[-2]

	# Calculates area of the cross-section
    def calcA(self, wantidx=None):
        if type(wantidx) != type(None):
            self.sA = (self.xm[wantidx]*self.y[wantidx] - self.x[wantidx]*self.ym[wantidx]).sum() * 0.5
        else:
            self.sA = (self.xm*self.y - self.x*self.ym).sum() * 0.5
        A = fabs(self.sA)
        return A

	# Find left and right points defining a height above the cross-section
	# bottom
    def findLR(self, h):
        a_h = self.ymin + h
        condL = logical_and(self.y > a_h, a_h > self.yp)
        condR = logical_and(self.y < a_h, a_h < self.yp)

        L = where(condL)[0][0] + 1
        R = where(condR)[0][0]
        return L,R

    def calcNormalFlow(self,depth, slope,f=0.1):
        wetidx = self.y - self.ymin < depth
        Pw = self.calcP(wantidx=wetidx)
        A = self.calcA(wantidx=wetidx)
        D_H = 4.*A/Pw
        Q = A*sqrt(2.*g*slope*D_H/f)
        return Q

    def discharge_residual(self, depth, slope, f, desiredQ):
        return desiredQ - self.calcNormalFlow(depth,slope,f=f)

    def calcFlowDepth(self,Q, slope,f=0.1):
        maxdepth = self.ymax - self.ymin
        calcFullFlow = self.calcNormalFlow(maxdepth,slope, f=f)
        if Q>calcFullFlow:
            calc95PerFlow = self.calcNormalFlow(0.95*maxdepth, slope,f=f)
            if Q>calc95PerFlow:
                print("Pipe is full.")
                return -1
        else:
            fd = brentq(self.discharge_residual, 1e-6, maxdepth, args=(slope,f,Q))
        return fd

    def calcPipeFullHeadGrad(self,Q,slope,f=0.1):
        Pw = self.calcP()
        A = self.calcA()
        D_H = 4.*A/Pw
        return (Q**2/A**2)*f/(2.*g*D_H)
