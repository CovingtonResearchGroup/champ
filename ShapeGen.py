from numpy import sin, cos, pi, sqrt, fabs, arctan2, linspace
import numpy as np


# Generate x, y points for an ellipse. Can set rotation angle with theta
def genEll(r1,r2,theta=0,n=1000):

	t=linspace(0, 2*pi-2*pi/n, n-1)
	x=r1*cos(t)
	y=r2*sin(t)


	if(theta!=0):

		tx=x
		ty=y

		x=tx*cos(theta)-ty*sin(theta)
		y=tx*sin(theta)+ty*cos(theta)

	return x, y

# Generate x, y points for a circle
def genCirc(r,n=1000):

	return genEll(r, r,n=n)

# Generate x, y points for the lower half of a circle
def genSemiCirc(r,n=1000):

	return genSemiEll(r, r,n=n)

# Generate x, y points for the lower half of an ellipse
def genSemiEll(r1,r2,n=1000):

	t=linspace(-1*pi, 0, n)
	x=r1*cos(t)
	y=r2*sin(t)

	return x, y
