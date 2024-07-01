"""Functions to generate cross-section x-y pairs for different shapes."""

from numpy import sin, cos, pi, linspace, zeros, round, ones, concatenate

shape_name_map = {"trapezoid": "genTrap",
                  "ellipse": "genEll",
                  "semi-ellipse":"genSemiEll",
                  "circle": "genCirc",
                  "semi-circle": "genSemiCirc",
                  "rectangular":"genRect",}


def name_to_function(name):
    return shape_name_map[name]


# Generate x, y points for an ellipse. Can set rotation angle with theta
def genEll(r1, r2, theta=0, n=1000):
    """Generate an ellipse.

    Parameters
    ----------
    r1 : float
        x intercept of ellipse.
    r2 : float
        y intercept of ellipse.
    theta : float
        rotation angle of ellipse.
    n : int
        Number of points in cross-section.
    Returns
    -------
    x,y : ndarray
        x and y values for points in cross-section.
    """

    t = linspace(0, 2 * pi - 2 * pi / n, n - 1)
    x = r1 * cos(t)
    y = r2 * sin(t)

    if theta != 0:

        tx = x
        ty = y

        x = tx * cos(theta) - ty * sin(theta)
        y = tx * sin(theta) + ty * cos(theta)

    return x, y


# Generate x, y points for a circle
def genCirc(r, n=1000):
    """Generate a circle.

    Parameters
    ----------
    r : float
        Radius of circle.
    n : int
        Number of points in cross-section.
    Returns
    -------
    x,y : ndarray
        x and y values for points in cross-section.
    """

    return genEll(r, r, n=n)


# Generate x, y points for the lower half of a circle
def genSemiCirc(r, n=1000):
    """Generate a semi-circle.

    Parameters
    ----------
    r : float
        Radius of circle.
    n : int
        Number of points in cross-section.
    Returns
    -------
    x,y : ndarray
        x and y values for points in cross-section.
    """

    return genSemiEll(r, r, n=n)


# Generate x, y points for the lower half of an ellipse
def genSemiEll(r1, r2, n=1000):
    """Generate a half ellipse.

    Parameters
    ----------
    r1 : float
        x intercept of ellipse.
    r2 : float
        y intercept of ellipse.
    theta : floatw_h_ratio = width/height
        rotation angle of ellipse.
    n : int
        Number of points in cross-section.
    Returns
    -------
    x,y : ndarray
        x and y values for points in cross-section.
    """

    t = linspace(-1 * pi, 0, n)
    x = r1 * cos(t)
    y = r2 * sin(t)

    return x, y


def genTrap(bottom_width=1, side_slope=1, height=1, n=1000):
    """Generate an trapezoidal open channel.

    Parameters
    ----------
    bottom_width : float
        Width of flat bed of trapezoid.
    side_slope : float
        slope of trapezoid sides.
    height : float
        total height of sides.
    n : int
        Number of points in cross-section.
    Returns
    -------
    x,y : ndarray
        x and y values for points in cross-section.
    """

    b = bottom_width
    z = side_slope
    h = height
    x0 = -b / 2 - z * h
    x = linspace(x0, -x0, n)
    y = zeros(n)
    y[x < -b / 2] = -(x[x < -b / 2] + b / 2) / z
    y[x > b / 2] = (x[x > b / 2] - b / 2) / z

    return x, y



def genRect(width=1, height=1, n=500):
    w_h_ratio = width/height
    n_height = int(round( (n/(w_h_ratio + 1))/2))
    n_width = int((n - 2*n_height)/2)
    x1 = zeros(n_height)
    x2 = linspace(0,width,n_width+1)
    x3 = width*ones(n_height+1)
    x4 = linspace(width,0,n_width+1)
    y1 = linspace(height,0,n_height)
    y2 = zeros(n_width+1)
    y3 = linspace(0,height,n_height+1)
    y4 = height*ones(n_width+1)
    x = concatenate([x1,x2[1:],x3[1:],x4[1:]])
    y = concatenate([y1,y2[1:],y3[1:],y4[1:]])
    x = x - width/2
    return x, y
