import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_approx_equal,
    assert_array_almost_equal,
)

from chansim.crossSection import CrossSection
from chansim.utils.ShapeGen import genCirc

r = 1.0
n = 1000
x, y = genCirc(r, n=n)
xc_Circ = CrossSection(x, y)


def test_xc_area():
    A = xc_Circ.calcA()
    area_analytical = np.pi * r ** 2.0
    assert_almost_equal(area_analytical, A, decimal=4)


def test_half_xc_area():
    A = xc_Circ.calcA(depth = (xc_Circ.y.max() - xc_Circ.y.min())/2)
    area_analytical = 0.5 * np.pi * r ** 2.0
    assert_approx_equal(area_analytical, A, significant=3)


def test_xc_p():
    P = xc_Circ.calcP()
    P_analytical = 2 * np.pi * r
    assert_almost_equal(P_analytical, P, decimal=4)


def test_half_xc_p():
    P = xc_Circ.calcP(wantidx=xc_Circ.y < 0.0)
    P_analytical = np.pi * r
    assert_approx_equal(P_analytical, P, significant=3)


def test_A_interp():
    fd = r * 0.5
    xc_Circ.setFD(fd)
    xc_Circ.setMaxVelPoint(fd)
    xc_Circ.create_A_interp()
    A_from_interp = xc_Circ.A_interp(fd)
    #wetidx = xc_Circ.y < xc_Circ.y.min() + fd
    A_from_calcA = xc_Circ.calcA(depth = fd)
    assert_approx_equal(A_from_interp, A_from_calcA, significant=2)


def test_P_interp():
    fd = r * 0.5
    xc_Circ.setFD(fd)
    xc_Circ.setMaxVelPoint(fd)
    xc_Circ.create_P_interp()
    P_from_interp = xc_Circ.P_interp(fd)
    wetidx = xc_Circ.y < xc_Circ.y.min() + fd
    P_from_calcP = xc_Circ.calcP(wantidx=wetidx)
    assert_approx_equal(P_from_interp, P_from_calcP, significant=3)


def test_findLR():
    L, R = xc_Circ.findLR(r)
    x_L = -1.0
    x_R = +1.0
    assert_almost_equal(xc_Circ.x[L], x_L, decimal=4)
    assert_almost_equal(xc_Circ.x[R], x_R, decimal=4)


def test_setMaxVelPoint():
    fd = 0.67 * r
    xc_Circ.setMaxVelPoint(fd)
    assert_almost_equal(xc_Circ.xmaxVel, 0, decimal=4)
    assert_approx_equal(xc_Circ.ymaxVel, xc_Circ.y.min() + fd, significant=2)


def test_findCentroid():
    cx, cy = xc_Circ.findCentroid()
    assert_almost_equal(cx, 0, decimal=4)
    assert_almost_equal(cy, 0, decimal=4)


def test_calcR_l():
    fd = r
    xc_Circ.setMaxVelPoint(fd)
    r_l = xc_Circ.calcR_l()
    r_arr = np.ones(len(xc_Circ.x)) * r
    assert_array_almost_equal(r_arr, r_l, decimal=6)


def test_calcR_l_wantidx():
    fd = r * 0.5
    xc_Circ.setMaxVelPoint(fd)
    wetidx = xc_Circ.y < xc_Circ.y.min() + fd
    r_l = xc_Circ.calcR_l(wantidx=wetidx)
    assert_almost_equal(r_l.min(), fd, decimal=2)
    theta = np.arcsin(r / 2.0)
    assert_almost_equal(r_l.max(), r * np.cos(theta), decimal=2)


def test_calcUmax_line():
    fd = r
    xc_Circ.setMaxVelPoint(fd)
    Q = 0.25
    A = 0.5 * np.pi * r ** 2.0
    u_avg = Q / A
    U_max_analytical_line = u_avg / (
        r / (r - xc_Circ.z0) - 1.0 / np.log(r / xc_Circ.z0)
    )
    xc_Circ.calcUmax(Q, method="line")
    umax = xc_Circ.umax
    assert_almost_equal(umax, U_max_analytical_line, decimal=4)


def test_calcUmax_area():
    fd = r
    xc_Circ.setMaxVelPoint(fd)
    Q = 0.25
    A = 0.5 * np.pi * r ** 2.0
    u_avg = Q / A
    U_max_analytical_area = (
        u_avg
        * np.log(r / xc_Circ.z0)
        / (
            (
                r ** 2 * np.log(r / xc_Circ.z0)
                - 1.5 * r ** 2
                + 2 * r * xc_Circ.z0
                - xc_Circ.z0 ** 2 / 2.0
            )
            / (xc_Circ.z0 ** 2 + r ** 2 - 2 * r * xc_Circ.z0)
        )
    )
    xc_Circ.calcUmax(Q, method="area")
    umax = xc_Circ.umax
    assert_almost_equal(umax, U_max_analytical_area, decimal=4)


def test_calcT_b():
    fd = r
    xc_Circ.setMaxVelPoint(fd)
    A = 0.5 * np.pi * r ** 2.0
    P_w = np.pi * r
    slope = 0.001
    xc_Circ.setEnergySlope(slope)
    rho = 1000.0  # kg/m^3
    g = 9.8  # m/s^2
    T_avg_analytical = rho * g * A * slope / P_w
    T_b = xc_Circ.calcT_b()
    T_b_avg = T_b.mean()
    assert_approx_equal(T_b_avg, T_avg_analytical, significant=3)


def test_erode_power_law():
    A = 0.5 * np.pi * r ** 2.0
    P_w = np.pi * r
    a = 1.0
    K = 1e-5
    dt = 1.0
    f = 0.1
    slope = 0.001
    xc_Circ.create_A_interp()
    xc_Circ.create_P_interp()
    xc_Circ.Q = xc_Circ.calcNormalFlow(r, slope=slope, f=f)
    rho = 1000.0  # kg/m^3
    g = 9.8  # m/s^2
    T_avg_analytical = rho * g * A * slope / P_w
    erode_avg = dt * K * T_avg_analytical ** a
    xc_Circ.erode_power_law(a=a, dt=dt, K=K)
    dr_avg = np.mean(xc_Circ.dr)
    assert_approx_equal(dr_avg, erode_avg, significant=2)


# Recreate XC after erosion
r = 1.0
n = 1000
x, y = genCirc(r, n=n)
xc_Circ = CrossSection(x, y)


def test_erode_power_law_layered():
    # Recreate XC after erosion
    r = 1.0
    n = 1000
    x, y = genCirc(r, n=n)
    xc_Circ = CrossSection(x, y)
    a = 1.0
    dt = 1.0
    f = 0.1
    slope = 0.001
    Q = 5.0
    xc_Circ.create_A_interp()
    xc_Circ.create_P_interp()
    delh = xc_Circ.calcPipeFullHeadGrad(Q)
    xc_Circ.setEnergySlope(delh)
    xc_Circ.calcNormalFlowDepth(Q, slope, f=f)
    # Zero erosion step rearranges x and y values so they will
    # align with dr after next erosion step.
    xc_Circ.erode_power_law_layered(a=a, dt=dt, K=[0, 0], layer_elevs=[0.0])
    xc_Circ.erode_power_law_layered(a=a, dt=dt, K=[1e-5, 2e-5], layer_elevs=[0.0])
    upper_avg_erosion = xc_Circ.dr[xc_Circ.y > 0].mean()
    lower_avg_erosion = xc_Circ.dr[xc_Circ.y < 0].mean()
    assert_approx_equal(upper_avg_erosion / 2, lower_avg_erosion, significant=3)


def test_calcNormalFlow():
    # Recreate XC after erosion
    r = 1.0
    n = 1000
    x, y = genCirc(r, n=n)
    xc_Circ = CrossSection(x, y)
    xc_Circ.create_A_interp()
    xc_Circ.create_P_interp()
    fd = r
    slope = 0.001
    Q_calc = xc_Circ.calcNormalFlow(fd, slope, f=0.1, use_interp=False)
    Q_by_hand = 0.98347  # Calculated by hand from D-W eqn
    assert_approx_equal(Q_calc, Q_by_hand, significant=2)


def test_calcNormalFlow_with_interp():
    # Recreate XC after erosion
    r = 1.0
    n = 1000
    x, y = genCirc(r, n=n)
    xc_Circ = CrossSection(x, y)
    fd = r
    slope = 0.001
    xc_Circ.setMaxVelPoint(fd)
    xc_Circ.create_A_interp()
    xc_Circ.create_P_interp()
    Q_calc = xc_Circ.calcNormalFlow(fd, slope, f=0.1, use_interp=True)
    Q_by_hand = 0.98347
    assert_approx_equal(Q_calc, Q_by_hand, significant=2)


def test_calcNormalFlowDepth():
    # Recreate XC after erosion
    r = 1.0
    n = 1000
    x, y = genCirc(r, n=n)
    xc_Circ = CrossSection(x, y)
    xc_Circ.create_A_interp()
    xc_Circ.create_P_interp()
    Q = 0.98347  # by hand half full circular pipe
    slope = 0.001
    fd = xc_Circ.calcNormalFlowDepth(Q, slope, f=0.1)
    assert_approx_equal(fd, 1.0, significant=2)


def test_calcPipeFullHeadGrad():
    # Recreate XC after erosion
    r = 1.0
    n = 1000
    x, y = genCirc(r, n=n)
    xc_Circ = CrossSection(x, y)
    Q = 1.0
    dh_calc = xc_Circ.calcPipeFullHeadGrad(Q, f=0.1)
    dh_by_hand = 0.00025847
    assert_approx_equal(dh_calc, dh_by_hand, significant=3)


def test_erode():
    # Recreate XC after erosion
    r = 1.0
    n = 1000
    x, y = genCirc(r, n=n)
    xc_Circ = CrossSection(x, y)
    x, y = genCirc(r, n=n)
    xc_Circ = CrossSection(x, y)
    erode_factor = 0.05
    xc_Circ.setMaxVelPoint(r * 2)
    r_l = xc_Circ.calcR_l()
    dr = r_l * erode_factor
    A_before = xc_Circ.calcA()
    xc_Circ.erode(dr, trim=False)
    A_after = xc_Circ.calcA()
    assert_approx_equal(
        A_before / A_after, (1 / (1 + erode_factor)) ** 2, significant=4
    )


def test_erode_half():
    # Recreate XC after erosion
    r = 1.0
    n = 1000
    x, y = genCirc(r, n=n)
    xc_Circ = CrossSection(x, y)
    x, y = genCirc(r, n=n)
    xc_Circ = CrossSection(x, y)
    erode_factor = 0.05
    xc_Circ.setFD(r)
    xc_Circ.setMaxVelPoint(r)
    wetidx = xc_Circ.wetidx
    r_l = xc_Circ.calcR_l(wantidx=wetidx)
    dr = r_l * erode_factor
    A_before = xc_Circ.calcA()
    xc_Circ.erode(dr, trim=False)
    A_after = xc_Circ.calcA()
    assert_approx_equal(
        A_before / A_after, (2 / (1 + (1 + erode_factor) ** 2)), significant=4
    )
