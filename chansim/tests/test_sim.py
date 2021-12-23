import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_approx_equal,
    assert_array_almost_equal,
    assert_allclose,
)

from chansim.sim import singleXC, multiXC

g = 9.8
L_per_m3 = 1000.0


def test_multiXC_calc_flow():
    n = 5
    x = np.linspace(0, 5000, n)
    slope = 0.001
    z = x * slope
    r = 1.0 * np.ones(n - 1)
    Q = 0.98347
    sim = multiXC(x, z, init_radii=r, Q_w=Q, xc_n=1500)
    sim.calc_flow()
    fds = sim.fd_mids
    fds_by_hand = 1.0 * np.ones(n - 1)
    assert_allclose(fds, fds_by_hand, rtol=0.011)


def test_multiXC_calc_flow_full_pipe():
    n = 5
    x = np.linspace(0, 5000, n)
    slope = 0.001
    z = x * slope
    r = 1.0 * np.ones(n - 1)
    Q = 5.0
    f = 0.1
    sim = multiXC(x, z, init_radii=r, Q_w=Q, f=f, xc_n=1500)
    sim.calc_flow()
    del_h = Q ** 2.0 / (np.pi * r[0] ** 2) ** 2 * (f / (2 * g * 2 * r[0]))
    h_analytical = x * del_h + 1.0
    assert_allclose(sim.h, h_analytical, rtol=0.001)


def test_multiXC_calc_flow_full_pipe_varied_r():
    n = 5
    x = np.linspace(0, 5000, n)
    slope = 0.001
    z = x * slope
    r = 1.0 * np.linspace(0.5, 1.5, n - 1)
    Q = 5.0
    f = 0.1
    sim = multiXC(x, z, init_radii=r, Q_w=Q, f=f, xc_n=1500)
    sim.calc_flow()
    del_h = Q ** 2.0 / (np.pi * r ** 2) ** 2 * (f / (2 * g * 2 * r))
    h_analytical = np.zeros(n)
    h_analytical[1:] = np.cumsum(del_h) * sim.L_arr
    h_analytical += 2 * r[0] + sim.z_arr[0]
    assert_allclose(sim.h, h_analytical, rtol=0.001)

