import numpy as np
from numpy.testing import assert_allclose

from champ.sim import singleXC, multiXC

g = 9.8
L_per_m3 = 1000.0
f = 0.1
slope = 0.001
r = 1.0
Q_half_circle = np.sqrt(slope * np.pi ** 2 * g * r ** 5 / f)


def test_singleXC_calc_flow():
    sim = singleXC(init_radius=r, Q_w=Q_half_circle, slope=slope, xc_n=1500, f=f)
    sim.calc_flow()
    fd = sim.xc.fd
    fd_by_hand = 1.0
    assert_allclose(fd, fd_by_hand, rtol=0.011)


def test_multiXC_calc_flow():
    n = 5
    x = np.linspace(0, 5000, n)
    z = x * slope
    init_radii = r * np.ones(n - 1)
    sim = multiXC(x, z, init_radii=init_radii, Q_w=Q_half_circle, xc_n=1500, f=f)
    sim.calc_flow()
    fds = sim.fd_mids
    fds_by_hand = 1.0 * np.ones(n - 1)
    assert_allclose(fds, fds_by_hand, rtol=0.011)


def test_singleXC_calc_flow_full_pipe():
    Q = 5.0
    sim = singleXC(init_radius=r, slope=slope, Q_w=Q, f=f, xc_n=1500)
    sim.calc_flow()
    del_h = Q ** 2.0 / (np.pi * r ** 2) ** 2 * (f / (2 * g * 2 * r))
    assert_allclose(sim.xc.eSlope, del_h, rtol=0.001)


def test_multiXC_calc_flow_full_pipe():
    n = 5
    x = np.linspace(0, 5000, n)
    z = x * slope
    init_radii = r * np.ones(n - 1)
    Q = 5.0
    sim = multiXC(x, z, init_radii=init_radii, Q_w=Q, f=f, xc_n=1500)
    sim.calc_flow()
    del_h = (
        Q ** 2.0 / (np.pi * init_radii[0] ** 2) ** 2 * (f / (2 * g * 2 * init_radii[0]))
    )
    h_analytical = x * del_h + 1.0
    assert_allclose(sim.h, h_analytical, rtol=0.001)


def test_multiXC_calc_flow_full_pipe_varied_r():
    n = 5
    x = np.linspace(0, 5000, n)
    slope = 0.001
    z = x * slope
    init_radii = r * np.linspace(0.5, 1.5, n - 1)
    Q = 5.0
    f = 0.1
    sim = multiXC(x, z, init_radii=init_radii, Q_w=Q, f=f, xc_n=1500)
    sim.calc_flow()
    del_h = Q ** 2.0 / (np.pi * init_radii ** 2) ** 2 * (f / (2 * g * 2 * init_radii))
    h_analytical = np.zeros(n)
    h_analytical[1:] = np.cumsum(del_h) * sim.L_arr
    h_analytical += 2 * init_radii[0] + sim.z_arr[0]
    assert_allclose(sim.h, h_analytical, rtol=0.001)


def test_multiXC_backflooded_flow():
    n = 5
    x = np.linspace(0, 5000, n)
    slope = 0.001
    z = x * slope
    init_radii = [0.1, 1, 1, 1]
    Q = 0.01
    f = 0.1
    sim = multiXC(x, z, init_radii=init_radii, Q_w=Q, f=f, xc_n=1500)
    sim.calc_flow()
    delh = (sim.h[1:] - sim.h[:-1]) / sim.L_arr
    V_bw = np.sqrt(2 * g * delh * sim.D_H_w / f)
    V = V_bw
    V_norm = np.sqrt(2 * g * sim.slopes * sim.D_H_w / f)
    V[sim.flow_type == "norm"] = V_norm[sim.flow_type == "norm"]
    assert_allclose(V * sim.A_w, Q * np.ones(n - 1), rtol=0.05)
