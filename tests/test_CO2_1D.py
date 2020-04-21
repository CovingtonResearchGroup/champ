import numpy as np
from numpy.testing import *

from CO2_sim_1D import CO2_1D


def test_calc_flow_depth():
    n=5
    x = np.linspace(0, 5000,n)
    slope = 0.001
    z = x*slope
    r = 1.*np.ones(n-1)
    Q = 0.98347
    sim = CO2_1D(x, z, init_radii=r, Q_w = Q)
    sim.calc_flow_depths()
    fds = sim.fd_mids
    fds_by_hand = 1.0*np.ones(n-1)
    assert_allclose(fds, fds_by_hand, rtol=0.001)


def test_calc_flow_depth():
    n=5
    x = np.linspace(0, 5000,n)
    slope = 0.001
    z = x*slope
    r = 1.*np.ones(n-1)
    Q_low = 0.000001
    Q_half = 0.98347
    sim = CO2_1D(x, z, init_radii=r, Q_w = Q_half,
                T_cave = 10., dH=100., T_outside=20.)
    #Bring water up to half full (checked in prev test)
    sim.calc_flow_depths()
    sim.calc_air_flow()
    assert_approx_equal(sim.Q_a, -0.81233, significant=3)
    #Empty pipe
    sim.Q_w = Q_low
    sim.calc_flow_depths()
    sim.calc_air_flow()
    assert_approx_equal(sim.Q_a, -1.624661, significant=3)
