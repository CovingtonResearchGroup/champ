from numpy.testing import (
    # assert_almost_equal,
    assert_approx_equal,
    # assert_array_almost_equal,
)

from champ.runSim import runSim

slope_ref = 0.01
Q_ref = 1.0
L_ref = 1000
dz_ref = L_ref * slope_ref
w_ref = 1.77217  # Reference value from test simulation.
ref_erosion = 0.000270  # m/yr
plotdir = "./tests/test-figs/"


def test_equil_singleXC():
    sim_params = {"Q_w": Q_ref, "slope": slope_ref, "adaptive_step": True}
    sim = runSim(r_init=2.0, n=1, endtime=5000, plotdir=plotdir, sim_params=sim_params)
    L, R = sim.xc.findLR(sim.xc.fd)
    w = sim.xc.x[R] - sim.xc.x[L]
    assert_approx_equal(w, w_ref, 3)


def test_equil_multiXC():
    sim_params = {"Q_w": 1, "adaptive_step": True}
    sim = runSim(
        n=5,
        L=L_ref,
        dz=dz_ref,
        r_init=2.0,
        endtime=1000,
        dz0_dt=ref_erosion,
        plotdir=plotdir,
        sim_params=sim_params,
    )

    assert_approx_equal(sim.slopes[1:].mean(), slope_ref, 2)
