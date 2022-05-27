from numpy.testing import (
    # assert_almost_equal,
    assert_approx_equal,
    # assert_array_almost_equal,
)
import shutil
from champ.runSim import runSim
import numpy as np
import pickle
import glob
import os
import time

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
    shutil.rmtree(plotdir)


def test_layered_erosion_singleXC():
    K1 = 5e-10
    K2 = 1e-10
    sim_params = {
        "Q_w": 5.0,
        "slope": 0.001,
        "adaptive_step": False,
        "K": [K1, K2],
        "layer_elevs": [0],
    }
    r_init = 0.15
    sim = runSim(r_init=r_init, n=1, endtime=1, plotdir=plotdir, sim_params=sim_params)
    dr = np.sqrt(sim.xc.x ** 2 + sim.xc.y ** 2) - r_init
    dr_up = dr[sim.xc.y > 0].mean()
    dr_down = dr[sim.xc.y < 0].mean()
    assert_approx_equal(dr_up / dr_down, K2 / K1, 3)


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
    shutil.rmtree(plotdir)


def test_spim_equiv():
    sim_params = {"Q_w": 1, "adaptive_step": True, "xc_n": 300}
    approx_eq_slope = 7e-3
    sim = runSim(
        n=5,
        L=L_ref,
        dz=L_ref * approx_eq_slope,
        r_init=2.0,
        endtime=10000,
        dz0_dt=ref_erosion,
        plotdir=plotdir,
        run_equiv_spim=True,
        sim_params=sim_params,
        plot_every=5000,
    )
    final_spim_snap = glob.glob(os.path.join(plotdir, "spim", "snapshot*"))[-1]
    spim_f = open(final_spim_snap, "rb")
    spim = pickle.load(spim_f)
    spim_f.close()
    equil_snap = glob.glob(os.path.join(plotdir, "equil-sim", "snapshot*"))[0]
    equil_f = open(equil_snap, "rb")
    eq = pickle.load(equil_f)
    equil_f.close()
    eq_slope = eq.slopes.mean()
    eq_erosion = (eq.dz / eq.old_dt).mean()
    spim_slope = spim.slopes.mean()
    spim_erosion = (spim.dz / spim.old_dt).mean()
    sim_slope = sim.slopes.mean()
    sim_erosion = (sim.dz / sim.old_dt).mean()
    assert_approx_equal(eq_slope, spim_slope, 2)
    assert_approx_equal(eq_slope, sim_slope, 2)
    assert_approx_equal(eq_erosion, spim_erosion, 2)
    assert_approx_equal(eq_erosion, sim_erosion, 2)
    # For some reason this fails on my laptop, but not on Github. It may relate
    # to something holding the files open, but I can't figure out what.
    shutil.rmtree(plotdir)
    # This also didn't work (with up to 10 retries)
    """delete_tries = 0
    files_deleted = False
    while delete_tries < 10 and not files_deleted:
        try:
            shutil.rmtree(plotdir)
            files_deleted = True
        except:
            time.sleep(1)
            delete_tries += 1"""
