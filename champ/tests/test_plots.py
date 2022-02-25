import numpy as np
import os
import shutil
import glob
from champ.sim import multiXC
from champ.viz.standard_timestep_plots import make_all_standard_timestep_plots
from champ.viz.make_spim_plots_from_snapshots import make_spim_plots_from_snapshots
from champ.viz.make_mayavi_frames import make_frames
from champ.runSim import runSim


slope_ref = 0.01
Q_ref = 1.0
L_ref = 1000
dz_ref = L_ref * slope_ref
w_ref = 1.77217  # Reference value from test simulation.
ref_erosion = 0.000270  # m/yr

g = 9.8
L_per_m3 = 1000.0
f = 0.1
slope = 0.001
r = 1.0
Q_half_circle = np.sqrt(slope * np.pi ** 2 * g * r ** 5 / f)
plotdir = "./tests/test-figs/"
timestep_str = "1"


def test_make_all_timestep_plots():
    n = 5
    x = np.linspace(0, 5000, n)
    z = x * slope
    init_radii = r * np.ones(n - 1)
    sim = multiXC(x, z, init_radii=init_radii, Q_w=Q_half_circle, xc_n=500, f=f)
    sim.calc_flow()
    sim.erode()
    make_all_standard_timestep_plots(sim, plotdir, timestep_str)
    assert os.path.isfile(os.path.join(plotdir, "XC-" + timestep_str + ".png"))
    assert os.path.isfile(
        os.path.join(plotdir, "Elevation-Profile-" + timestep_str + ".png")
    )
    assert os.path.isfile(os.path.join(plotdir, "Slope-" + timestep_str + ".png"))
    assert os.path.isfile(os.path.join(plotdir, "3D-XC-" + timestep_str + ".png"))
    shutil.rmtree(plotdir)


def test_spim_plots():
    sim_params = {"Q_w": 1, "adaptive_step": False}
    runSim(
        n=5,
        L=L_ref,
        dz=dz_ref,
        r_init=2.0,
        endtime=50,
        dz0_dt=ref_erosion,
        plotdir=plotdir,
        snapshot_every=10,
        snapshot_by_years=False,
        plot_every=10,
        plot_by_years=False,
        sim_params=sim_params,
    )
    make_spim_plots_from_snapshots(plotdir, every=1)
    assert os.path.isfile(os.path.join(plotdir, "1-Erosion-slope-width-v-distance.png"))
    assert os.path.isfile(os.path.join(plotdir, "1-Morphodynamics.png"))
    assert os.path.isfile(os.path.join(plotdir, "1-Final-Morphology.png"))
    shutil.rmtree(plotdir)


def test_make_mayavi_frames():
    sim_params = {
        "Q_w": 1,
        "adaptive_step": True,
        "layer_elevs": [0, -5, -15],
        "K": [1e-5, 2e-5, 1e-5, 2e-5],
    }
    runSim(
        n=5,
        L=L_ref,
        dz=dz_ref,
        r_init=2.0,
        endtime=50,
        dz0_dt=ref_erosion,
        plotdir=plotdir,
        snapshot_every=10,
        snapshot_by_years=False,
        plot_every=10,
        plot_by_years=False,
        sim_params=sim_params,
    )
    make_frames(plotdir, 0)
    snaps = glob.glob(plotdir + "*.pkl")
    nsnaps = len(snaps)
    frames = glob.glob(os.path.join(plotdir, "anim", "*.png"))
    nframes = len(frames)
    assert nsnaps == nframes
    # shutil.rmtree(plotdir)
