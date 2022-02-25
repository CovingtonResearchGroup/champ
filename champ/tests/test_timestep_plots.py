import numpy as np
from champ.sim import multiXC
from champ.viz.standard_timestep_plots import make_all_standard_timestep_plots
import os

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
    sim = multiXC(x, z, init_radii=init_radii, Q_w=Q_half_circle, xc_n=1500, f=f)
    sim.calc_flow()
    sim.erode()
    make_all_standard_timestep_plots(sim, plotdir, timestep_str)
    assert os.path.isfile(os.path.join(plotdir, "XC-" + timestep_str + ".png"))
    assert os.path.isfile(
        os.path.join(plotdir, "Elevation-Profile-" + timestep_str + ".png")
    )
    assert os.path.isfile(os.path.join(plotdir, "Slope-" + timestep_str + ".png"))
    assert os.path.isfile(os.path.join(plotdir, "3D-XC-" + timestep_str + ".png"))
