from champ.sim import multiXCmultiQ
from champ.runSim import runSim
import matplotlib.pyplot as plt
import numpy as np

sim_params = {
    "adaptive_step": True,
    "K": [5e-7, 5e-7, 5e-7],
    "a": 1.75,
    "layer_elevs": [-200, -100],
    "uplift": 0.0001,
    "nQ": 20,
    "Q_mean": 0.1,
    "Q_min_mult": 0.1,
    "Q_max_mult": 30,
    "T_c": 200,
    "max_frac_erode": 0.01,
    "xc_n": 500,
    "layer_solubility": [False, True, False],
    "K_sol": 1e-5,
    "a_sol": 0.5,
}
sim2 = runSim(
    n=20,
    L=1000,
    endtime=10000000,
    plotdir="./champ/Examples/yaml-files/boone-v6-normed/",
    multiQ=True,
    plot_every=1000,
    snapshot_every=1000,
    r_init=1,
    n_plot_processes=2,
    sim_params=sim_params,
    start_from_snapshot_num=3240000,
)
