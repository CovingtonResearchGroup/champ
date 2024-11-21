from champ.sim import multiXCmultiQ
from champ.runSim import runSim
import matplotlib.pyplot as plt
import numpy as np

sim_params = {
    "adaptive_step": True,
    "K": [2e-5, 2e-5, 2e-5],
    "a": 1.5,
    "layer_elevs": [-4, -2],
    "uplift": 0.00005,
    "nQ": 10,
    "Q_min_mult": 0.1,
    "Q_max_mult": 10,
    "T_c": 5,
    "max_frac_erode": 0.005,
    "xc_n": 1000,
    "layer_solubility": [False, True, False],
    "K_sol": 1e-5,
}
sim2 = runSim(
    n=10,
    endtime=140000,
    plotdir="./multiXCmultiQ_Ksol/",
    multiQ=True,
    plot_every=500,
    n_plot_processes=4,
    sim_params=sim_params,
    start_from_snapshot_num=75000,
)
