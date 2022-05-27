"""Functions for creating plots from simulation snapshots. These
plots involve comparisons of pairs of simulations (normally with and
without CO2 exhange.) Designed to be run from command line with CO2 plotdir
as first command line argument and No CO2 exchange plotdir as second command
line argument. An optional third command line argument specifies the number
of snapshots to skip while plotting results.
"""

from glob import glob
import pickle
from matplotlib import ticker, rcParams, rc
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import sys
import os


font = {"size": 14}

rc("font", **font)


def get_results(plotdir):
    """Load results from snapshot files.

    Parameters
    ----------
    plotdir : string
        Name of directory containing snapshots.

    Returns
    -------
    results_dict : dict
        Dictionary of simulation results.

    """

    snapshots = glob(plotdir + "*.pkl")
    snapshots.sort()
    erosion = []
    width = []
    slope = []
    A = []
    Pw = []
    fd = []
    T_b_mean = []
    T_b_max = []
    years = []
    x = []
    z = []
    snap_every = 1000  # Note this seems hard-coded and should be fixed
    for snapnum, snapshot in enumerate(snapshots):
        print("loading snapshot ", snapnum)
        f = open(snapshot, "rb")
        snap_sim = pickle.load(f)
        f.close()
        years.append(snapnum * snap_every * snap_sim.dt_erode)
        erosion.append(snap_sim.dz / snap_sim.dt_erode)
        A.append(snap_sim.A_w)
        Pw.append(snap_sim.P_w)
        fd.append(snap_sim.fd_mids)
        slope.append(snap_sim.slopes)
        # Get data from XCs
        these_widths = np.zeros(len(snap_sim.xcs))
        these_T_b_mean = np.zeros(len(snap_sim.xcs))
        these_T_b_max = np.zeros(len(snap_sim.xcs))
        for i, xc in enumerate(snap_sim.xcs):
            these_T_b_mean[i] = xc.T_b.mean()
            these_T_b_max[i] = xc.T_b.max()
            if snap_sim.flow_type[i] != "full":
                L, R = xc.findLR(xc.fd)
                these_widths[i] = xc.x[R] - xc.x[L]
            else:
                these_widths[i] = 0.0
        # Get data from nodes
        width.append(these_widths)
        T_b_mean.append(these_T_b_mean)
        T_b_max.append(these_T_b_max)
        x.append(snap_sim.x_arr)
        z.append(snap_sim.z_arr)
        if snapnum == 0:
            year0 = years[0]
            z_base0 = snap_sim.z_arr[0]
        if snapnum == 1:
            year1 = years[1]
            z_base1 = snap_sim.z_arr[0]
            dz_dt_base = (z_base1 - z_base0) / (year1 - year0)

    erosion = np.array(erosion)
    width = np.array(width)
    slope = np.array(slope)
    A = np.array(A)
    Pw = np.array(Pw)
    fd = np.array(fd)
    T_b_mean = np.array(T_b_mean)
    T_b_max = np.array(T_b_max)
    years = np.array(years)
    x = np.array(x)
    z = np.array(z)
    return {
        "erosion": abs(erosion),
        "width": width,
        "slope": slope,
        "A": A,
        "Pw": Pw,
        "fd": fd,
        "T_b_mean": T_b_mean,
        "T_b_max": T_b_max,
        "years": years,
        "x": x,
        "z": z,
        "dz_dt_base": dz_dt_base,
    }


def plot_erosion_slope_width_over_distance(snapshotdir, res, every=5):
    """
    Make plot of Erosion, slope,  width over time
    """

    cmap = plt.get_cmap("nipy_spectral")
    n_steps = int(np.ceil(res["erosion"].shape[0] / every))
    cyc = plt.cycler(color=[cmap(k) for k in np.linspace(0, 1, n_steps)])
    rcParams["axes.prop_cycle"] = cyc

    max_years = res["years"][-1]
    cNorm = plt.Normalize(vmin=0, vmax=max_years)
    sMap = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap)

    cb_formatter = ticker.ScalarFormatter()
    # cb_formatter.set_scientific(True)
    cb_formatter.set_powerlimits((0, 0))
    cb_formatter.set_useMathText(True)

    x = (res["x"][0, :-1] + res["x"][0, 1:]) / 2.0

    fig, axs = plt.subplots(3, 1, figsize=(6, 9), sharex=True)

    axs[0].plot(x, abs(res["erosion"].transpose()[::, ::every]))
    axs[0].set_ylabel("Erosion (m/yr)")
    ymin = 0.9 * res["erosion"].min()
    ymax = 1.1 * res["erosion"].max()
    axs[0].set_ylim([ymin, ymax])
    axs[0].grid("--", which="both")
    axs[0].tick_params(direction="in", labelsize=10)
    axs[0].plot([x[0], x[-1]], [-res["dz_dt_base"], -res["dz_dt_base"]], "--k")
    cb = plt.colorbar(sMap, ax=axs[0], format=cb_formatter)
    cb.set_label("Years")

    axs[1].semilogy(x, abs(res["slope"].transpose()[::, ::every]))
    axs[1].set_ylabel("Slope")
    ymin = 0.9 * res["slope"].min()
    ymax = 1.1 * res["slope"].max()
    axs[1].set_ylim([ymin, ymax])
    axs[1].grid("--", which="both")
    axs[1].tick_params(direction="in", labelsize=10)

    cb = plt.colorbar(sMap, ax=axs[1], format=cb_formatter)
    cb.set_label("Years")

    axs[2].plot(x, abs(res["width"].transpose()[::, ::every]))
    axs[2].set_xlabel("Distance upstream (m)")
    axs[2].set_ylabel("Width (m)")
    ymin = 0.9 * res["width"].min()
    ymax = 1.1 * res["width"].max()
    axs[2].set_ylim([ymin, ymax])
    axs[2].grid("--")
    axs[2].tick_params(direction="in", labelsize=10)
    cb = plt.colorbar(sMap, ax=axs[2], format=cb_formatter)
    cb.set_label("Years")

    plt.tight_layout(h_pad=0.0, w_pad=0.0)

    plt.savefig(os.path.join(snapshotdir, "1-Erosion-slope-width-v-distance.png"))


def plot_morphodynamics(snapshotdir, res):
    cmap = plt.get_cmap("viridis")
    n_xc = res["erosion"].shape[1]
    rcParams["axes.prop_cycle"] = plt.cycler(
        color=[cmap(k) for k in np.linspace(0, 1, n_xc)]
    )

    cNorm = plt.Normalize(vmin=0, vmax=res["x"][0, -1])
    plt.cm.ScalarMappable(norm=cNorm, cmap=cmap)

    fig, axs = plt.subplots(3, 1, figsize=(5, 12), sharex=True)

    axs[0].semilogy(res["years"], res["slope"])
    axs[0].set_ylabel("Slope")
    ymin = 0.9 * res["slope"].min()
    ymax = 1.1 * res["slope"].max()
    axs[0].set_ylim([ymin, ymax])
    axs[0].tick_params(labelsize=10)

    axs[1].plot(res["years"], res["width"])
    axs[1].set_ylabel("Width (m)")
    ymin = 0.9 * res["width"].min()
    ymax = 1.051 * res["width"].max()
    axs[1].set_ylim([ymin, ymax])
    axs[1].tick_params(labelsize=10)

    axs[2].plot(res["years"], abs(res["erosion"]))
    axs[2].set_xlabel("Years")
    axs[2].set_ylabel("Erosion (m/yr)")
    ymin = res["erosion"].min()
    ymax = 1.1 * res["erosion"].max()
    axs[2].set_ylim([ymin, ymax])
    axs[2].plot([0, 100000], [0.00125, 0.00125], "k--")
    axs[2].plot(
        [res["years"][0], res["years"][-1]],
        [-res["dz_dt_base"], -res["dz_dt_base"]],
        "--k",
    )
    axs[2].tick_params(labelsize=10)

    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig(os.path.join(snapshotdir, "1-Morphodynamics.png"))


def final_step_morphology(snapshotdir, res):

    plt.figure()
    plt.semilogy(res["width"][-1], res["slope"][-1], "ko")
    w_i = res["width"][-1][0]
    s_i = res["slope"][-1][0]
    w_f = res["width"][-1][-1]
    w_opt = np.linspace(w_i, w_f, 20)
    C_prop = s_i * w_i ** (16.0 / 3.0)
    s_opt = C_prop * w_opt ** (-16.0 / 3.0)
    plt.plot(w_opt, s_opt, "--k")
    m, b, r, p, err = linregress(np.log10(res["width"][-1]), np.log10(res["slope"][-1]))
    plt.plot(res["width"][-1], res["width"][-1] ** m * 10 ** b, ":k")

    plt.xlabel("Width (m)")
    plt.ylabel("Slope")
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    opt = -16.0 / 3.0
    opt_str = r"$S \propto w^{" + str(opt)[:5] + "}$"
    fit_str = r"$S \propto w^{" + str(m)[:5] + "}$"

    plt.legend(["With CO2", "Optimal: " + opt_str, "Fit: " + fit_str])
    plt.tight_layout()
    plt.savefig(os.path.join(snapshotdir, "1-Final-Morphology.png"))

    # s_f = co2['slope'][-1][-1]


def make_spim_plots_from_snapshots(snapshotdir, every=10):
    print("Loading results...")
    res = get_results(snapshotdir)

    plot_erosion_slope_width_over_distance(snapshotdir, res, every=every)
    plot_morphodynamics(snapshotdir, res)
    final_step_morphology(snapshotdir, res)
    # slope_width_all_times(snapshotdir, co2, noco2)


if __name__ == "__main__":
    snapshotdir = sys.argv[1]
    if len(sys.argv) > 3:
        every = int(sys.argv[3])
    else:
        every = 5
    make_spim_plots_from_snapshots(snapshotdir, every=every)
