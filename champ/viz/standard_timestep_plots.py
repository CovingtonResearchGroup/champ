from pylab import (
    figure,
    plot,
    xlabel,
    ylabel,
    savefig,
    legend,
    close,
    yscale,
    tight_layout,
    xlim,
    ceil,
    arange,
)
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os


def make_all_standard_timestep_plots(sim, plotdir, timestep_str, xmax=5):
    # Check whether plotdir exists
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    # Avoid opening GUI
    matplotlib.use("Agg")
    plot_overlapping_XCs(sim, plotdir, timestep_str)
    if not sim.singleXC:
        plot_elevation_profile(sim, plotdir, timestep_str)
        plot_slope_profile(sim, plotdir, timestep_str)
        plot_3D_XCs(sim, plotdir, timestep_str, xmax=xmax)
    # close("all")


def plot_overlapping_XCs(sim, plotdir, timestep_str):
    fig = figure()
    if not sim.singleXC:
        n = len(sim.xcs)
        if sim.xcs[0].x_total is not None:
            plot(sim.xcs[0].x_total, sim.xcs[0].y_total)
        plot(sim.xcs[0].x, sim.xcs[0].y)
        wl = sim.xcs[0].fd + sim.xcs[0].y.min()
        plot([-0.5, 0.5], [wl, wl])
        if sim.xcs[int(ceil(n / 2.0))].x_total is not None:
            plot(
                sim.xcs[int(ceil(n / 2.0))].x_total, sim.xcs[int(ceil(n / 2.0))].y_total
            )
        plot(sim.xcs[int(ceil(n / 2.0))].x, sim.xcs[int(ceil(n / 2.0))].y)
        if sim.xcs[-1].x_total is not None:
            plot(sim.xcs[-1].x_total, sim.xcs[-1].y_total)
        plot(sim.xcs[-1].x, sim.xcs[-1].y)
    else:
        if sim.xc.x_total is not None:
            plot(sim.xc.x_total, sim.xc.y_total)
        plot(sim.xc.x, sim.xc.y)
        wl = sim.xc.fd + sim.xc.y.min()
        plot([-0.5, 0.5], [wl, wl])
    xlabel("Cross-channel distance (m)")
    ylabel("Relative elevation (m)")
    savefig(os.path.join(plotdir, "XC-" + timestep_str + ".png"))
    close(fig)


def plot_elevation_profile(sim, plotdir, timestep_str, with_h=True):
    fig = figure()
    if with_h:
        plot(sim.x_arr, sim.h)
    plot(sim.x_arr, sim.z_arr)
    xlabel("Distance (m)")
    ylabel("Elevation (m)")
    legend(["h", "z"])
    savefig(os.path.join(plotdir, "Elevation-Profile-" + timestep_str + ".png"))
    close(fig)


def plot_concentration_profile(sim, plotdir, timestep_str):
    fig = figure()
    plot(sim.x_arr, sim.CO2_w)
    plot(sim.x_arr, sim.CO2_a)
    plot(sim.x_arr, sim.Ca)
    legend(["w", "a", "Ca"])
    savefig(os.path.join(plotdir, "Concentration-Profile-" + timestep_str + ".png"))
    close(fig)


def plot_slope_profile(sim, plotdir, timestep_str):
    fig = figure()
    xmid = (sim.x_arr[1:] + sim.x_arr[:-1]) / 2.0
    plot(xmid, sim.slopes)
    if len(xmid) == len(sim.dz):
        plot(xmid, abs(sim.dz) / sim.old_dt)
    else:
        plot(sim.x_arr, abs(sim.dz) / sim.old_dt)
    yscale("log")
    xlabel("Distance (m)")
    ylabel("Slope/Erosion rate (m/yr)")
    legend(["slope", "erosion rate"])
    tight_layout()
    savefig(os.path.join(plotdir, "Slope-" + timestep_str + ".png"))
    close(fig)


def plot_3D_XCs(sim, plotdir, timestep_str, xmax=5):
    # Create 3d XC fig
    fig = figure()
    ax = fig.add_subplot(111, projection="3d")
    xLs = []
    xRs = []
    xcs = sim.xcs
    n = len(sim.xcs)
    for i in arange(n - 1):
        if xcs[i].x_total is not None:
            x_xc_plot = xcs[i].x_total
            y_xc_plot = xcs[i].y_total - xcs[i].y_total.min() + sim.z_arr[i + 1]
        else:
            x_xc_plot = xcs[i].x
            y_xc_plot = xcs[i].y - xcs[i].y.min() + sim.z_arr[i + 1]
        z_xc_plot = sim.x_arr[i + 1]
        fd = xcs[i].fd
        L, R = xcs[i].findLR(fd)
        xL = xcs[i].x[L]
        xR = xcs[i].x[R]
        xLs.append(xL)
        xRs.append(xR)
        plot(x_xc_plot, y_xc_plot, z_xc_plot, zdir="y", color="k")
        plot(
            [xL, xR],
            [fd + sim.z_arr[i + 1], fd + sim.z_arr[i + 1]],
            z_xc_plot,
            zdir="y",
            color="blue",
        )

    # Construct water surface polygons
    verts = []
    for i in range(n - 2):
        these_verts = [
            (xLs[i], sim.x_arr[i + 1], xcs[i].fd + sim.z_arr[i + 1]),
            (xLs[i + 1], sim.x_arr[i + 2], xcs[i + 1].fd + sim.z_arr[i + 2]),
            (xRs[i + 1], sim.x_arr[i + 2], xcs[i + 1].fd + sim.z_arr[i + 2]),
            (xRs[i], sim.x_arr[i + 1], xcs[i].fd + sim.z_arr[i + 1]),
            (xLs[i], sim.x_arr[i + 1], xcs[i].fd + sim.z_arr[i + 1]),
        ]
        verts.append(these_verts)
    water_poly = Poly3DCollection(verts, facecolors="blue")
    water_poly.set_alpha(0.35)
    ax.add_collection3d(water_poly)  # , zdir='y')
    xlim([-xmax, xmax])
    ax.view_init(elev=10, azim=-35)
    ax.set_xlabel("Cross-channel distance (m)")
    ax.set_ylabel("Longitudinal distance (m)")
    ax.set_zlabel("Elevation (m)")
    savefig(os.path.join(plotdir, "3D-XC-" + timestep_str + ".png"))
    close(fig)
