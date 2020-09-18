from pylab import figure, plot, xlabel, ylabel, savefig, legend, close, yscale, tight_layout, xlim, ceil, arange

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def make_all_standard_timestep_plots(sim, plotdir, timestep_str):
    plot_overlapping_XCs(sim, plotdir, timestep_str)
    plot_elevation_profile(sim, plotdir, timestep_str)
    plot_concentration_profile(sim, plotdir, timestep_str)
    plot_slope_profile(sim, plotdir, timestep_str)
    plot_3D_XCs(sim, plotdir, timestep_str)
    close('all')



def plot_overlapping_XCs(sim, plotdir, timestep_str):
    figure()
    n = len(sim.xcs)
    if type(sim.xcs[0].x_total) != type(None):
        plot(sim.xcs[0].x_total ,sim.xcs[0].y_total)
    plot(sim.xcs[0].x ,sim.xcs[0].y)
    wl = sim.fd_mids[0]+ sim.xcs[0].y.min()
    plot([-.5,.5], [wl,wl])
    if type(sim.xcs[int(ceil(n/2.))].x_total) != type(None):
        plot(sim.xcs[int(ceil(n/2.))].x_total ,sim.xcs[int(ceil(n/2.))].y_total)
    plot(sim.xcs[int(ceil(n/2.))].x ,sim.xcs[int(ceil(n/2.))].y)
    if type(sim.xcs[-1].x_total) != type(None):
        plot(sim.xcs[-1].x_total ,sim.xcs[-1].y_total)
    plot(sim.xcs[-1].x ,sim.xcs[-1].y)
    savefig(plotdir+'XC-'+timestep_str+'.png')

def plot_elevation_profile(sim, plotdir, timestep_str):
    figure()
    plot(sim.x_arr, sim.h)
    plot(sim.x_arr,sim.z_arr)
    xlabel('Distance (m)')
    ylabel('Elevation (m)')
    legend(['h','z'])
    savefig(plotdir+'Elevation-Profile-'+timestep_str+'.png')

def plot_concentration_profile(sim, plotdir, timestep_str):
    figure()
    plot(sim.x_arr,sim.CO2_w)
    plot(sim.x_arr,sim.CO2_a)
    plot(sim.x_arr,sim.Ca)
    legend(['w','a','Ca'])
    savefig(plotdir+'Concentration-Profile-'+timestep_str+'.png')

def plot_slope_profile(sim, plotdir, timestep_str):
    figure()
    xmid = (sim.x_arr[1:] + sim.x_arr[:-1])/2.
    plot(xmid,sim.slopes)
    plot(xmid, abs(sim.dz))
    yscale('log')
    xlabel('Distance (m)')
    ylabel('Slope/Erosion rate')
    tight_layout()
    savefig(plotdir+'Slope-'+timestep_str+'.png')

def plot_3D_XCs(sim, plotdir, timestep_str):
    #Create 3d XC fig
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    xLs = []
    xRs = []
    xcs = sim.xcs
    n = len(sim.xcs)
    for i in arange(n-1):
        if type(xcs[i].x_total) != type(None):
            x_xc_plot = xcs[i].x_total
            y_xc_plot = xcs[i].y_total- xcs[i].y_total.min()+sim.z_arr[i+1]
        else:
            x_xc_plot = xcs[i].x
            y_xc_plot = xcs[i].y- xcs[i].y.min()+sim.z_arr[i+1]
        z_xc_plot = sim.x_arr[i+1]
        fd = xcs[i].fd
        L,R = xcs[i].findLR(fd)
        xL = xcs[i].x[L]
        xR = xcs[i].x[R]
        xLs.append(xL)
        xRs.append(xR)
        plot(x_xc_plot,y_xc_plot, z_xc_plot, zdir='y', color='k')
        plot([xL,xR], [fd+sim.z_arr[i+1],fd+sim.z_arr[i+1]], z_xc_plot,zdir='y', color='blue')

    #Construct water surface polygons
    verts = []
    for i in range(n-2):
        these_verts = [(xLs[i],sim.x_arr[i+1],xcs[i].fd+sim.z_arr[i+1]),
                        (xLs[i+1],sim.x_arr[i+2],xcs[i+1].fd+sim.z_arr[i+2]),
                        (xRs[i+1],sim.x_arr[i+2],xcs[i+1].fd+sim.z_arr[i+2]),
                        (xRs[i],sim.x_arr[i+1],xcs[i].fd+sim.z_arr[i+1]),
                        (xLs[i],sim.x_arr[i+1],xcs[i].fd+sim.z_arr[i+1])]
        verts.append(these_verts)
    water_poly = Poly3DCollection(verts, facecolors='blue')
    water_poly.set_alpha(0.35)
    ax.add_collection3d(water_poly)#, zdir='y')
    xlim([-5,5])
    ax.view_init(elev=10, azim=-35)
    savefig(plotdir+'3D-XC-'+timestep_str+'.png')
