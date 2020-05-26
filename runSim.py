"""
runSim is a convenience function for running simulations and producing
plot outputs for a desired set of model parameters. This is mostly
designed to be run from the command line using a yaml that contains the
desired parameters. The name of the yaml file is included as the first
command line argument.

For example use:

python runSim.py low-co2-example.yml

"""

from pylab import *
import pickle
import sys
import os

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from model_parameter_loader import load_params

from CO2_sim_1D import CO2_1D


def runSim(n=5, L=1000, dz=1, z_arr=None,
            r_init=1, endstep=1000,
            plotdir='./default-figs/',
            start_from_snapshot_num =0,
            dz0_dt = 0.00025,
            snapshot_every=1000,
            plot_every=100,
            T_outside_arr=None,
            CO2_1D_params = {}):

    """Run simulation using specified parameters.

    Parameters
    ----------
    n : int
        Number of nodes.
    L : float
        Length of entire channel (meters).
    dz : float
        Change in elevation over channel length (meters).
    z_arr : ndarray
        Array of node elevations. If given, then dz is ignored. Default=None.
    r_init : float
        Initial cross-section radius.
    endstep : int
        Timestep on which to end simulation.
    plotdir : string
        Path to directory that will hold plots and snapshots. This directory
        will be created if it does not exist.
    start_from_snapshot_num : int
        If set to a nonzero value, then the simulation will be
        started from that snapshot number within plotdir. If set
        to zero, then a new simulation is started.
    dz0_dt : float
        Rate of change of baselevel. This distance is subtracted
        from the elevation of the downstream boundary node during
        each timestep.
    snapshot_every : int
        Number of time steps after which to record a pickled CO2_1D object.
        These snapshots can easily be used to restart a simulation from a
        previous point.
    plot_every : int
        Number of time steps after which to create plots of simulation
        outputs.
    T_outside_arr : ndarray
        An array of outside air temperatures to cycle through during
        each timestep. If set to None (default) then only the single
        T_outside value supplied in CO2_1D_params will be used.
    CO2_1D_params : dict
        Dictionary of keyword arguments to be supplied to CO2_1D for
        initialization of simulation object.


    """


    if not os.path.isdir(plotdir):
        os.mkdir(plotdir)

    if start_from_snapshot_num == 0:
        #Create a new simulation
        x = linspace(0,L,n)
        if type(z_arr) == type(None):
            z = linspace(1.,1.+dz,n)
        else:
            z_arr = np.array(z_arr)
            z = z_arr
        if len(z) != len(x):
            print("Wrong number of elements in z_arr!")
            return -1
        r = r_init*ones(n-1)
        sim = CO2_1D(x,z, init_radii=r,**CO2_1D_params)
        startstep = 0
    else:
        #Restart from existing snapshot
        start_timestep_str = '%08d' % (start_from_snapshot_num,)
        snapshot = open(plotdir+'/snapshot-'+start_timestep_str+'.pkl', 'rb')
        sim = pickle.load(snapshot)
        startstep = start_from_snapshot_num

    #add tag into sim that gives parameter file
    sim.params_file = params_file

    for t in arange(startstep, endstep+1):
        print('t=',t, '**********************')
        sim.run_one_step(T_outside_arr = T_outside_arr)
        sim.z_arr[0] -= dz0_dt
        if t % plot_every == 0:
            timestep_str = '%08d' % (t,)
            print("Plotting timestep: ",t)
            figure()
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

            figure()
            plot(sim.x_arr, sim.h)
            plot(sim.x_arr,sim.z_arr)
            xlabel('Distance (m)')
            ylabel('Elevation (m)')
            legend(['h','z'])
            savefig(plotdir+'Elevation-Profile-'+timestep_str+'.png')

            figure()
            plot(sim.x_arr,sim.CO2_w)
            plot(sim.x_arr,sim.CO2_a)
            plot(sim.x_arr,sim.Ca)
            legend(['w','a','Ca'])
            savefig(plotdir+'Concentration-Profile-'+timestep_str+'.png')

            figure()
            xmid = (sim.x_arr[1:] + sim.x_arr[:-1])/2.
            plot(xmid,sim.slopes)
            plot(xmid, abs(sim.dz))
            yscale('log')
            xlabel('Distance (m)')
            ylabel('Slope/Erosion rate')
            tight_layout()
            savefig(plotdir+'Slope-'+timestep_str+'.png')
            #Create 3d XC fig
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            xLs = []
            xRs = []
            xcs = sim.xcs
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
            close('all')
            if t % snapshot_every == 0:
                f = open(plotdir+'/snapshot-'+timestep_str+'.pkl', 'wb')
                pickle.dump(sim, f)



if __name__ == '__main__':
    params_file = sys.argv[1]
    run_params = load_params(params_file)
    print('run_params=',run_params)
    runSim(**run_params)
