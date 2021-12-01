"""
runSim is a convenience function for running simulations and producing
plot outputs for a desired set of model parameters. This is mostly
designed to be run from the command line using a yaml that contains the
desired parameters. The name of the yaml file is included as the first
command line argument.

For example use:

python runSim.py low-co2-example.yml

"""

#from pylab import *
import pickle
import sys
import os
import numpy as np
#import debugpy
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from co2_evo.model_parameter_loader import load_params
from co2_evo.standard_timestep_plots import make_all_standard_timestep_plots
from co2_evo.CO2_sim_1D import CO2_1D, sim_1D


def runSim(n=5, L=1000, dz=1, z_arr=None,
            r_init=1, endstep=1000,
            plotdir='./default-figs/',
            snapdir=None,
            start_from_snapshot_num =0,
            dz0_dt = 0.00025,
            snapshot_every=1000,
            plot_every=100,
            T_outside_arr=None,
            CO2_1D_params = {},
            sim_1D_params = None):

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
    sim_1D_params : dict
        Dictionary of keyword arguments to be supplied to sim_1D for
        initialization of simulation object.


    """


    if not os.path.isdir(plotdir):
        os.mkdir(plotdir)

    if sim_1D_params == None:
        CO2_sim = True
    else:
        CO2_sim = False

    if start_from_snapshot_num == 0:
        #Create a new simulation
        x = np.linspace(0,L,n)
        if type(z_arr) == type(None):
            z = np.linspace(1.,1.+dz,n)
        else:
            z_arr = np.array(z_arr)
            z = z_arr
        if len(z) != len(x):
            print("Wrong number of elements in z_arr!")
            return -1
        r = r_init*np.ones(n-1)
        if CO2_sim:
            sim = CO2_1D(x,z, init_radii=r,  **CO2_1D_params)
        else:
            sim = sim_1D(x,z,init_radii=r, **sim_1D_params)
        startstep = 0
    else:
        #Restart from existing snapshot
        start_timestep_str = '%08d' % (start_from_snapshot_num,)
        if type(snapdir)==type(None):
            snapdir=plotdir
        snapshot = open(snapdir+'/snapshot-'+start_timestep_str+'.pkl', 'rb')
        sim = pickle.load(snapshot)
        startstep = start_from_snapshot_num

    #add tag into sim that gives parameter file
    sim.params_file = params_file

    for t in np.arange(startstep, endstep+1):
        print('t=',t, '**********************')
        if CO2_sim:
            sim.run_one_step(T_outside_arr = T_outside_arr)
        else:
            sim.run_one_step()

        sim.z_arr[0] -= dz0_dt * sim.dt_erode
        timestep_str = '%08d' % (t,)
        if t % plot_every == 0:
            print("Plotting timestep: ",t)
            make_all_standard_timestep_plots(sim, plotdir, timestep_str)

        if t % snapshot_every == 0:
            f = open(plotdir+'/snapshot-'+timestep_str+'.pkl', 'wb')
            pickle.dump(sim, f)



if __name__ == '__main__':
    """debugpy.listen(5678)
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
    """
    params_file = sys.argv[1]
    run_params = load_params(params_file)
    print('run_params=',run_params)
    runSim(**run_params)
