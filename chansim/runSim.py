"""
runSim is a convenience function for running simulations and producing
plot outputs for a desired set of model parameters. This is mostly
designed to be run from the command line using a yaml that contains the
desired parameters. The name of the yaml file is included as the first
command line argument.

For example use:

python runSim.py example-params.yml

"""

#from pylab import *
import pickle
import sys
import os
import numpy as np

from chansim.utils.model_parameter_loader import load_params
from chansim.viz.standard_timestep_plots import make_all_standard_timestep_plots
from chansim.chansim import singleXC, multiXC


def runSim(n=5, L=1000, dz=1, z_arr=None,
            r_init=1, endstep=1000,
            plotdir='./default-figs/',
            snapdir=None,
            start_from_snapshot_num =0,
            dz0_dt = 0.00025,
            snapshot_every=1000,
            plot_every=100,
            sim_params = None):

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
    sim_params : dict
        Dictionary of keyword arguments to be supplied to singleXC or multiXC for
        initialization of simulation object.


    """


    if not os.path.isdir(plotdir):
        os.mkdir(plotdir)

    if n==1:
        single_XC_sim = True
    else:
        single_XC_sim = False

    if start_from_snapshot_num == 0:
        #Create a new simulation
        if single_XC_sim:
            singleXC(init_radius=r_init, **sim_params)
        else:
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
            sim = multiXC(x,z,init_radii=r, **sim_params)
            startstep = 0
    else:
        #Restart from existing snapshot
        start_timestep_str = '%08d' % (start_from_snapshot_num,)
        if type(snapdir)==type(None):
            snapdir=plotdir
        snapshot = open(snapdir+'/snapshot-'+start_timestep_str+'.pkl', 'rb')
        sim = pickle.load(snapshot)
        startstep = start_from_snapshot_num
        #Update simulation parameters (allows changing yml)
        sim.update_params(sim_params)
        ### Note: This won't work for switching between layered and non-layered.
        ###   would need some extra code to do this.

    #add tag into sim that gives parameter file
    sim.params_file = params_file

    for t in np.arange(startstep, endstep+1):
        print('t=',t, '**********************')
        sim.run_one_step()

    if not single_XC_sim:
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
