"""
runSim is a convenience function for running simulations and producing
plot outputs for a desired set of model parameters. This is mostly
designed to be run from the command line using a yaml that contains the
desired parameters. The name of the yaml file is included as the first
command line argument.

For example use:

python runSim.py example-params.yml

"""

import pickle
import sys
import os
import numpy as np
import time
import copy
import multiprocessing as mp

from champ.utils.model_parameter_loader import load_params
from champ.viz.standard_timestep_plots import make_all_standard_timestep_plots
from champ.sim import singleXC, multiXC

params_file = None


def runSim(
    n=5,
    L=1000,
    dz=1,
    z_arr=None,
    r_init=1,
    endtime=1000,
    endstep=None,
    snapdir=None,
    plotdir="./default-figs/",
    start_from_snapshot_num=0,
    dz0_dt=0.00025,
    snapshot_every=1000,
    plot_every=100,
    snapshot_by_years=True,
    plot_by_years=True,
    n_plot_processes=1,
    sim_params={},
):

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
    endtime : int
        Time (years) at which to end simulation. Default=1000.
    endstep : int
        Timestep on which to end simulation. Default=None.
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
        Number of years/timesteps after which to record a pickled CO2_1D object.
        These snapshots can easily be used to restart a simulation from a
        previous point.
    plot_every : int
        Number of years/timesteps after which to create plots of simulation
        outputs.
    snapshot_by_years : boolean
        Whether shapshots should be taken after a certain number of years of simulation
        (True) or after a certain number of timesteps (False). Default is True.
    plot_by_years : boolean
        Whether plots should be created after a certain number of years of simulation
        (True) or after a certain number of timesteps (False). Default is True.
    n_plot_processes : int
        Number of multiprocessing processes to use for creating plots, which are run
        within a separate process from the simulation. Increase this number if plotting
        is slowing down your simulation and additional CPUs are available. Default is 1.
    sim_params : dict
        Dictionary of keyword arguments to be supplied to singleXC or multiXC for
        initialization of simulation object.

    """

    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    plot_queue = mp.JoinableQueue()
    plot_process_list = []
    for i in range(n_plot_processes):
        plot_process = mp.Process(target=make_plots, args=(plot_queue,), daemon=True)
        plot_process.start()
        plot_process_list.append(plot_process)

    if n == 1:
        single_XC_sim = True
    else:
        single_XC_sim = False

    if start_from_snapshot_num == 0:
        # Create a new simulation
        if single_XC_sim:
            sim = singleXC(init_radius=r_init, **sim_params)
        else:
            x = np.linspace(0, L, n)
            if z_arr is None:
                z = np.linspace(1.0, 1.0 + dz, n)
            else:
                z_arr = np.array(z_arr)
                z = z_arr
            if len(z) != len(x):
                print("Wrong number of elements in z_arr!")
                return -1
            r = r_init * np.ones(n - 1)
            sim = multiXC(x, z, init_radii=r, **sim_params)
    else:
        # Restart from existing snapshot
        start_timestep_str = "%08d" % (start_from_snapshot_num,)
        if snapdir is None:
            snapdir = plotdir
        snapshot = open(snapdir + "/snapshot-" + start_timestep_str + ".pkl", "rb")
        sim = pickle.load(snapshot)
        # Update simulation parameters (allows changing yml)
        sim.update_params(sim_params)
        # Note: This won't work for switching between layered and non-layered.
        # would need some extra code to do this.

    # add tag into sim that gives parameter file
    if params_file is not None:
        sim.params_file = params_file

    finished = False
    oldtimestep = None
    t_i = time.time()
    while not finished:
        sim.run_one_step()
        print("timestep=", sim.timestep, "   time=", sim.elapsed_time)
        # Reset timestep if we have adjusted for plot or snapshot
        if oldtimestep is not None:
            sim.dt_erode = oldtimestep
            oldtimestep = None

        # Check whether we have reached end of simulation
        if sim.elapsed_time >= endtime:
            finished = True
        if endstep is not None:
            if sim.timestep >= endstep:
                finished = True

        if not single_XC_sim:
            sim.z_arr[0] -= dz0_dt * sim.dt_erode

        # Output plots/snapshots by even timesteps or years
        if not plot_by_years:
            tstep = int(np.round(sim.timestep))
            if tstep % plot_every == 0:
                timestep_str = "%08d" % (tstep,)
                print("Plotting timestep: ", tstep)
                plot_tuple = (copy.deepcopy(sim), plotdir, timestep_str)
                plot_queue.put(plot_tuple)
        else:
            t = int(np.round(sim.elapsed_time))
            if t % plot_every == 0:
                time_str = "%08d" % (t,)
                print("Plotting at time: ", t)
                plot_tuple = (copy.deepcopy(sim), plotdir, time_str)
                plot_queue.put(plot_tuple)
        if not snapshot_by_years:
            tstep = int(np.round(sim.timestep))
            if tstep % snapshot_every == 0:
                timestep_str = "%08d" % (tstep,)
                print("Snapshot at timestep: ", tstep)
                f = open(plotdir + "/snapshot-" + timestep_str + ".pkl", "wb")
                pickle.dump(sim, f)
        else:
            t = int(np.round(sim.elapsed_time))
            if t % snapshot_every == 0:
                time_str = "%08d" % (t,)
                print("Snapshot at timestep: ", t)
                f = open(plotdir + "/snapshot-" + time_str + ".pkl", "wb")
                pickle.dump(sim, f)

        # Timestep adjustments for time-based plots and snapshots
        if plot_by_years:
            # Check whether we need to adjust timestep to hit next plot
            time_to_next_plot = plot_every - (sim.elapsed_time % plot_every)
            if sim.dt_erode > time_to_next_plot:
                oldtimestep = sim.dt_erode
                sim.dt_erode = time_to_next_plot

        if snapshot_by_years:
            # Check whether we need to adjust timestep to hit next snapshot
            time_to_next_snap = snapshot_every - (sim.elapsed_time % snapshot_every)
            if sim.dt_erode > time_to_next_snap:
                if plot_by_years:
                    if time_to_next_snap > time_to_next_plot:
                        # We will hit the snapshot later and get plot now
                        pass
                    else:
                        oldtimestep = sim.dt_erode
                        sim.dt_erode = time_to_next_snap
                else:
                    oldtimestep = sim.dt_erode
                    sim.dt_erode = time_to_next_snap

    # Make sure all plotting creation finishes up
    plot_queue.join()

    t_f = time.time()
    print(f"Runtime for simulation was {t_f - t_i}")
    return sim


def make_plots(plot_queue):
    while True:
        this_plot_tuple = plot_queue.get()
        make_all_standard_timestep_plots(*this_plot_tuple)
        plot_queue.task_done()


if __name__ == "__main__":
    """debugpy.listen(5678)
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
    """
    start_time = time.time()
    params_file = sys.argv[1]
    run_params = load_params(params_file)
    print("run_params=", run_params)
    runSim(**run_params)
    end_time = time.time()
    print(f"Simulation took {end_time-start_time:.2f} seconds to run.")
