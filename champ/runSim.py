"""
runSim is a convenience function for running simulations and producing
plot outputs for a desired set of model parameters. This is mostly
designed to be run from the command line using a yaml that contains the
desired parameters. The name of the yaml file is included as the first
command line argument.

For example use:

python runSim.py example-params.yml

"""

from logging import raiseExceptions
import pickle
import sys
import os
import numpy as np
import time
import copy
import multiprocessing as mp

from champ.utils.model_parameter_loader import load_params
from champ.viz.standard_timestep_plots import (
    make_all_standard_timestep_plots,
    plot_elevation_profile,
    plot_slope_profile,
)
from champ.sim import (
    singleXC,
    multiXC,
    multiXCNormalFlow,
    multiXCGVF,
    multiXCGVF_midXCs,
    spim,
)

params_file = None

# 10% allowed increase in timestep for plots or snapshots.
# This avoids cases with very small dt to hit snapshots or plots.
AllOWED_FRAC_DT_EXTENSION_FOR_OUTPUT = 1.1


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
    snapshot_every=1000,
    plot_every=100,
    snapshot_by_years=True,
    plot_by_years=True,
    n_plot_processes=1,
    run_equiv_spim=False,
    run_width_adjusting_spim=False,
    width_adjustment_exponent=(3/2)*(3/16)**0.9,
    equiv_sim_max_erode=0.05,
    flow_solver="Original",
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
    run_equiv_spim : boolean
        Whether to also run a stream power incision model (SPIM) case that is calibrated
        such that erodibility produces the same equilibrium slope as a multiXC model
        run for the same uplift, discharge, and incision exponent (a).
    run_width_adjusting_spim : boolean
        Whether to run a stream power incision model case that accounts for dynamic width,
        assuming width scales with slope, per Attal et al. (2008).
    width_adjustment_exponent : float
        Value to add to slope exponent, a, in SPIM to account for adjusting width. 
        Default=(3/2)*(3/16)**0.9.
    equiv_sim_max_erode : float
        Maximum erosion setting for equilibration sim used in equivalent SPIM run.
    flow_solver : string
        Solver to use for flow calculations. Options are Original (default), Normal,
        and GVF.
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
            if flow_solver == "Original":
                sim = multiXC(x, z, init_radii=r, **sim_params)
            elif flow_solver == "Normal":
                sim = multiXCNormalFlow(x, z, init_radii=r, **sim_params)
            elif flow_solver == "GVF":
                r = r_init * np.ones(n)
                sim = multiXCGVF(x, z, init_radii=r, **sim_params)
            elif flow_solver == "GVF_midXCs":
                sim = multiXCGVF_midXCs(x, z, init_radii=r, **sim_params)
            else:
                print("Must choose valid flow_solver: Original, Normal, or GVP!")
                raise ValueError("Flow solver value invalid.")
    else:
        # Restart from existing snapshot
        start_timestep_str = "%08d" % (start_from_snapshot_num,)
        if snapdir is None:
            snapdir = plotdir
        snapshot = open(snapdir + "/snapshot-" + start_timestep_str + ".pkl", "rb")
        sim = pickle.load(snapshot)
        snapshot.close()
        # Update simulation parameters (allows changing yml)
        sim.update_params(sim_params)
        # Note: This won't work for switching between layered and non-layered.
        # would need some extra code to do this.

    # add tag into sim that gives parameter file
    if params_file is not None:
        sim.params_file = params_file

    # Run equivalent SPIM simulation, if desired
    if run_equiv_spim and not single_XC_sim:
        # Setup and run equilibration sim
        if np.size(sim.K) > 1:
            K_eqsim = sim.K[-1]  # Use topmost layer
        else:
            K_eqsim = sim.K
        eq_dz = sim.z_arr[-1] - sim.z_arr[0]
        if isinstance(sim.uplift, list):
            uplift = sim.uplift[0]
        else:
            uplift = sim.uplift
        eq_sim = runEquilibrationSim(
            uplift,
            sim.Q_w,
            K_eqsim,
            a=sim.a,
            L=sim.L,
            dz=eq_dz,
            max_frac_erode=equiv_sim_max_erode,
            plotdir=os.path.join(plotdir, "equil-sim/"),
        )
        eq_slope = eq_sim["equil_slope"]
        K_equiv = calcEquivK(uplift, eq_slope, sim.a)
        spim_sim_params = {
            "Q_w": sim.Q_w,
            "a": sim.a,
            "K": K_equiv,
            "uplift": sim.uplift,
            "uplift_times": sim.uplift_times,
        }
        if "layer_elevs" in sim_params:
            spim_sim_params["layer_elevs"] = sim_params["layer_elevs"]
            K_fact = np.array(sim.K) / sim.K[-1]
            # All erodibilities as a multiplying factor of top layer
            spim_K = K_fact * K_equiv
            spim_sim_params["K"] = spim_K

        runSPIM(
            L=sim.L,
            dz=eq_dz,
            endtime=endtime,
            plotdir=os.path.join(plotdir, "spim/"),
            plot_every=plot_every,
            snapshot_every=snapshot_every,
            start_from_snapshot_num=start_from_snapshot_num,
            sim_params=spim_sim_params,
        )

        if run_width_adjusting_spim and not single_XC_sim:
            K_equiv = calcEquivK(uplift, eq_slope, sim.a + width_adjustment_exponent)
            spim_sim_params_adjusted = {
                "Q_w": sim.Q_w,
                "a": sim.a + width_adjustment_exponent,
                "K": K_equiv,
                "uplift": sim.uplift,
                "uplift_times": sim.uplift_times,
            }

            runSPIM(
                L=sim.L,
                dz=eq_dz,
                endtime=endtime,
                plotdir=os.path.join(plotdir, "spim-width-adjusting/"),
                plot_every=plot_every,
                snapshot_every=snapshot_every,
                start_from_snapshot_num=start_from_snapshot_num,
                sim_params=spim_sim_params_adjusted,
            )

    finished = False
    oldtimestep = None
    t_i = time.time()
    while not finished:
        sim.run_one_step()
        print("timestep=", sim.timestep, "   time=", sim.elapsed_time)
        # Reset timestep if we have adjusted for plot or snapshot

        # Check whether we have reached end of simulation
        if sim.elapsed_time >= endtime:
            finished = True
        if endstep is not None:
            if sim.timestep >= endstep:
                finished = True

        # if not single_XC_sim:
        #    sim.z_arr[0] -= dz0_dt * sim.dt_erode

        if oldtimestep is not None:
            sim.dt_erode = oldtimestep
            oldtimestep = None

        # Output plots/snapshots by even timesteps or years
        if not plot_by_years:
            tstep = int(np.round(sim.timestep))
            if tstep % plot_every == 0:
                timestep_str = "%08d" % (tstep,)
                print("Plotting timestep: ", tstep)
                plot_tuple = (copy.deepcopy(sim), plotdir, timestep_str)
                plot_queue.put(plot_tuple)
        else:
            t_int = int(np.round(sim.elapsed_time))
            time_to_next_plot = plot_every - (sim.elapsed_time % plot_every)
            # This logic is pretty convoluted, but it seems to correctly
            # handle cases where the timestep is less than 1 and therefore
            # simply rounding to the nearest int is ineffective.
            if (
                (t_int % plot_every == 0)
                and ((sim.elapsed_time - sim.dt_erode) % plot_every > 1)
                and (
                    (time_to_next_plot < sim.dt_erode)
                    or (plot_every - time_to_next_plot < sim.dt_erode)
                )
            ):
                time_str = "%08d" % (t_int,)
                print("Plotting at time: ", t_int)
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
            t_int = int(np.round(sim.elapsed_time))
            time_to_next_snap = snapshot_every - (sim.elapsed_time % snapshot_every)
            # This logic is pretty convoluted, but it seems to correctly
            # handle cases where the timestep is less than 1 and therefore
            # simply rounding to the nearest int is ineffective.
            if (
                (t_int % snapshot_every == 0)
                and ((sim.elapsed_time - sim.dt_erode) % snapshot_every > 1)
                and (
                    (time_to_next_snap < sim.dt_erode)
                    or (snapshot_every - time_to_next_snap < sim.dt_erode)
                )
            ):
                time_str = "%08d" % (t_int,)
                print("Snapshot at timestep: ", t_int)
                f = open(plotdir + "/snapshot-" + time_str + ".pkl", "wb")
                pickle.dump(sim, f)

        # Timestep adjustments for time-based plots and snapshots
        if plot_by_years:
            # Check whether we need to adjust timestep to hit next plot
            time_to_next_plot = plot_every - (sim.elapsed_time % plot_every)
            if AllOWED_FRAC_DT_EXTENSION_FOR_OUTPUT * sim.dt_erode > time_to_next_plot:
                oldtimestep = sim.dt_erode
                sim.dt_erode = time_to_next_plot

        if snapshot_by_years:
            # Check whether we need to adjust timestep to hit next snapshot
            time_to_next_snap = snapshot_every - (sim.elapsed_time % snapshot_every)
            if AllOWED_FRAC_DT_EXTENSION_FOR_OUTPUT * sim.dt_erode > time_to_next_snap:
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


def runEquilibrationSim(
    uplift,
    Q_w,
    K,
    a,
    mean_tol=0.005,
    diff_tol=0.01,
    n=5,
    L=1000,
    dz=20,
    r_init=10.0,
    xc_n=300,
    mintime=1000,
    plotdir="./equil-sim/",
    adaptive_step=True,
    max_frac_erode=0.05,  # Set higher to speed equilibration
):

    """Run multiXC simulation to topographic equilibrium.

    Parameters
    ----------
    uplift : float
        Rate of topographic uplift (m/year)
    Q_w : float
        Channel discharge (m^3/s)
    K : float
        Rock erodibility
    a : float
        Exponent in shear stress erosion rule. a=(3/2)*n,
        where n is the slope exponent in the stream power incision model.
    mean_tol : float
        Fractional tolerance to define convergence for difference between
        uplift and average erosion across the model channel (topographic equilibrium).
        Calculated as: abs(1.0 - abs(avg_erosion / uplift)). Convergence
        occurs when both this criteria and diff_tol are satisfied.
    diff_tol : float
        Fractional tolerance to define convergence based on the difference
        between minimum and maximum XC erosion rates. Calculated as:
        (Max erosion - Min Erosion) / avg_erosion. Convergence
        occurs when both this criteria and mean_tol are satisfied.
    n : int
        Number of nodes.
    L : float
        Length of entire channel (meters).
    dz : float
        Initial change in elevation over channel length (meters).
    r_init : float
        Initial cross-section radius.
    mintime : int
        Minimum time (years) for which to run the simulation. Default=1000.
    plotdir : string
        Path to directory that will hold plots and snapshots. This directory
        will be created if it does not exist.
    adaptive_step : boolean, optional
        Whether or not to adjust timestep dynamically. For equilibration simulations
        the default is True.
    max_frac_erode : float, optional
        Maximum fraction of radial distance to erode within a single timestep
        under adaptive time-stepping. If erosion exceeds this fraction, then
        the timestep will be reduced. If erosion is much less than this fraction,
        then the timestep will be increased. We have not conducted a detailed
        stability analysis. For equilibration simulations, we have chosen a
        relatively large default value of 0.05, which reduces time to convergence.
        If simulations become unstable, reduce this fraction.
    """
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    sim_params = {
        "Q_w": Q_w,
        "xc_n": xc_n,
        "adaptive_step": adaptive_step,
        "max_frac_erode": max_frac_erode,
        "a": a,
        "K": K,
        "uplift": uplift,
    }
    eq_sim = runSim(
        n=n,
        L=L,
        dz=dz,
        r_init=r_init,
        plotdir=plotdir,
        endtime=mintime,
        snapshot_every=10000,
        plot_every=10000,
        sim_params=sim_params,
    )
    converged = False
    nsteps = 0
    boost_step = 0
    stop_boost = False
    erosion_greater_than_uplift_steps = 0
    while not converged:
        eq_sim.run_one_step()
        # eq_sim.z_arr[0] -= uplift * eq_sim.dt_erode
        avg_erosion = eq_sim.dz.mean() / eq_sim.dt_erode
        if np.abs(1.0 - np.abs(avg_erosion / uplift)) < mean_tol:
            # Met mean erosion rate tolerance for equilibrium
            if (
                np.abs(
                    (eq_sim.dz.max() - eq_sim.dz.min()) / eq_sim.dt_erode / avg_erosion
                )
                < diff_tol
            ):
                # Met difference among nodes tolerance for equilibrium
                converged = True
        nsteps += 1
        if nsteps % 100 == 0:
            print("nsteps=", nsteps)
            print("Mean erosion =", avg_erosion)
            print("mean_tol=", np.abs(1.0 - np.abs(avg_erosion / uplift)))
            print(
                "diff_tol=",
                np.abs(
                    (eq_sim.dz.max() - eq_sim.dz.min()) / eq_sim.dt_erode / avg_erosion
                ),
            )
        if (
            (-1 * avg_erosion < uplift)
            and (nsteps > boost_step + 50)
            and not stop_boost
        ):
            print("Avg erosion =", avg_erosion)
            print("Uplift = ", uplift)
            boost_step = nsteps
            print("*****Boost slope to speed equilibration*****")
            dz = 2 * dz
            z_boost = np.linspace(0, dz, n)
            eq_sim.z_arr += z_boost

        # Avoid boosting after we have already had erosion rates greater than uplift
        if -1 * avg_erosion > uplift:
            erosion_greater_than_uplift_steps += 1
        if erosion_greater_than_uplift_steps > 5:
            stop_boost = True

    # Pickle and plot converged equilibrium state
    tstep = int(np.round(eq_sim.timestep))
    timestep_str = "%08d" % (tstep,)
    print("Snapshot of equilibrium state at timestep: ", tstep)
    snapfilename = plotdir + "/snapshot-" + timestep_str + ".pkl"
    f = open(snapfilename, "wb")
    pickle.dump(eq_sim, f)
    make_all_standard_timestep_plots(eq_sim, plotdir, timestep_str)
    # Calculate equilibrium morphology
    equil_slope = eq_sim.slopes.mean()
    equil_width = eq_sim.W.mean()
    return {
        "equil_slope": equil_slope,
        "equil_width": equil_width,
        "equilsnapshot": snapfilename,
        "sim": eq_sim,
    }


def calcEquivK(uplift, slope, a):
    """
    Calculate equivalent erodibility to produce equilibrium slope.

    Parameters
    ----------
    uplift : float
        Uplift rate in m/year.
    slope : float
        Equilibrium channel slope at provided uplift.
    a : float
        Shear stress erosion exponent.
    """

    # Calculate slope exponent from a
    n = a * (2.0 / 3.0)
    K_equiv = uplift * slope ** (-n)
    return K_equiv


def runSPIM(
    n=1000,
    L=1000,
    dz=1,
    z_arr=None,
    endtime=1000,
    plotdir="./spim/",
    plot_every=100,
    snapshot_every=1000,
    start_from_snapshot_num=0,
    snapshot_by_years=True,
    plot_by_years=True,
    CFL_crit=0.9,
    sim_params={},
):

    """Run Stream power incision model (SPIM) simulation using specified parameters.

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
    endtime : int
        Time (years) at which to end simulation. Default=1000.
    plotdir : string
        Path to directory that will hold plots and outputs. This directory
        will be created if it does not exist. If starting from previous snapshot,
        snapshot must be in this directory.
    start_from_snapshot_num : int
        If set to a nonzero value, then the simulation will be
        started from that snapshot number within plotdir. If set
        to zero, then a new simulation is started.
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
    sim_params : dict
        Dictionary of keyword arguments to be supplied to spim for
        initialization of simulation object.
    CFL_crit : float, optional
        Timestep is adjusted to produce this Courant-Friedrich-Lax number.
        Default is 0.9.
    """
    print("Running SPIM simulation...")
    print("plotdir is", plotdir)
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    if start_from_snapshot_num == 0:
        x = np.linspace(0, L, n)
        if z_arr is None:
            z = np.linspace(1.0, 1.0 + dz, n)
        else:
            z_arr = np.array(z_arr)
            z = z_arr
        if len(z) != len(x):
            print("Wrong number of elements in z_arr!")
            return -1
        sim = spim(x, z, **sim_params)
    else:
        # Restart from existing snapshot
        start_timestep_str = "%08d" % (start_from_snapshot_num,)
        snapshot = open(plotdir + "/snapshot-" + start_timestep_str + ".pkl", "rb")
        sim = pickle.load(snapshot)
        snapshot.close()
        # Update simulation parameters (allows changing yml)
        sim.update_params(sim_params)

    # add tag into sim that gives parameter file
    if params_file is not None:
        sim.params_file = params_file

    finished = False
    oldtimestep = None
    t_i = time.time()
    while not finished:
        # Determine stable timestep
        Celerity = sim.K_arr[1:] * sim.slopes ** (sim.n - 1)
        # Set timestep for stable CFL criteria
        # Calculate best timestep, unless we are wanting to hit year for plotting
        if oldtimestep is None:
            sim.dt_erode = CFL_crit * sim.dx / (np.abs(Celerity).max())
        else:
            # Reset timestep next time if we have adjusted for plot or snapshot
            oldtimestep = None
        if plot_by_years:
            if sim.dt_erode > plot_every:
                sim.dt_erode = plot_every
        if snapshot_by_years:
            if sim.dt_erode > snapshot_every:
                sim.dt_erode = snapshot_every
        # Run erosion
        sim.run_one_step()
        print(
            "timestep=",
            sim.timestep,
            "   time=",
            sim.elapsed_time,
            " dt=",
            sim.dt_erode,
            "dt_old=",
            sim.old_dt,
            " mean eros=",
            np.mean(sim.dz / sim.old_dt),
        )

        # Check whether we have reached end of simulation
        if sim.elapsed_time >= endtime:
            finished = True

        # Output plots/snapshots by even timesteps or years
        if not plot_by_years:
            tstep = int(np.round(sim.timestep))
            if tstep % plot_every == 0:
                timestep_str = "%08d" % (tstep,)
                print("Plotting timestep: ", tstep)
                plot_elevation_profile(sim, plotdir, timestep_str, with_h=False)
                plot_slope_profile(sim, plotdir, timestep_str)
        else:
            t = int(np.round(sim.elapsed_time))
            if t % plot_every == 0:
                time_str = "%08d" % (t,)
                print("Plotting at time: ", t)
                plot_elevation_profile(sim, plotdir, time_str, with_h=False)
                plot_slope_profile(sim, plotdir, time_str)
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

    t_f = time.time()
    print(f"Runtime for simulation was {t_f - t_i}")
    return sim


def main():
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


if __name__ == "__main__":
    main()
