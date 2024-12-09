import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import glob
import os

def make_all_profile_plots(snapdir):
    snapshots = glob.glob(os.path.join(snapdir, 'snapshot*.pkl'))
    plotdir = os.path.join(snapdir, 'profile_plots/')
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    matplotlib.use("Agg")
    for snap in snapshots:
        with open(snap, 'rb') as f:
            sim = pickle.load(f)
        plot_profiles(sim, plotdir=plotdir)
        plt.close('all')


def calc_W_eff_for_XCs(sim):
    n_XCs = len(sim.xcs)
    nQ = sim.nQ
    mean_erosion = np.zeros([nQ, n_XCs])
    mean_diss = np.zeros([nQ, n_XCs])
    widths = np.zeros([nQ, n_XCs])
    for i, Q in enumerate(sim.Q_arr):
        sim.Q_w = Q
        sim.calc_flow(use_old_fd=False, create_interp=False)
        widths[i,:] = sim.W
        dt_frac = sim.pdf_Q_frac[i]
        for j, xc in enumerate(sim.xcs):
            if not sim.layered_sim:
                xc.erode_power_law(
                                a=sim.a,
                                K=sim.K,
                                T_c=sim.T_c,
                                dt=sim.dt_erode * dt_frac,
                                resample=False,
                                trim=False,
                                no_erode=True,
                            )
            else:
                if len(sim.init_z) == len(sim.xcs):
                        absolute_layer_elevs = sim.layer_elevs - sim.init_z[j]
                else:
                    absolute_layer_elevs = sim.layer_elevs - sim.init_z[j + 1]
                xc.erode_power_law_layered(
                    a=sim.a,
                    K=sim.K,
                    T_c=sim.T_c,
                    layer_elevs=absolute_layer_elevs,
                    dt=sim.dt_erode * dt_frac,
                    resample=False, 
                    trim=False,
                    no_erode=True,
                )
                xc.dr_mech = xc.dr
                if sim.layer_solubility is not None:
                        dr_tmp = np.zeros(xc.n)
                        dr_tmp[xc.wetidx] += xc.dr_mech
                        
                        if len(sim.layer_solubility) == len(sim.K):
                            K_sol_list = np.zeros(len(sim.K))
                            K_sol_list[sim.layer_solubility] = sim.K_sol
                            xc.erode_power_law_layered(
                                a=sim.a_sol,
                                K=K_sol_list,
                                layer_elevs=absolute_layer_elevs,
                                dt=sim.dt_erode * dt_frac,
                                trim=False,
                                resample=False,
                                no_erode=True,
                            )
                            xc.dr_diss = xc.dr
                            dr_tmp[xc.wetidx] += xc.dr_diss
                            xc.dr = dr_tmp[xc.wetidx]
                        else:
                            print(
                                "Number of layer solubility entries must equal number of layers."
                            )
                            raise IndexError
            mean_erosion[i,j] = xc.dr_mech.mean()/sim.dt_erode
            mean_diss[i,j] = xc.dr_diss.mean()/sim.dt_erode
    mean_total = mean_diss + mean_erosion
    Q_max_idx = mean_total.argmax(axis=0)
    W_eff = np.zeros(n_XCs)
    for i in np.arange(n_XCs):
        W_eff[i] = widths[Q_max_idx[i],i]
    return W_eff


def plot_profiles(sim, sim2=None, plotdir=None):
    t_int = int(np.round(sim.elapsed_time))
    time_str = "%08d" % (t_int,)
    xmid = (sim.x_arr[1:] + sim.x_arr[:-1]) / 2.0
    sol_xs = sim.x_arr[np.logical_and(sim.z_arr<sim.layer_elevs[1], sim.z_arr>sim.layer_elevs[0])]
    if len(sol_xs)>1:
        sol_xs_sim = (sol_xs[1:] + sol_xs[:-1])/2.
        sol_x_min = sol_xs_sim.min()
        sol_x_max = sol_xs_sim.max()

    if sim2 is not None:
        sol_xs_2 = sim2.x_arr[np.logical_and(sim2.z_arr<sim2.layer_elevs[1], sim2.z_arr>sim2.layer_elevs[0])]
        if len(sol_xs_2)>1:
            sol_xs_sim_2 = (sol_xs_2[1:] + sol_xs_2[:-1])/2.
            sol_x_min_2 = sol_xs_sim_2.min()
            sol_x_max_2 = sol_xs_sim_2.max()

    plt.figure()
    W_eff = calc_W_eff_for_XCs(sim)
    plt.plot(xmid, W_eff, 'k')
    if sim2 is not None:
        W_eff2 = calc_W_eff_for_XCs(sim2)
        plt.plot(xmid, W_eff2, 'k--')
    plt.xlabel('Distance (m)')
    plt.ylabel('Channel width (m)')
    if len(sol_xs)>1:
        plt.axvspan(sol_x_min, sol_x_max, color='red', alpha=0.3)
    if sim2 is not None:
        if len(sol_xs)>1:
            plt.axvspan(sol_x_min_2, sol_x_max_2, color='blue', alpha=0.3)
    plt.tight_layout()
    if plotdir is not None:
        figfile = os.path.join(plotdir, 'width-'+time_str+'.png')
        plt.savefig(figfile)


    plt.figure()
    plt.plot(xmid, sim.slopes, 'k')
    if sim2 is not None:
        plt.plot(xmid, sim2.slopes, 'k--')
    plt.xlabel('Distance (m)', fontsize=18)
    plt.ylabel('Channel slope', fontsize=18)
    if len(sol_xs)>1:
        plt.axvspan(sol_x_min, sol_x_max, color='red', alpha=0.3)
    if sim2 is not None:
        if len(sol_xs)>1:
            plt.axvspan(sol_x_min_2, sol_x_max_2, color='blue', alpha=0.3)
    plt.tight_layout()
    if plotdir is not None:
        figfile = os.path.join(plotdir, 'slope-'+time_str+'.png')
        plt.savefig(figfile)


    plt.figure()
    plt.plot(xmid, -sim.dz/sim.dt_erode, 'k')
    if sim2 is not None:
        plt.plot(xmid, -sim2.dz/sim2.dt_erode, 'k--')
    plt.xlabel('Distance (m)', fontsize=18)
    plt.ylabel('Erosion rate (m/yr)', fontsize=18)
    if len(sol_xs)>1:
        plt.axvspan(sol_x_min, sol_x_max, color='red', alpha=0.3)
    if sim2 is not None:
        if len(sol_xs)>1:
            plt.axvspan(sol_x_min_2, sol_x_max_2, color='blue', alpha=0.3)
    plt.tight_layout()
    if plotdir is not None:
        figfile = os.path.join(plotdir, 'erosion-'+time_str+'.png')
        plt.savefig(figfile)


def plot_erosion_for_Qs(sim, xc_idx=2, single=False, plotdir=None):
    t_int = int(np.round(sim.elapsed_time))
    time_str = "%08d" % (t_int,)
    XC_str = "%03d" % (xc_idx,)
    Q_strs = []
    mean_erosion = []
    mean_diss = []
    cmap = plt.get_cmap('rainbow')
    logQ_arr = np.log10(sim.Q_arr)
    norm = plt.Normalize(logQ_arr[0], logQ_arr[-1])
    if single:
        xc = sim.xc
    else:
        xc = sim.xcs[xc_idx]
    erosion_lines = []
    for i, Q in enumerate(sim.Q_arr):
        sim.Q_w = Q
        if single:
            sim.calc_flow()
        else:
            sim.calc_flow(use_old_fd=False, create_interp=False)
        #sim.erode(dt_frac=sim.pdf_Q_frac[i])
        dt_frac = sim.pdf_Q_frac[i]
        if not sim.layered_sim:
            xc.erode_power_law(
                            a=sim.a,
                            K=sim.K,
                            T_c=sim.T_c,
                            dt=sim.dt_erode * dt_frac,
                            resample=False,
                            trim=False,
                            no_erode=True,
                        )
        else:
            if single:
                absolute_layer_elevs = sim.layer_elevs
            else:
                if len(sim.init_z) == len(sim.xcs):
                        absolute_layer_elevs = sim.layer_elevs - sim.init_z[xc_idx]
                else:
                    absolute_layer_elevs = sim.layer_elevs - sim.init_z[xc_idx + 1]
            xc.erode_power_law_layered(
                a=sim.a,
                K=sim.K,
                T_c=sim.T_c,
                layer_elevs=absolute_layer_elevs,
                dt=sim.dt_erode * dt_frac,
                resample=False, 
                trim=False,
                no_erode=True,
            )
            xc.dr_mech = xc.dr
            if sim.layer_solubility is not None:
                    dr_tmp = np.zeros(xc.n)
                    dr_tmp[xc.wetidx] += xc.dr_mech
                    
                    if len(sim.layer_solubility) == len(sim.K):
                        K_sol_list = np.zeros(len(sim.K))
                        K_sol_list[sim.layer_solubility] = sim.K_sol
                        xc.erode_power_law_layered(
                            a=sim.a_sol,
                            K=K_sol_list,
                            layer_elevs=absolute_layer_elevs,
                            dt=sim.dt_erode * dt_frac,
                            trim=False,
                            resample=False,
                            no_erode=True,
                        )
                        xc.dr_diss = xc.dr
                        dr_tmp[xc.wetidx] += xc.dr_diss
                        xc.dr = dr_tmp[xc.wetidx]
                    else:
                        print(
                            "Number of layer solubility entries must equal number of layers."
                        )
                        raise IndexError
        #print("num wetidx = ", len(xc.dr) )
        #print("Energy slope =", xc.eSlope)
        if sim.layer_solubility is not None:
            lines = plt.plot(xc.x[xc.wetidx], xc.dr_mech/sim.dt_erode, c=cmap(norm(logQ_arr[i])))
            plt.plot(xc.x[xc.wetidx], xc.dr_diss/sim.dt_erode, '--', c=cmap(norm(logQ_arr[i])))            
            mean_erosion.append(xc.dr_mech.mean()/sim.dt_erode)
            mean_diss.append(xc.dr_diss.mean()/sim.dt_erode)
        else:     
            lines = plt.plot(xc.x[xc.wetidx], xc.dr/sim.dt_erode, c=cmap(norm(logQ_arr[i])))
            mean_erosion.append(xc.dr.mean()/sim.dt_erode)
        erosion_lines += lines
        Q_strs.append('Q = '+str(Q)[:5])
        
    plt.legend(erosion_lines, Q_strs, loc='upper right', bbox_to_anchor=(1.35,1.1))
    plt.xlabel('Cross-channel distance (m)', fontsize=16)
    plt.ylabel('Effective Erosion rate (m/yr)', fontsize=16)
    plt.subplots_adjust(right=0.85)
    plt.tight_layout()
    if plotdir is not None:
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        figfile = os.path.join(plotdir, 'XC-Erosion-by-Qs-'+time_str+'-XC'+XC_str+'.png')
        plt.savefig(figfile)


    plt.figure()
    plt.semilogx(sim.Q_arr, sim.pdf_Q_frac, 'o-', label='Discharge (left)')
    plt.xlabel(r'Discharge (m$^3$/s)', fontsize=18)
    plt.ylabel('Probability density', fontsize=18)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.semilogx(sim.Q_arr, mean_erosion, 's--k', label='Mechanical Erosion')
    if sim.layer_solubility is not None:
         ax2.semilogx(sim.Q_arr, mean_diss, '*:k', label='Dissolution')
    plt.ylabel('Effective Erosion rate (m/yr)', fontsize=16)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.tight_layout()
    if plotdir is not None:
        figfile = os.path.join(plotdir, 'PDF-Erosion-'+time_str+'-XC'+XC_str+'.png')
        plt.savefig(figfile)
