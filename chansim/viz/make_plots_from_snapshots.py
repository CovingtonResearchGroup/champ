"""Functions for creating plots from simulation snapshots. These
plots involve comparisons of pairs of simulations (normally with and
without CO2 exhange.) Designed to be run from command line with CO2 plotdir
as first command line argument and No CO2 exchange plotdir as second command
line argument. An optional third command line argument specifies the number
of snapshots to skip while plotting results. 
"""

from pylab import *
from glob import glob
import pickle
from olm.calcite import concCaEqFromPCO2
from matplotlib import ticker
from scipy.stats import linregress

#from . import CO2_sim_1D

###################################
## Plotting convenience funtions ##
###################################


def power_10_label(value,pos):
    if value > 0:
        power = floor(log10(value))
        rem = value/10**power
        #if rem == 1:
        #    return r"$10^{%01d}$" % log10(value)
        #else:
        startlabel = r'$' + str(rem)[0] + r'\times'
        endlabel = r"10^{%01d}$" % log10(value)
        return startlabel+endlabel
    else:
        return r'$0^{ }$'

font = {'size'   : 14}

matplotlib.rc('font', **font)


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


    snapshots = glob(plotdir+'*.pkl')
    snapshots.sort()
    erosion = []
    width = []
    slope = []
    A = []
    Pw = []
    CO2_w = []
    Ca = []
    fd = []
    T_b_mean = []
    T_b_max = []
    years = []
    Ca_Ca_eq = []
    x = []
    snap_every = 1000
    for snapnum, snapshot in enumerate(snapshots):
        print('loading snapshot ',snapnum)
        f = open(snapshot, 'rb')
        snap_sim = pickle.load(f)
        f.close()
        years.append(snapnum*snap_every*snap_sim.dt_erode)
        erosion.append(snap_sim.dz/snap_sim.dt_erode)
        Ca.append(snap_sim.Ca)
        CO2_w.append(snap_sim.CO2_w)
        A.append(snap_sim.A_w)
        Pw.append(snap_sim.P_w)
        fd.append(snap_sim.fd_mids)
        slope.append(snap_sim.slopes)
        #Get data from XCs
        these_widths = zeros(len(snap_sim.xcs))
        these_T_b_mean = zeros(len(snap_sim.xcs))
        these_T_b_max = zeros(len(snap_sim.xcs))
        for i, xc in enumerate(snap_sim.xcs):
            these_T_b_mean[i] = xc.T_b.mean()
            these_T_b_max[i] = xc.T_b.max()
            if snap_sim.flow_type[i] != 'full':
                L, R = xc.findLR(xc.fd)
                these_widths[i] = xc.x[R] - xc.x[L]
            else:
                these_widths[i] = 0.
        #Get data from nodes
        these_Ca_Ca_eq = zeros(len(snap_sim.x_arr))
        for i, dist in enumerate(snap_sim.x_arr):
            this_CO2_w = snap_sim.CO2_w[i]*snap_sim.pCO2_high
            this_Ca = snap_sim.Ca[i]*snap_sim.Ca_eq_0
            this_Ca_eq = concCaEqFromPCO2(this_CO2_w, T_C=snap_sim.T_cave)
            these_Ca_Ca_eq[i] = this_Ca/this_Ca_eq
        Ca_Ca_eq.append(these_Ca_Ca_eq)
        width.append(these_widths)
        T_b_mean.append(these_T_b_mean)
        T_b_max.append(these_T_b_max)
        x.append(snap_sim.x_arr)
        if snapnum==0:
            year0 = years[0]
            z_base0 = snap_sim.z_arr[0]
        if snapnum==1:
            year1 = years[1]
            z_base1 = snap_sim.z_arr[0]
            dz_dt_base = (z_base1 - z_base0)/(year1 - year0)

    erosion = array(erosion)
    width = array(width)
    slope = array(slope)
    A = array(A)
    Pw = array(Pw)
    CO2_w = array(CO2_w)
    Ca = array(Ca)
    Ca_Ca_eq = array(Ca_Ca_eq)
    fd = array(fd)
    T_b_mean = array(T_b_mean)
    T_b_max = array(T_b_max)
    years = array(years)
    x = array(x)
    return {'erosion':abs(erosion), 'width':width, 'slope':slope,
           'A':A, 'Pw':Pw, 'CO2_w':CO2_w, 'Ca':Ca, 'Ca_Ca_eq':Ca_Ca_eq,
           'fd':fd, 'T_b_mean':T_b_mean, 'T_b_max':T_b_max, 'years':years, 'x':x,
           'dz_dt_base':dz_dt_base}



def plot_erosion_slope_width_over_distance(plotdir_co2, co2, noco2, every=5):
    ####################################################
    ### Make plot of Erosion, slope,  width over time
    ####################################################

    cmap = get_cmap('nipy_spectral')
    n_steps= max([int(ceil(co2['Ca'].shape[0]/every)),int(ceil(noco2['Ca'].shape[0]/every))])
    cyc = cycler(color=[cmap(k) for k in linspace(0,1,n_steps)])
    rcParams['axes.prop_cycle'] = cyc

    max_years = max([co2['years'][-1], noco2['years'][-1]])
    cNorm = Normalize(vmin=0,vmax=max_years)
    sMap = cm.ScalarMappable(norm=cNorm, cmap = cmap)

    cb_formatter = ticker.ScalarFormatter()
    #cb_formatter.set_scientific(True)
    cb_formatter.set_powerlimits((0,0))
    cb_formatter.set_useMathText(True)

    x = (co2['x'][0,:-1] + co2['x'][0,1:])/2.

    fig, axs = subplots(4,2, figsize=(10,12),
                        gridspec_kw={'width_ratios':[0.85,1]},
                       sharex=True, sharey='row')

    axs[0,0].set_title('With CO2 Dynamics')
    axs[0,0].plot(x, abs(co2['erosion'].transpose()[::,::every]))
    axs[0,0].set_ylabel('Erosion (m/yr)')
    ymin = 0.9*min([co2['erosion'].min(), noco2['erosion'].min()])
    ymax =1.1* max([co2['erosion'].max(), noco2['erosion'].max()])
    axs[0,0].set_ylim([ymin,ymax])
    axs[0,0].grid('--', which='both')
    axs[0,0].tick_params(direction='in', labelsize=10)
    axs[0,0].plot([x[0], x[-1]], [-co2['dz_dt_base'],-co2['dz_dt_base']], '--k')

    axs[0,1].set_title('Without CO2 Dynamics')
    axs[0,1].plot(x, abs(noco2['erosion'].transpose()[::,::every]))
    cb=colorbar(sMap, ax=axs[0,1],format=cb_formatter)
    cb.set_label('Years')
    axs[0,1].grid('--', which='both')
    axs[0,1].tick_params(direction='in', labelsize=10)
    axs[0,1].plot([x[0], x[-1]], [-noco2['dz_dt_base'],-noco2['dz_dt_base']], '--k')

    axs[1,0].semilogy(x, abs(co2['slope'].transpose()[::,::every]))
    axs[1,0].set_ylabel('Slope')
    ymin = 0.9*min([co2['slope'].min(), noco2['slope'].min()])
    ymax =1.1* max([co2['slope'].max(), noco2['slope'].max()])
    axs[1,0].set_ylim([ymin,ymax])
    axs[1,0].grid('--', which='both')
    axs[1,0].tick_params(direction='in', labelsize=10)

    axs[1,1].semilogy(x, abs(noco2['slope'].transpose()[::,::every]))
    cb=colorbar(sMap, ax=axs[1,1],format=cb_formatter)
    cb.set_label('Years')
    axs[1,1].grid('--', which='both')
    axs[1,1].tick_params(direction='in', labelsize=10)

    axs[2,0].plot(x, abs(co2['width'].transpose()[::,::every]))
    axs[2,0].set_xlabel('Distance upstream (m)')
    axs[2,0].set_ylabel('Width (m)')
    ymin = 0.9*min([co2['width'].min(), noco2['width'].min()])
    ymax =1.1* max([co2['width'].max(), noco2['width'].max()])
    axs[2,0].set_ylim([ymin,ymax])
    axs[2,0].grid('--')
    axs[2,0].tick_params(direction='in', labelsize=10)


    axs[2,1].plot(x, abs(noco2['width'].transpose()[::,::every]))
    axs[2,1].set_xlabel('Distance upstream (m)')
    cb=colorbar(sMap, ax=axs[2,1],format=cb_formatter)
    cb.set_label('Years')
    axs[2,1].grid('--')
    axs[2,1].tick_params(direction='in', labelsize=10)

    axs[3,0].plot(co2['x'].transpose()[::,::every], abs(co2['Ca_Ca_eq'].transpose()[::,::every]))
    axs[3,0].set_xlabel('Distance upstream (m)')
    axs[3,0].set_ylabel(r'$Ca/Ca_{\rm eq}$')
    ymin = 0.9*min([co2['Ca_Ca_eq'].min(), noco2['Ca_Ca_eq'].min()])
    ymax =1.1* max([co2['Ca_Ca_eq'].max(), noco2['Ca_Ca_eq'].max()])
    axs[3,0].set_ylim([ymin,ymax])
    axs[3,0].grid('--')
    axs[3,0].tick_params(direction='in', labelsize=10)

    axs[3,1].plot(noco2['x'].transpose()[::,::every], abs(noco2['Ca_Ca_eq'].transpose()[::,::every]))
    axs[3,1].set_xlabel('Distance upstream (m)')
    cb=colorbar(sMap, ax=axs[3,1],format=cb_formatter)
    cb.set_label('Years')
    axs[3,1].grid('--')
    axs[3,1].tick_params(direction='in', labelsize=10)

    tight_layout(h_pad=0.0,w_pad=0.)

    savefig(plotdir_co2+'/1-Erosion-slope-width-v-distance.png')




def plot_ca_co2_over_time(plotdir_co2, co2, noco2):
    ####################################################
    ### Make plot of Erosion, slope,  width over time
    ####################################################

    cmap = get_cmap('viridis')
    n_xc= co2['Ca'].shape[1]
    rcParams['axes.prop_cycle'] = cycler(color=[cmap(k) for k in linspace(0,1,n_xc)])


    cNorm = Normalize(vmin=0,vmax=co2['x'][0,-1])
    sMap = cm.ScalarMappable(norm=cNorm, cmap = cmap)

    fig, axs = subplots(2,2, figsize=(10,7),
                        gridspec_kw={'width_ratios':[0.85,1]},
                       sharex=True, sharey='row')


    axs[0,0].set_title('With CO2 dynamics')
    axs[0,0].plot(co2['years'], co2['Ca'])
    axs[0,0].set_ylabel(r'${\rm Ca}$')
    ymin = 0.9998*min([co2['Ca'].min(), noco2['Ca'].min()])
    ymax = 1.0002*max([co2['Ca'].max(), noco2['Ca'].max()])
    axs[0,0].set_ylim([ymin,ymax])
    axs[0,0].tick_params(labelsize=10)

    axs[0,1].set_title('Without CO2 dynamics')
    axs[0,1].plot(noco2['years'], noco2['Ca'])
    cb = colorbar(sMap, ax=axs[0,1])
    cb.set_label('Distance upstream (m)')
    axs[0,1].tick_params(labelsize=10)

    axs[1,0].plot(co2['years'], co2['CO2_w'])
    axs[1,0].set_xlabel('Years')
    axs[1,0].set_ylabel('Dissolved CO2')
    ymin = 0.9*min([co2['CO2_w'].min(), noco2['CO2_w'].min()])
    ymax = 1.1*max([co2['CO2_w'].max(), noco2['CO2_w'].max()])
    axs[1,0].set_ylim([ymin,ymax])
    axs[1,0].tick_params(labelsize=10)

    axs[1,1].plot(noco2['years'], noco2['CO2_w'])
    axs[1,1].set_xlabel('Years')
    axs[1,1].xaxis.set_major_formatter(ticker.FuncFormatter(power_10_label))
    cb = colorbar(sMap,ax=axs[1,1])
    cb.set_label('Distance upstream (m)')
    axs[1,1].tick_params(labelsize=10)

    tight_layout(h_pad=0, w_pad=0)
    savefig(plotdir_co2+'/1-Ca-CO2-v-time.png')


def plot_morphodynamics(plotdir_co2, co2, noco2):
    cmap = get_cmap('viridis')
    n_xc= co2['Ca'].shape[1]
    rcParams['axes.prop_cycle'] = cycler(color=[cmap(k) for k in linspace(0,1,n_xc)])

    cNorm = Normalize(vmin=0,vmax=co2['x'][0,-1])
    sMap = cm.ScalarMappable(norm=cNorm, cmap = cmap)

    fig, axs = subplots(4,2, figsize=(10,16),
                        gridspec_kw={'width_ratios':[0.85,1]},
                       sharex=True, sharey='row')

    axs[0,0].set_title('With CO2 Dynamics')
    axs[0,0].plot(co2['years'], co2['Ca_Ca_eq'])
    axs[0,0].set_ylabel(r'${\rm Ca/Ca_{\rm eq}}$')
    ymin = 0.98*min([co2['Ca_Ca_eq'].min(), noco2['Ca_Ca_eq'].min()])
    ymax = 1.02*max([co2['Ca_Ca_eq'].max(), noco2['Ca_Ca_eq'].max()])
    axs[0,0].set_ylim([ymin,ymax])
    axs[0,0].tick_params(labelsize=10)

    axs[0,1].set_title('Without CO2 Dynamics')
    axs[0,1].plot(noco2['years'], noco2['Ca_Ca_eq'])
    cb = colorbar(sMap,ax=axs[0,1])
    cb.set_label('Distance upstream (m)')
    axs[0,1].tick_params(labelsize=10)

    axs[1,0].semilogy(co2['years'], co2['slope'])
    axs[1,0].set_ylabel('Slope')
    ymin = 0.9*min([co2['slope'].min(), noco2['slope'].min()])
    ymax = 1.1*max([co2['slope'].max(), noco2['slope'].max()])
    axs[1,0].set_ylim([ymin,ymax])
    axs[1,0].tick_params(labelsize=10)

    axs[1,1].semilogy(noco2['years'], noco2['slope'])
    cb = colorbar(sMap,ax=axs[1,1])
    cb.set_label('Distance upstream (m)')
    axs[1,1].tick_params(labelsize=10)

    axs[2,0].plot(co2['years'], co2['width'])
    axs[2,0].set_ylabel('Width (m)')
    ymin = 0.9*min([co2['width'].min(), noco2['width'].min()])
    ymax = 1.051*max([co2['width'].max(), noco2['width'].max()])
    axs[2,0].set_ylim([ymin,ymax])
    axs[2,0].tick_params(labelsize=10)

    axs[2,1].plot(noco2['years'], noco2['width'])
    cb = colorbar(sMap,ax=axs[2,1])
    cb.set_label('Distance upstream (m)')
    axs[2,1].tick_params(labelsize=10)

    axs[3,0].plot(co2['years'], abs(co2['erosion']))
    axs[3,0].set_xlabel('Years')
    axs[3,0].set_ylabel('Erosion (m/yr)')
    ymin = min([co2['erosion'].min(), noco2['erosion'].min()])
    ymax = 1.1*max([co2['erosion'].max(), noco2['erosion'].max()])
    axs[3,0].set_ylim([ymin,ymax])
    axs[3,0].plot([0,100000],[0.00125,0.00125], 'k--')
    axs[3,0].plot([co2['years'][0], co2['years'][-1]], [-co2['dz_dt_base'],-co2['dz_dt_base']], '--k')
    axs[3,0].tick_params(labelsize=10)

    axs[3,1].plot(noco2['years'], abs(noco2['erosion']))
    axs[3,1].set_xlabel('Years')
    cb = colorbar(sMap,ax=axs[3,1])
    cb.set_label('Distance upstream (m)')
    axs[3,1].plot([0,100000],[0.00125,0.00125], 'k--')
    axs[3,1].xaxis.set_major_formatter(ticker.FuncFormatter(power_10_label))
    axs[3,1].plot([noco2['years'][0], noco2['years'][-1]], [-noco2['dz_dt_base'],-noco2['dz_dt_base']], '--k')
    axs[3,1].tick_params(labelsize=10)

    tight_layout(h_pad=0, w_pad=0)
    savefig(plotdir_co2+'/1-Morphodynamics.png')


## This plot seems too crammed for use. All points quickly
## approach scaling relationship
"""
def slope_width_all_times(plotdir_co2, co2, noco2):

    cmap = get_cmap('nipy_spectral')
    n_steps= max([int(ceil(co2['Ca'].shape[0]/every)),int(ceil(noco2['Ca'].shape[0]/every))])
    cyc = cycler(color=[cmap(k) for k in linspace(0,1,n_steps)])
    rcParams['axes.prop_cycle'] = cyc

    max_years = max([co2['years'][-1], noco2['years'][-1]])
    cNorm = Normalize(vmin=0,vmax=max_years)
    sMap = cm.ScalarMappable(norm=cNorm, cmap = cmap)

    cb_formatter = ticker.ScalarFormatter()
    cb_formatter.set_powerlimits((0,0))
    cb_formatter.set_useMathText(True)

    fig, axs = subplots(1,2, figsize=(10,6),
                        gridspec_kw={'width_ratios':[0.85,1]})
#                       sharex=True, sharey=True)

    axs[0,0].semilogy(co2['width'].transpose()[::,::every], co2['slope'].transpose()[::,::every])
    xlabel('Width (m)')
    ylabel('Slope')

    axs[0,1].semilogy(noco2['width'].transpose()[::,::every], noco2['slope'].transpose()[::,::every])
    xlabel('Width (m)')
    ylabel('Slope')
    ax = gca()
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    cb=colorbar(sMap, ax=axs[0,1],format=cb_formatter)
    cb.set_label('Years')


    tight_layout()
    savefig(plotdir_co2+'/1-Slope-width-all-times.png')
"""


def final_step_morphology(plotdir_co2, co2, noco2):

    figure()
    semilogy(co2['width'][-1], co2['slope'][-1], 'ko')
    semilogy(noco2['width'][-1], noco2['slope'][-1], 'bs')
    w_i = co2['width'][-1][0]
    s_i = co2['slope'][-1][0]
    w_f = co2['width'][-1][-1]
    w_opt = linspace(w_i,w_f,20)
    C_prop = s_i*w_i**(16./3.)
    s_opt = C_prop*w_opt**(-16./3.)
    plot(w_opt, s_opt, '--k')
    m, b, r, p, err = linregress(log10(co2['width'][-1]), log10(co2['slope'][-1]))
    plot(co2['width'][-1], co2['width'][-1]**m * 10**b, ':k')

    xlabel('Width (m)')
    ylabel('Slope')
    ax = gca()
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    opt = -16./3.
    opt_str = r'$S \propto w^{' + str(opt)[:5]+'}$'
    fit_str = r'$S \propto w^{' + str(m)[:5]+'}$'

    legend(['With CO2', 'Without CO2', 'Optimal: '+opt_str, 'Fit: '+fit_str])
    tight_layout()
    savefig(plotdir_co2+'/1-Final-Morphology.png')

    #s_f = co2['slope'][-1][-1]



def make_plots_from_snapshots(plotdir_co2, plotdir_no_co2, every=10):
    print('Loading results with CO2...')
    co2 = get_results(plotdir_co2)
    print('Loading results without CO2...')
    noco2 = get_results(plotdir_no_co2)

    plot_erosion_slope_width_over_distance(plotdir_co2, co2, noco2, every=every)
    plot_ca_co2_over_time(plotdir_co2, co2, noco2)
    plot_morphodynamics(plotdir_co2, co2, noco2)
    final_step_morphology(plotdir_co2, co2, noco2)
    #slope_width_all_times(plotdir_co2, co2, noco2)

if __name__ == '__main__':
    plotdir_co2 = sys.argv[1]
    plotdir_no_co2 = sys.argv[2]
    if len(sys.argv) > 3:
        every = int(sys.argv[3])
    else:
        every=5
    make_plots_from_snapshots(plotdir_co2, plotdir_no_co2, every=every)
