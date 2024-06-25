from champ.runSim import runSim
import numpy as np
n = 50
z = np.zeros(n)
for i, this_z in enumerate(z):
    if i!=0:
        #z[i] = z[i-1] + 0.5
        if i < 15 or i > 40:
            z[i] = z[i-1] + 0.1
        else:
            z[i] = z[i-1] + 3

sim_params = {'Q_w':0.05,
              'xc_n':500,
              'uplift':1e-5,
              'a':2,
              'K':1e-11,
              'mixed_regime':True,
              'trim':False,
            }

sim = runSim(
    n=n,
    L=500,
    z_arr = z,
    endtime=1,
    plotdir='./test-mixed-gvf/',
    flow_solver='GVF',
    snapshot_every=1,
    plot_every=1,
    sim_params=sim_params,
)