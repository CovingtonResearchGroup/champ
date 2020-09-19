#from pylab import *
import numpy as np
import matplotlib.pyplot as mpl
from CO2_sim_1D import CO2_1D

n=20
x = np.linspace(0,5000,n)
z = np.linspace(0,10,n)
np.random.seed(7)
r = 0.2*np.ones(n-1) + 0.1*np.random.rand(n-1)
#r[5] = 0.5
sim = CO2_1D(x,z, init_radii=r, 
             Q_w=.2, 
             T_outside=20., 
             D_a=35., D_w=35.,
             adv_disp_stabil_factor=0.5,
             reduction_factor=0.05,
            dt_erode=1.,
            xc_n=500)
ntimes = 1000
fd_old=0.
for t in np.arange(ntimes):
    print('t=',t)
    sim.calc_flow_depths()
    fd_new = sim.fd_mids[0]
    if t>0 and np.abs(fd_new-fd_old)>0.05:
        print(asdf)
    fd_old = fd_new
    sim.calc_air_flow()
    #if sim.A_a.min()==0:
    #    print(asdf)
    sim.calc_steady_state_transport()
    sim.erode_xcs()
    #if t==160:
    #    print(asdf)
    if t % 10 == 0:
        timestep_str = '%04d' % (t,)
        print("Plotting timestep: ",t)
        mpl.figure()
        mpl.plot(sim.xcs[0].x ,sim.xcs[0].y)
        wl = sim.fd_mids[0]+ sim.xcs[0].y.min()
        mpl.plot([-.5,.5], [wl,wl])
        #plot(sim.xcs[10].x ,sim.xcs[10].y)
        #plot(sim.xcs[-1].x ,sim.xcs[-1].y)
        mpl.savefig('./XC-'+timestep_str+'.png')
        mpl.figure()
        xmid = (x[1:] + x[:-1])/2.
        mpl.plot(x, sim.h)
        mpl.plot(x,z)
        mpl.plot(x,sim.CO2_w)
        mpl.plot(x,sim.CO2_a)
        mpl.plot(x,sim.Ca)
        mpl.legend(['h','z','w','a','Ca'])
        mpl.savefig('./Profile-'+timestep_str+'.png')
        mpl.close('all')
        
