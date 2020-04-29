from pylab import *
import pickle

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from CO2_sim_1D import CO2_1D

n=10
x = linspace(0,5000,n)
z = linspace(1,3,n)
r = 1.*ones(n-1)# + 0.1*np.random.rand(n-1)
#r[5] = 0.5
sim = CO2_1D(x,z, init_radii=r,
             Q_w=0.25,
             dH=30.,
             T_outside=20.,
             Ca_upstream=0.8,
             T_cave=10.,
             D_a=35., D_w=35.,
             adv_disp_stabil_factor=0.5,
             reduction_factor=0.01,
            dt_erode=1.,
            xc_n=1500, trim=True)
ntimes = 200000
T_cold = 0.
T_hot = 20.
plotdir='./high-with-CO2-figs/
for t in arange(ntimes):
    print('t=',t, '**********************')
    sim.run_one_step(T_outside_arr = [T_cold, T_hot])
    sim.z_arr[0] -= 0.00025
    if t % 100 == 0:
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
        plot(x, sim.h)
        plot(x,z)
        xlabel('Distance (m)')
        ylabel('Elevation (m)')
        legend(['h','z'])
        savefig(plotdir+'Elevation-Profile-'+timestep_str+'.png')

        figure()
        plot(x,sim.CO2_w)
        plot(x,sim.CO2_a)
        plot(x,sim.Ca)
        legend(['w','a','Ca'])
        savefig(plotdir+'Concentration-Profile-'+timestep_str+'.png')

        figure()
        xmid = (x[1:] + x[:-1])/2.
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
        if t % 1000 == 0:
            f = open(plotdir+'snapshot-'+timestep_str+'.pkl', 'wb')
            pickle.dump(sim, f)
