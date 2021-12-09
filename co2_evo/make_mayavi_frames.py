import os, sys, glob
import pickle
import numpy as np
from scipy import interpolate
from mayavi import mlab


def make_frames(snap_dir):
    mlab.figure(size=(2000,1400))

    snaplist = glob.glob(os.path.join(snap_dir, 'snap*.pkl'))
    snaplist.sort()

    anim_dir = os.path.join(snap_dir,'anim')
    if not os.path.isdir(anim_dir):
        os.makedirs(anim_dir)

    for j, pkl in enumerate(snaplist):
        print('processing snapshot ',j)
        f = open(pkl, 'rb')
        xcsim = pickle.load(f)
        f.close()

        nxcs = len(xcsim.xcs)
        npoints = len(xcsim.xcs[0].x)

        xcx = []
        xcy = []
        xcx_wet = []
        xcy_wet = []
        xLs = []
        xRs = []
        yw = []
        z_arr = xcsim.z_arr
        xc_z_arr = (z_arr[1:] + z_arr[:-1])/2.
        for i, xcs in enumerate(xcsim.xcs):
            xcs.setFD(xcs.fd)
            wantidx = xcs.y - xcs.ymin < 1.75*xcs.fd
            nx_wet = xcs.x[xcs.wetidx]
            ny_wet = xcs.y[xcs.wetidx]
            ymin = ny_wet.min()
            ny_wet = ny_wet - ymin + xc_z_arr[i]
            nx = xcs.x[wantidx]
            ny = xcs.y[wantidx]
            ymin = ny.min()
            ny = ny - ymin + xc_z_arr[i]
            tck, u = interpolate.splprep([nx, ny], u=None, k=1, s=0.)
            un = np.linspace(u.min(), u.max(), int(npoints/2))# if n!=nx.size else nx.size)
            nx, ny = interpolate.splev(un, tck, der=0)
            xcx.append(nx)
            xcy.append(ny)

            tck, u = interpolate.splprep([nx_wet, ny_wet], u=None, k=1, s=0.)
            un = np.linspace(u.min(), u.max(), int(npoints/2))# if n!=nx.size else nx.size)
            nx_wet, ny_wet = interpolate.splev(un, tck, der=0)
            xcx_wet.append(nx_wet)
            xcy_wet.append(ny_wet)

            L,R = xcs.findLR(xcs.fd)
            xL = xcs.x[L]
            xR = xcs.x[R]
            xLs.append(xL)
            xRs.append(xR)
            yw.append(xc_z_arr[i] + xcs.fd)
            
        xcx = np.array(xcx)
        xcy = np.array(xcy)

        xcx_wet = np.array(xcx_wet)
        xcy_wet = np.array(xcy_wet)

        xcz = np.ones((nxcs,int(npoints/2)))
        xcz = xcz.transpose() * np.arange(nxcs)/5.
        xcz = xcz.transpose()

        xwL = np.array([xLs, xRs])
        ywL = np.array([yw, yw])
        zwL = np.ones((2,nxcs))
        zwL = zwL * np.arange(nxcs)/5

        s1 = mlab.mesh(xcx, xcz, xcy, opacity=0.3, colormap='autumn')
        s2 = mlab.mesh(xcx_wet, xcz, xcy_wet, color=(50/255,131/255,168/255), opacity=0.6)
        s3 = mlab.mesh(xwL, zwL, ywL, color=(50/255,131/255,168/255), opacity=0.6)
        minz_arr = xcz[xcy>-5]
        if len(minz_arr) > 0:
            minz = minz_arr.min()
            plot_contact = True
        else:
            plot_contact = False
        maxz = xcz.max()
        minx = -5
        maxx = 5
        if plot_contact:
            s4 = mlab.mesh([[minx,maxx],[minx,maxx]],[[maxz,maxz],[minz,minz]],[[-5,-5],[-5,-5]],
                color=(184/255, 108/255,28/255), opacity=0.75)
        
        mlab.view(azimuth=300, roll=-60, elevation=70, distance=30)
        framenum = '%05d' % (j,)
        mlab.savefig(os.path.join(anim_dir,framenum+'.png'))
        mlab.clf()

#anim_sim()
#mlab.show()
    mlab.close()


if __name__ == '__main__':
    snap_dir = sys.argv[1]
    make_frames(snap_dir)
