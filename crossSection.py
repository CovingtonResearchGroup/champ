from numpy import sin, cos, pi, fabs, sign, roll, arctan2, diff, sum, hypot,\
                    logical_and, where, linspace, sqrt
import numpy as np
from scipy import interpolate
from scipy.optimize import brentq, root_scalar, minimize_scalar
import matplotlib.pyplot as plt


g=9.8 #m/s^2
rho_w = 998.2 #kg/m^3
SMALL = 1e-6
use_centroid_fraction =0.98#switch to  max vel at centroid if over this fraction of ymax
trim_factor = 2. #Trim xc points with y above trim_factor*fd

class CrossSection:

    def __init__(self, x, y, z0=0.01/30.):#z0=roughness height
        self.n = len(x)
        self.x = x
        self.y = y
        self.x_total = None
        self.y_total = None
        self.ymin = min(y)
        self.ymax = max(y)
        self.create_pm()
        self.z0=z0
        self.setFD(self.ymax - self.ymin)
        self.Q = 0.

    # Create arrays of x+1, y+1, x-1, x+1
    def create_pm(self):
        self.xm = roll(self.x, 1)
        self.ym = roll(self.y, 1)
        self.xp = roll(self.x, self.x.size-1)
        self.yp = roll(self.y, self.y.size-1)

    def calcP(self, wantidx=None):
        if type(wantidx) != type(None):
            l = hypot(self.x[wantidx] - self.xp[wantidx], self.y[wantidx] - self.yp[wantidx])
        else:
            l = hypot(self.x - self.xp, self.y - self.yp)
        if len(l)>0:
            maxidx = np.argmax(l)
            if l[maxidx]>100*l.min():
                #If perimeter connects across non-existent ceiling, cut this out
                l[maxidx] = 0.
        P = abs(sum(l))
        return P
        #self.pp = cumsum(self.l)
		#self.P = self.pp[-2]

	# Calculates area of the cross-section
    def calcA(self, wantidx=None):
        if type(wantidx) != type(None):
            if len(self.y[wantidx]>0):
                maxy = self.y[wantidx].max()
                self.sA = (self.xm[wantidx]*(self.y[wantidx]-maxy) - self.x[wantidx]*(self.ym[wantidx]-maxy)).sum() * 0.5
            else:
                return 0.
        else:
            maxy = self.y.max()
            self.sA = (self.xm*(self.y-maxy) - self.x*(self.ym-maxy)).sum() * 0.5
        A = fabs(self.sA)
        return A

    def create_A_interp(self, n_points=30):
        maxdepth = self.ymax - self.ymin
        max_interp = self.fd*1.25
        if max_interp > maxdepth:
            max_interp = maxdepth
        num_xc_points = len(self.y[self.y-self.ymin<max_interp])
        #print('xc points=',num_xc_points, ' maxdpeth=',maxdepth, '  max_interp=',max_interp)
        if num_xc_points<n_points/3.:
            n_points = int(np.round(num_xc_points/3.))
        depth_arr = np.linspace(0,max_interp,n_points)
        As = []
        for depth in depth_arr:
            wantidx = self.y-self.ymin<depth
            As.append(self.calcA(wantidx=wantidx))
        As = np.array(As)
        A_interp = interpolate.interp1d(depth_arr,As,kind='cubic',bounds_error=False,fill_value=(As[0],As[-1]))
        self.A_interp = A_interp

    def create_P_interp(self,n_points=30):
        maxdepth = self.ymax - self.ymin
        max_interp = self.fd*1.25
        if max_interp > maxdepth:
            max_interp = maxdepth
        num_xc_points = len(self.y[self.y-self.ymin<max_interp])
        #print('xc points=',num_xc_points, ' maxdpeth=',maxdepth, '  max_interp=',max_interp)
        if num_xc_points<n_points/3.:
            n_points = int(np.round(num_xc_points/3.))
        depth_arr = np.linspace(0,max_interp,n_points)
        Ps = []
        for depth in depth_arr:
            wantidx = self.y-self.ymin<depth
            l = hypot(self.x[wantidx] - self.xp[wantidx], self.y[wantidx] - self.yp[wantidx])
            Ps.append(abs(l.sum()))
        Ps = np.array(Ps)
        P_interp = interpolate.interp1d(depth_arr,Ps,kind='cubic',bounds_error=False,fill_value=(Ps[0],Ps[-1]))
        self.P_interp = P_interp


	# Find left and right points defining a height above the cross-section
	# bottom
    def findLR(self, h):
        a_h = self.ymin + h
        condL = logical_and(self.y > a_h, a_h >= self.yp)
        condR = logical_and(self.y < a_h, a_h <= self.yp)
        #below_idx = where(self.y<a_h)[0]
        #L = below_idx[0]
        L=where(condL)[0][0] + 1
        #R = below_idx[-1]
        R=where(condR)[0][0]
        return L,R

    def setFD(self,fd):
        maxdepth = self.ymax - self.ymin
        if fd>maxdepth:
            fd=maxdepth
        self.fd = fd
        self.wetidx = self.y - self.ymin<=fd

    def setMaxVelPoint(self,fd):
        self.setFD(fd)
        if fd>(self.ymax-self.ymin)*use_centroid_fraction:
            #Treat as full pipe
            mx, my = self.findCentroid()
        else:
            #open channel
            L,R = self.findLR(fd)
            mx = 0
            my = self.y[R]
        self.xmaxVel = mx
        self.ymaxVel = my

    def findCentroid(self):
        m = self.xm*self.y-self.x*self.ym
        A = self.calcA()
        cx = (1/(6*A))*((self.x + self.xm)*m).sum()#A was self.sA. not sure if this matters
        cy = (1/(6*A))*((self.y + self.ym)*m).sum()
        return cx, cy

    def calcR_l(self, wantidx=None):
        if type(wantidx) != type(None):
            self.r_l = np.hypot(self.x[wantidx]-self.xmaxVel, self.y[wantidx]-self.ymaxVel)
        else:
            self.r_l = np.hypot(self.x-self.xmaxVel, self.y-self.ymaxVel)
        return self.r_l

	# Find the value of U_max by weighted law of the wall
    def calcUmax(self, Q, method="area"):
        wetidx = self.wetidx#self.y-self.ymin <self.fd
        r_l = self.calcR_l(wantidx=wetidx)
        z0 = self.z0
        Ax = self.xmaxVel
        Ay = self.ymaxVel
        Bx = roll(self.x[wetidx],1)
        By = roll(self.y[wetidx],1)
        Cx = self.x[wetidx]
        Cy = self.y[wetidx]

        a_top = Ax*(By-Cy) + Bx*(Cy-Ay) + Cx*(Ay-By)
        a_i = fabs(a_top/2.)

        if method=="line":
            u_i = r_l/(r_l-z0) - 1/np.log(r_l/z0)

        if method=="area":
            u_top = r_l**2*np.log(r_l/z0) - 3./2*r_l**2 + 2.*r_l*z0 - z0**2/2.
            u_bottom = z0**2 + r_l**2 - 2*z0*r_l
            u_i = 1./np.log(r_l/z0)*u_top/u_bottom

        self.umax = Q/(a_i*u_i).sum()
        print("Umax=",self.umax)

    def calcT_b(self):
        vgrad2 = self.calcVGrad2()
        print('max vgrad2=',vgrad2.max())
        psi = self.calcPsi()
        Awet = self.calcA(wantidx=self.wetidx)
        self.T_b = psi*rho_w*Awet*vgrad2
        return self.T_b

    def calcVGrad2(self):
        wetidx=self.wetidx
        phi = np.arctan2(self.ymaxVel - self.y[wetidx],self.xmaxVel -self.x[wetidx])
        alpha = np.arctan2(self.yp[wetidx] - self.ym[wetidx], self.xp[wetidx]-self.xm[wetidx])
        alpha[0] = alpha[1]
        alpha[-1] = alpha[-2]
        self.vgrad2 = ((self.umax/self.z0)*(1./np.log(self.r_l/self.z0))*np.fabs(np.sin(phi-alpha)))**2.
        print('min sin term=',np.fabs(np.sin(phi-alpha)).min())
        return self.vgrad2

    def calcPsi(self):
        wetidx = self.wetidx
        l = np.hypot(self.x[wetidx] - self.xp[wetidx], self.y[wetidx] - self.yp[wetidx])
        print("min l=",l.min(), "max l=",l.max())
        sum = ((self.vgrad2)*l).sum()
        self.wet_ls = l
        self.psi = g*self.eSlope/sum
        return self.psi

    def setEnergySlope(self,slope):
        self.eSlope = slope

    def set_F_xc(self,F_xc):
        self.F_xc = F_xc

    def erode_power_law(self, a=1., K=1e-5):
        self.setMaxVelPoint(self.fd)
        self.calcUmax(self.Q)
        T_b = self.calcT_b()
        #print('max T_b=', T_b.max())
        self.dr = K*T_b**a
        self.erode(self.dr)

    def erode(self, dr, resample=True, n=None):
        if n==None:
            n=self.n
        wetidx=self.wetidx
        theta = np.arctan2(self.xp-self.xm, self.yp-self.ym)
        nx = self.x
        ny = self.y
        nx[wetidx] = self.x[wetidx] + dr*cos(theta[wetidx])
        ny[wetidx] = self.y[wetidx] - dr*sin(theta[wetidx])

        #Once flow drops far enough below ceiling, trim XC
        tmp_ymin = min(ny)
        trim_y = (self.fd*trim_factor + tmp_ymin)
        trim=False
        if trim_y<max(ny):
            trim = True
            #Initialize total xc arrays if first trimming event
            if type(self.x_total) == type(None):
                first_trim = True
                self.x_total = nx
                self.y_total = ny
            else:
                first_trim = False
            nx = nx[ny<trim_y]
            ny = ny[ny<trim_y]
            if not first_trim:
                #create new total xc arrays from old and wet portions
                self.x1 =x1= self.x_total[np.logical_and(self.x_total<0,self.y_total>trim_y)]
                self.y1 =y1= self.y_total[np.logical_and(self.x_total<0,self.y_total>trim_y)]
                #Slightly trim high-res XC to remove any connection across top
                self.x2 =x2= nx[ny<trim_y-0.02*(ny.max()-ny.min())]
                self.y2 =y2= ny[ny<trim_y-0.02*(ny.max()-ny.min())]
                self.x3 =x3= self.x_total[np.logical_and(self.x_total>0,self.y_total>trim_y)]
                self.y3 =y3= self.y_total[np.logical_and(self.x_total>0,self.y_total>trim_y)]
                x_total_tmp = np.concatenate([x1,x2,x3])
                y_total_tmp = np.concatenate([y1,y2,y3])
                tck, u = interpolate.splprep([x_total_tmp, y_total_tmp], u=None, k=1, s=0.)
                un = linspace(u.min(), u.max(), n)# if n!=nx.size else nx.size)
                self.x_total, self.y_total = interpolate.splev(un, tck, der=0)
                #print(asdsf)

        #print('trim_y=',trim_y)
        #print('len nx=', len(nx))
        #print('tmp ymin=',tmp_ymin)
        #print('fd=',self.fd)
        #Resample points by fitting spline
        if resample:
            #s = nx.size#+np.sqrt(2*nx.size)
            tck, u = interpolate.splprep([nx, ny], u=None, k=1, s=0.)
            un = linspace(u.min(), u.max(), n)# if n!=nx.size else nx.size)
            nx, ny = interpolate.splev(un, tck, der=0)


		# New coordinates
        y_roll = ny.size - ny.argmax()
        nx = roll(nx, y_roll)
        ny = roll(ny, y_roll)
#        if trim:
#            nx = nx[ny<trim_y-0.02*(ny.max()-ny.min())]#Slightly trim to remove roof
#            ny = ny[ny<trim_y-0.02*(ny.max()-ny.min())]
        self.x = nx
        self.y = ny
        self.create_pm()
#        self.x = (self.xp + self.xm)/2.
#        self.y = (self.yp + self.ym)/2.
#        self.create_pm()
        self.ymin = min(ny)
        #only reset ymax if it is increasing
        #if max(ny)>self.ymax:
        self.ymax = max(ny)
        self.n = len(nx)


    # Counter clockwise function to determine if we drew points in the correct
    # direction
    #def ccw(self, x, y, xm, ym, nx, ny):
    #    return (x - xm) * (ny - ym) > (y - ym) * (nx - xm)


    def calcNormalFlow(self,depth, slope,f=0.1, use_interp=True):
        if use_interp:
            Pw = self.P_interp(depth)#self.calcP(wantidx=wetidx)
            A = self.A_interp(depth)#self.calcA(wantidx=wetidx)
        else:
            wetidx = self.y-self.ymin<depth
            Pw = self.calcP(wantidx=wetidx)
            A = self.calcA(wantidx=wetidx)
            #print('Pw=',Pw, '  A=,',A)
        if Pw>0 and A>0 and depth>0:
            D_H = 4.*A/Pw
            Q = sign(slope)*A*sqrt(2.*g*abs(slope)*D_H/f)
            #print('Q=',Q)
        else:
            Q=0.
        return Q

    def normal_discharge_residual(self, depth, slope, f, desiredQ):
        return desiredQ - self.calcNormalFlow(depth,slope,f=f)

    def abs_normal_discharge_residual(self, depth, slope, f, desiredQ):
        return np.abs(desiredQ - self.calcNormalFlow(depth,slope,f=f))


    def head_discharge_residual(self,y_in,y_out,L,slope,f, desiredQ):
        avg_flow_depth = (y_out + y_in)/2.
        head_slope = (y_in - y_out)/L + slope
        return desiredQ - self.calcNormalFlow(avg_flow_depth,head_slope,f=f)

    def crit_flow_depth_residual(self,depth,Q):
        #wetidx = self.y - self.ymin < depth
        #print(wetidx)
        #print(depth)
        A = self.A_interp(depth)#self.calcA(wantidx=wetidx)
        L,R = self.findLR(depth)
        W = self.x[R] - self.x[L]
        #print(depth,W,A)
        return A**3/W - Q**2/g

    def abs_crit_flow_depth_residual(self,depth,Q):
        #wetidx = self.y - self.ymin < depth
        #print(wetidx)
        #print(depth)
        A = self.A_interp(depth)#self.calcA(wantidx=wetidx)
        L,R = self.findLR(depth)
        W = self.x[R] - self.x[L]
        #print(depth,W,A)
        return abs(A**3/W - Q**2/g)


    def calcNormalFlowDepth(self,Q, slope,f=0.1, old_fd=None):
        self.Q = Q
        maxdepth = self.ymax - self.ymin
        if type(old_fd) == type(None):
            upper_bound = maxdepth
        else:
            upper_bound = old_fd*1.1#25
        calcFullFlow = self.calcNormalFlow(maxdepth,slope, f=f, use_interp=False)
        if Q>=calcFullFlow and not self.ymax>self.y.max():
            #calc95PerFlow = self.calcNormalFlow(0.95*maxdepth, slope,f=f)
            #if Q>calc95PerFlow:
            #    print("Pipe is full.")
            return -1
        else:
            #dl = self.calcP()/self.n
            #dQ = abs(self.calcNormalFlow(self.fd+dl,slope,f=f) - self.calcNormalFlow(self.fd,slope,f=f))
            #print('dQ=',dQ)
            #sol = root_scalar(self.normal_discharge_residual, args=(slope,f,Q), x0=self.fd, x1=self.fd*0.75,xtol=dQ)
            #fd = sol.root
            #print('about to minimize')
            sol = minimize_scalar(self.abs_normal_discharge_residual, bounds=[SMALL,upper_bound], args=(slope,f,Q), method='bounded' )
            #print('found min')
            fd = sol.x
#            if fd<0:
#                fd = brentq(self.normal_discharge_residual, SMALL, maxdepth, args=(slope,f,Q))
            print('Q residual for',fd,' =', self.abs_normal_discharge_residual(fd,slope,f,Q))
            if fd >= maxdepth:
                self.setFD(fd)
                return -1
        self.setFD(fd)
        return self.fd

    def calcCritFlowDepth(self,Q):
        maxdepth = self.ymax - self.ymin
        fd = self.fd
        upper_bound = fd*1.25
        if upper_bound>0.99*maxdepth:
            upper_bound=0.99*maxdepth
        #Commented this out when changed to abs(), don't think we need it anymore
        #if self.crit_flow_depth_residual(fd*0.01,Q)>0:
        #    return -1
        #crit_depth = brentq(self.crit_flow_depth_residual, maxdepth*0.01, maxdepth*0.95, args=(Q,))
        sol = minimize_scalar(self.abs_crit_flow_depth_residual, bounds=[SMALL,upper_bound], args=(Q,), method='bounded')
        crit_depth = sol.x
        return crit_depth

    def calcPipeFullHeadGrad(self,Q,f=0.1):
        self.Q = Q
        Pw = self.calcP()
        A = self.calcA()
        D_H = 4.*A/Pw
        return (Q**2/A**2)*f/(2.*g*D_H)

    def calcUpstreamHead(self,Q,slope,y_out, L, f=0.1):
        maxdepth = self.ymax - self.ymin
        head_res_a = self.head_discharge_residual(SMALL, y_out,L,slope,f,Q)
        head_res_b = self.head_discharge_residual(maxdepth, y_out,L,slope,f,Q)
        if np.sign(head_res_a)*np.sign(head_res_b)==-1:
            sol = root_scalar(self.head_discharge_residual, args=(y_out,L,slope,f,Q),  bracket=(SMALL,maxdepth))#x0=y_out, x1=maxdepth)#, bracket=(SMALL,maxdepth) )
            y_in = sol.root
            return y_in
        else:
            return -1
