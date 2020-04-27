import numpy as np
from scipy.linalg import solve_banded
from olm.calcite import concCaEqFromPCO2, createPalmerInterpolationFunctions, palmerRate, calc_K_H,\
                        solutionFromCaPCO2, palmerFromSolution
from olm.general import CtoK

from crossSection import CrossSection
from ShapeGen import genCirc, genEll
from numpy.random import rand,seed
from scipy.optimize import brentq
from scipy.signal import savgol_filter

#Constants
g=9.8#m/s^2
rho_limestone = 2.6#g/cm^3
rho_w = 998.2#kg/m^3
D_Ca = 10**-9#m^2/s
nu = 1.3e-6#m^2/s at 10 C
Sc = nu/D_Ca
g_mol_CaCO3 = 100.09
L_per_m3 = 1000.
secs_per_year =  3.154e7
secs_per_hour = 60.*60.
cm_m = 100.

###
## gas trasfer vel, typical values ~10 cm/hr for small streams (Wanningkhof 1990)
####

class CO2_1D:

    def __init__(self, x_arr, z_arr, D_w=30., D_a=30, Q_w=0.1,
    pCO2_high=5000*1e-6, pCO2_outside=500*1e-6, f=0.1,
    T_cave=10, T_outside=20., gas_transf_vel=0.1/secs_per_hour, abs_tol=1e-5, rel_tol=1e-5,
    CO2_w_upstream=1., Ca_upstream=0.5, h0=0., rho_air_cave = 1.225, dH=50.,
    init_shape = 'circle', init_radii = 0.5, init_offsets = 0., xc_n=1000,
    adv_disp_stabil_factor=0.9, impure=True,reduction_factor=0.1, dt_erode=1.,
    downstream_bnd_type='normal'):
        self.n_nodes = x_arr.size
        self.L = x_arr.max() - x_arr.min()
        self.x_arr = x_arr
        self.z_arr = z_arr#z is zero of xc coords
        self.L_arr = x_arr[1:]- x_arr[:-1]
        self.D_w = D_w
        self.D_a = D_a
        self.adv_disp_stabil_factor = adv_disp_stabil_factor
        #Calculate stable timestep
        self.dt_ad_dim = adv_disp_stabil_factor*self.L_arr[0]**2./np.min([self.D_w,self.D_a]) #endtime/(ntimes-1)
        self.Q_w = Q_w
        self.Q_a = 0.
        self.pCO2_high = pCO2_high
        self.pCO2_outside = pCO2_outside
        self.rho_air_cave = rho_air_cave
        self.dH = dH
        self.T_cave = T_cave
        self.T_cave_K = CtoK(T_cave)
        self.T_outside = T_outside
        self.T_outside_K = CtoK(T_outside)
        self.gas_transf_vel = gas_transf_vel
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.CO2_w_upstream = CO2_w_upstream
        self.Ca_upstream = Ca_upstream
        self.reduction_factor = reduction_factor
        self.dt_erode = dt_erode
        self.xc_n = xc_n
        self.downstream_bnd_type = downstream_bnd_type

        self.V_w = np.zeros(self.n_nodes - 1)
        self.V_a = np.zeros(self.n_nodes - 1)
        self.A_w = np.zeros(self.n_nodes - 1)
        self.A_a = np.zeros(self.n_nodes - 1)
        self.P_w = np.zeros(self.n_nodes - 1)
        self.P_a = np.zeros(self.n_nodes - 1)
        self.D_H_w = np.zeros(self.n_nodes - 1)
        self.D_H_a = np.zeros(self.n_nodes - 1)
        self.W = np.zeros(self.n_nodes - 1)
        self.Lambda_a = np.zeros(self.n_nodes - 1)
        self.Lambda_w = np.zeros(self.n_nodes - 1)

        self.fd_mids = np.zeros(self.n_nodes-1)
        self.init_offsets = np.ones(self.n_nodes-1) * init_offsets
        self.up_offsets = np.zeros(self.n_nodes-1)
        self.down_offsets = np.zeros(self.n_nodes-1)
        self.h = np.zeros(self.n_nodes)
        self.h0 = h0
        self.f=f
        self.flow_type = np.zeros(self.n_nodes-1,dtype=object)

        self.K_H = calc_K_H(self.T_cave_K) #Henry's law constant mols dissolved per atm
        self.Ca_eq_0 = concCaEqFromPCO2(self.pCO2_high, T_C=T_cave)
        self.palmer_interp_funcs = createPalmerInterpolationFunctions(impure=impure)


        #Initialize cross-sections
        self.xcs = []
        #self.maxdepths = np.zeros(self.n_nodes-1)
        self.radii = init_radii*np.ones(self.n_nodes-1)
        ymins = []
        for i in np.arange(self.n_nodes-1):
            x, y = genCirc(self.radii[i],n=xc_n)
            y = y + self.init_offsets[i]
            this_xc = CrossSection(x,y)
            self.xcs.append(this_xc)
            #self.maxdepths[i] = this_xc.ymax - this_xc.ymin
            ymins.append(this_xc.ymin)
        self.ymins = np.array(ymins)
        #Reset z to bottom of cross-sections
        self.z_arr[1:] = self.z_arr[1:] + self.ymins
        self.z_arr[0] = self.z_arr[0] + self.ymins[0]
        self.slopes = (self.z_arr[1:] - self.z_arr[:-1])/(self.x_arr[1:] - self.x_arr[:-1])


        #Create b arrays for each concentration variable
        self.bCO2_a = np.zeros(self.n_nodes-1)
        self.bCO2_w = np.zeros(self.n_nodes-1)
        self.bCa = np.zeros(self.n_nodes-1)



    def calc_flow_depths(self):
        # Loop through cross-sections and solve for flow depths,
        # starting at downstream end
        for i, xc in enumerate(self.xcs):
            old_fd = self.fd_mids[i]
            if old_fd <=0:
                #print('zero or neg flow depth')
                old_fd = xc.ymax - xc.ymin
            xc.create_A_interp()
            xc.create_P_interp()
            #print('xc=',i)
            #Try calculating flow depth
            backflooded= (self.h[i]-self.z_arr[i+1]-xc.ymax+xc.ymin+self.up_offsets[i])>0#Should I really use the offset here?
            over_normal_capacity=False
            if not backflooded:
                norm_fd = xc.calcNormalFlowDepth(self.Q_w,self.slopes[i],f=self.f, old_fd=old_fd)
                if norm_fd==-1:
                    over_normal_capacity=True
            #print('norm_fd=', norm_fd, '  maxdepth=',xc.ymax - xc.ymin)
            if over_normal_capacity or backflooded:
                self.flow_type[i] = 'full'
                if i==0:
                    #if downstream boundary set head to top of pipe
                    self.h[0]= self.z_arr[0] + xc.ymax - xc.ymin
                #We have a full pipe, calculate head gradient instead
                delh = xc.calcPipeFullHeadGrad(self.Q_w,f=self.f)
                self.h[i+1] = self.h[i] + delh * self.L_arr[i]
                self.fd_mids[i] = xc.ymax - xc.ymin
            else:
                #crit_fd = xc.calcCritFlowDepth(self.Q_w)
                y_star = norm_fd#min([crit_fd,norm_fd])
                y_out = self.h[i] - self.z_arr[i]  + self.down_offsets[i]
                downstream_critical = y_star>y_out and y_star>0# and i>0
                partial_backflood = norm_fd < self.h[i] - self.z_arr[i+1]  +self.up_offsets[i]
                downstream_less_normal = norm_fd>y_out
                if partial_backflood: #upstream node is flooded above normal depth
                    self.flow_type[i] = 'pbflood'
                    y_in = xc.calcUpstreamHead(self.Q_w,self.slopes[i],y_out,self.L_arr[i],f=self.f)
                    if y_in>0:
                        self.h[i+1] = self.z_arr[i+1] + y_in - self.up_offsets[i]
                        self.fd_mids[i] = (y_out + y_in)/2.
                    else:
                        #We need full pipe to push needed Q
                        delh = xc.calcPipeFullHeadGrad(self.Q_w,f=self.f)
                        self.h[i+1] = self.h[i] + delh * self.L_arr[i]
                        self.fd_mids[i] = xc.ymax - xc.ymin
                        self.flow_type[i] = 'full'
                #elif downstream_critical:
                #    self.flow_type[i] = 'dwnscrit'
                #    #Use minimum of critical or normal depth for downstream y
                #    y_in = xc.calcUpstreamHead(self.Q_w,self.slopes[i],y_star,self.L_arr[i],f=self.f)
                #    self.fd_mids[i] = (y_in + y_star)/2.
                #    self.h[i+1] = self.z_arr[i+1] + y_in
                #    if i==0:
                #        self.h[0]=self.z_arr[0] + y_star#norm_fd #y_star
                #elif downstream_less_normal:
                #    self.flow_type[i] = 'dwnslessnorm'
                #    y_in = xc.calcUpstreamHead(self.Q_w,self.slopes[i],y_out,self.L_arr[i],f=self.f)
                #    self.h[i+1] = self.z_arr[i+1] + y_in
                #    self.fd_mids[i] = (y_out+y_in)/2.
                else:
                    self.flow_type[i] = 'norm'
                    if i==0:
                        self.h[i] = norm_fd + self.z_arr[i]  - self.down_offsets[i]
                    #dz = slopes[i]*(x[i+1] - x[i])
                    self.h[i+1] = self.z_arr[i+1] + norm_fd  - self.up_offsets[i]
                    self.fd_mids[i] = norm_fd
            # Calculate flow areas, wetted perimeters, hydraulic diameters,
            # free surface widths, and velocities
            wetidx = (xc.y - xc.ymin) < self.fd_mids[i]
            self.A_w[i] = xc.calcA(wantidx=wetidx)
            #print(self.A_w[i])
            self.P_w[i] = xc.calcP(wantidx=wetidx)
            self.V_w[i] = -self.Q_w/self.A_w[i]
            self.D_H_w[i] = 4*self.A_w[i]/self.P_w[i]
            #print(self.flow_type[i])
            if self.flow_type[i] != 'full':
                #print('getting width')
                L,R = xc.findLR(self.fd_mids[i])
                #print('got L,R')
                self.W[i] = xc.x[R] - xc.x[L]
            else:
                self.W[i] = 0.
            #Set water line in cross-section object
            #print('setting fd')
            xc.setFD(self.fd_mids[i])
            #print('done with this xc')

    def calc_air_flow(self):
        dT = self.T_outside - self.T_cave
        dP_tot = self.rho_air_cave*g*self.dH*dT/self.T_outside_K
        R_air = np.zeros(self.n_nodes-1)
        for i, xc in enumerate(self.xcs):
            if type(xc.x_total) != type(None):
                dryidx = (xc.y_total - xc.y_total.min())>self.fd_mids[i]
                self.A_a[i] = xc.calcA(wantidx=dryidx, total=True, zeroAtUmax=False)
                self.P_a = xc.calcP(wantidx=dryidx, total=True)
            else:
                dryidx = (xc.y - xc.ymin)>self.fd_mids[i]
                self.A_a[i] = xc.calcA(wantidx=dryidx, zeroAtUmax=False)
                self.P_a = xc.calcP(wantidx=dryidx)
            if self.A_a[i]>0:
                self.D_H_a[i] = 4.*self.A_a[i]/self.P_a
                R_air[i] = self.rho_air_cave*self.f*self.L_arr[i]/(2.*self.D_H_a[i]*self.A_a[i]**2.)
            else:
                R_air[i] = np.inf
        self.Q_a = -np.sqrt(abs(dP_tot/R_air.sum()))*np.sign(dP_tot)

        if self.A_a.min()>0:
            self.V_a = self.Q_a/self.A_a
        else:
            self.V_a[:] = 0.
        print("Air discharge = ",self.Q_a, ' m^3/s')

    def calc_steady_state_transport(self, palmer=False):
        self.update_dimnless_params()
        self.initialize_conc_arrays()

        if np.sign(self.V_a[0])==np.sign(self.V_w[0]) or self.V_a[0]==0.:
            self.calc_conc_from_upstream( palmer=palmer)
        else:
            #Calculate air downstream bnd value using linear shooting method
            g1 = 1.#self.pCO2_high*0.5#pCO2_outside
            g2 = 0.5#self.pCO2_high
            self.calc_conc_from_upstream(CO2_a_upstream=g1, palmer=palmer)
            CO2_down_1 = self.CO2_a[0]
            self.calc_conc_from_upstream(CO2_a_upstream=g2, palmer=palmer)
            CO2_down_2 = self.CO2_a[0]
            #print('CO2_down_1=',CO2_down_1, '  CO2_down_2=',CO2_down_2)
            if CO2_down_1 != CO2_down_2: #These are equal for very low airflow, for which we ignore this calculation
                CO2_a_upstream_corrected = g1 + \
                    (g2 - g1)/(CO2_down_2-CO2_down_1)*(self.pCO2_outside/self.pCO2_high-CO2_down_1)
                self.calc_conc_from_upstream(CO2_a_upstream=CO2_a_upstream_corrected,palmer=palmer)
                if np.abs(self.CO2_a[0] - self.pCO2_outside/self.pCO2_high) > self.abs_tol or True in np.isnan(self.CO2_a):
                    print("Linear shooting method failed...residual=",self.CO2_a[0] - self.pCO2_outside/self.pCO2_high)
                    CO2_a_upstream_brent = brentq(self.downstream_CO2_residual, self.pCO2_outside/self.pCO2_high, 1., xtol=self.abs_tol)
            else:
                #For case of low airflow velocity, need to fix upstream air CO2 to water value
                self.CO2_a[-1] = self.CO2_w[-1]

    def downstream_CO2_residual(self,CO2_a_upstream):
        mod_CO2_downstream = self.calc_conc_from_upstream(CO2_a_upstream=CO2_a_upstream)
        return self.CO2_a[0] - self.pCO2_outside/self.pCO2_high

    def calc_conc_from_upstream(self, CO2_a_upstream=None, palmer=False):
        if CO2_a_upstream != None:
            self.CO2_a[-1] = CO2_a_upstream
        if self.A_a.min()>0:
            K_w = self.gas_transf_vel*self.W/self.A_w
            K_a = self.gas_transf_vel*self.W/self.A_a
        else:
            K_a = 0.0*self.W
            K_w = 0.0*self.W
        #Loop backwards through concentration arrays
        F = np.zeros(self.n_nodes - 1)

        #Check this, not sure it's right
        mm_yr_to_mols_sec = 100.*rho_limestone/g_mol_CaCO3/secs_per_year/100./(self.D_H_w/2.)

        for i in np.arange(self.n_nodes-1, 0, -1):
            this_CO2_w = self.CO2_w[i]*self.pCO2_high
            this_CO2_a = self.CO2_a[i]*self.pCO2_high
            #print("CO2_a=",this_CO2_a,"  CO2_w=",this_CO2_w)
            this_Ca = self.Ca[i]*self.Ca_eq_0
            if palmer:
                sol = solutionFromCaPCO2(this_Ca, this_CO2_w, T_C=self.T_cave)
                F[i-1] = palmerFromSolution(sol, PCO2=this_CO2_w)
                R = F[i-1]*mm_yr_to_mols_sec[i-1]
            else:
                this_xc = self.xcs[i-1]
                if self.flow_type[i-1] == 'norm':
                    #use bed slope for energy slope in this case
                    eSlope = self.slopes[i-1]
                else:
                    eSlope = (self.h[i] - self.h[i-1])/self.L_arr[i-1]
                #print('xc i=',i-1,'eSlope=',eSlope)
                this_xc.setEnergySlope(eSlope)
                this_xc.setMaxVelPoint(self.fd_mids[i-1])
                this_xc.calcUmax(self.Q_w)
                T_b = this_xc.calcT_b()
                #print('i=',i)
                #print('min T_b=', T_b.min())
                #print('max T_b=', T_b.max())
                #print('mean T_b=', T_b.mean())
                if T_b.min()<0:
                    print(asdf)
                eps = 5*nu*Sc**(-1./3.)/np.sqrt(T_b/rho_w)
                #print('eps=',eps.mean())
                Ca_Eq = concCaEqFromPCO2(this_CO2_w, T_C=self.T_cave)
                #print('Ca=',this_Ca,'   Ca_eq=',Ca_Eq)
                F_xc = self.reduction_factor*D_Ca/eps*(Ca_Eq - this_Ca)*L_per_m3
                #Smooth F_xc with savgol_filter
                window = int(np.ceil(len(F_xc)/5)//2*2+1)
                F_xc = savgol_filter(F_xc,window,3)
                if F_xc.min() < 0:
                    #Don't allow precipitation
                    F_xc = F_xc*0.0
                this_xc.set_F_xc(F_xc)
                P_w = this_xc.wet_ls.sum()
                F[i-1] = np.sum(F_xc*this_xc.wet_ls)/P_w #Units of F are mols/m^2/sec
                #print('F=',F[i-1])
                R = F[i-1]*P_w*self.L_arr[i-1]#4.*F[i-1]/self.D_H_w[i-1]
            R_CO2 = R/self.K_H
            #dx is negative, so signs on dC terms flip
            if self.A_a.min()>0:
                if self.V_a[i-1] != 0:
                    dCO2_a = -self.L_arr[i-1]*K_a[i-1]/self.V_a[i-1]*(this_CO2_w - this_CO2_a)
                    if np.abs(dCO2_a) > np.abs(this_CO2_w - this_CO2_a):
                        #Don't let CO2 change more than difference between air and water
                        #print("Low v case")
                        dCO2_a = 0.#(this_CO2_w - this_CO2_a)
                        this_CO2_a = this_CO2_w
                        self.CO2_a[i] = self.CO2_w[i]
                else:
                    #For zero airflow, CO2_a goes to CO2_w
                    dCO2_a = (this_CO2_w - this_CO2_a)
            else:
                dCO2_a = 0.
                this_CO2_a = this_CO2_w
                self.CO2_a[i] = self.CO2_w[i]
            dCO2_w = self.L_arr[i-1]*K_w[i-1]/self.V_w[i-1]*(this_CO2_w - this_CO2_a) - R_CO2/self.Q_w/L_per_m3#R_CO2/self.V_w[i-1]
            dCa = R/self.Q_w/L_per_m3#-self.L_arr[i-1]*R/self.V_w[i-1]
            #print(dCO2_a,dCO2_w,dCa)
            self.CO2_a[i-1] = (this_CO2_a + dCO2_a)/self.pCO2_high
            self.CO2_w[i-1] = (this_CO2_w + dCO2_w)/self.pCO2_high
            self.Ca[i-1] = (this_Ca + dCa)/self.Ca_eq_0
            if self.CO2_a[i-1]<0:
                self.CO2_a[i-1]=0.
            if self.CO2_w[i-1]<0:
                self.CO2_w[i-1]=0.
            if self.Ca[i-1]<0:
                self.Ca[i-1]=0.

        self.F = F

    def erode_xcs(self):
        F_to_m_yr = g_mol_CaCO3*secs_per_year/rho_limestone/cm_m**3
        old_ymins = self.ymins.copy()
        for i,xc in enumerate(self.xcs):
            xc.dr = F_to_m_yr*xc.F_xc*self.dt_erode
            #print('i=',i,'  max dr=', xc.dr.max(), '  max F_xc=',xc.F_xc.max())
            #xc.dr = savgol_filter(xc.dr,15,3,mode='wrap')
            xc.erode(xc.dr)
            self.ymins[i]= xc.ymin
        #Adjust slopes
        dz = self.ymins - old_ymins
        self.dz = dz
        #print('dz=',dz)
        Celerity_times_dt = np.abs(max(dz/self.slopes))
        CFL = Celerity_times_dt/min((self.x_arr[1:] - self.x_arr[:-1]))
        print('CFL=',CFL)
        self.z_arr[1:] = self.z_arr[1:] + dz
        #bed_elevs = self.z_arr[1:] + ymins
        #new_slopes = self.slopes
        #new_slopes[1:-1] = (bed_elevs[2:] - bed_elevs[:-2])/(self.L_arr[2:] + self.L_arr[:-2])
        #new_slopes[0] = (bed_elevs[0] - self.z_arr[0])/self.L_arr[0]
        #new_slopes[-1] = new_slopes[-2]
        self.slopes = (self.z_arr[1:] - self.z_arr[:-1])/(self.x_arr[1:] - self.x_arr[:-1])
        """
        #        print('ymins=',ymins)
        #        dys = ymins[0:-1]+self.down_offsets[0:-1] - (ymins[1:]+self.up_offsets[1:])
                print('dys=',dys)
                for i,xc in enumerate(self.xcs):
                    if i==0:
                        dy_down = 0.
                    else:
                        dy_down = -0.5*dys[i-1]
                    if i==len(self.xcs)-1:
                        dy_up = 0.
                    else:
                        dy_up = 0.5*dys[i]
                    self.down_offsets[i] -= dy_down
                    self.up_offsets[i] -= dy_up
                    dslope = (dy_down - dy_up)/self.L_arr[i]
                    print('xc=',i,'  dslope=',dslope)
                    print('down_offset=',self.down_offsets[i],'up_offset=',self.up_offsets[i])
                    self.slopes[i] = self.slopes[i] + dslope
                print('dys after=', ymins[0:-1]+self.down_offsets[0:-1] - (ymins[1:]+self.up_offsets[1:]))
        """

    def set_T_outside(self, T_outside_C):
        self.T_outside = T_outside_C
        self.T_outside_K = CtoK(T_outside_C)


    def update_adv_disp_M_water(self):
        #Construct Adv-disp matrix for water
        dt = self.dt_ad
        dx = self.dx_ad
        Pe_w = self.Pe_w
        M_upper_water = (self.V_w/(4.*dx*self.V_w_mean) - 1./(2.*Pe_w*dx**2.))*np.ones(self.n_nodes-1)
        M_lower_water = (-self.V_w/(4.*dx*self.V_w_mean) - 1./(2.*Pe_w*dx**2.))*np.ones(self.n_nodes-1)
        M_mid_water = (1./dt+1./(Pe_w*dx**2.))*np.ones(self.n_nodes-1)
        M_upper_water[0] = 0.
        M_lower_water[-1] = 0.
        #bnds for positive V_w
        #M_lower_water[-2] = -dt/(2.*dx)
        #M_mid_water[-1] = 1. + dt/(2*dx)
        M_upper_water[1] = self.V_w[0]/(2.*dx*self.V_w_mean)
        M_mid_water[0] = 1./dt - self.V_w[0]/(2*dx*self.V_w_mean)
        self.M_water = np.vstack((M_upper_water, M_mid_water, M_lower_water))

    def update_adv_disp_M_air(self):
        dt = self.dt_ad
        dx = self.dx_ad
        Pe_a = self.Pe_a
        T = self.T
        M_upper_air = (self.V_a/(4.*dx*self.V_w_mean) - 1./(2.*Pe_a*T*dx**2.))*np.ones(self.n_nodes-1)
        M_lower_air = (-self.V_a/(4.*dx*self.V_w_mean) - 1./(2.*Pe_a*T*dx**2.))*np.ones(self.n_nodes-1)
        M_mid_air = (1./dt+1./(Pe_a*T*dx**2.))*np.ones(self.n_nodes-1)
        M_upper_air[0] = 0.
        M_lower_air[-1] = 0.
        if self.V_a[0]>0:
            M_lower_air[-2] = -self.V_a[-1]/(2.*dx*self.V_w_mean)
            M_mid_air[-1] = 1./dt + self.V_a[-1]/(2*dx*self.V_w_mean)
        else:
            M_upper_air[1] = self.V_a[0]/(2.*dx*self.V_w_mean)
            M_mid_air[0] = 1./dt - self.V_a[0]/(2*dx*self.V_w_mean)
        self.M_air = np.vstack((M_upper_air, M_mid_air, M_lower_air))

    def update_bCO2_a(self):
        #Set up shorter variable names for readability
        CO2_a = self.CO2_a
        CO2_w = self.CO2_w
        dt = self.dt_ad
        dx = self.dx_ad
        T = self.T
        V_a = self.V_a
        V_w_mean = self.V_w_mean
        Pe_a = self.Pe_a
        if self.V_a[0]>0:
            core_beg = 0
            core_end = -1
            conc_bnd = 0
            diff_bnd = -1
            diff_bnd2 = -2
        else:
            core_beg = 1
            core_end = self.n_nodes-1
            conc_bnd = -1
            diff_bnd = 0
            diff_bnd2 = 1

        self.bCO2_a[core_beg:core_end] = CO2_a[1:-1]*(1./dt-1./(Pe_a*T*dx**2.)) \
           + CO2_a[0:-2]*((V_a[core_beg:core_end]/V_w_mean)/(4.*dx) + 1./(2.*Pe_a*T*dx**2.)) \
           + CO2_a[2:]* ((-V_a[core_beg:core_end]/V_w_mean)/(4.*dx) + 1./(2.*Pe_a*T*dx**2.)) \
           - self.Lambda_a[core_beg:core_end]*(CO2_a[1:-1] - CO2_w[1:-1]) #Last line here is added to previous C-N solution to include reaction
        self.bCO2_a[conc_bnd] += ((np.sign(V_a[0])*V_a[conc_bnd]/V_w_mean)/(4.*dx) + 1./(2.*Pe_a*T*dx**2.))*self.CO2_a_upstream
        self.bCO2_a[diff_bnd] = (1./dt - (np.sign(V_a[0])*V_a[diff_bnd]/V_w_mean)/(2.*dx))*CO2_a[diff_bnd] \
            + ((np.sign(V_a[0])*V_a[diff_bnd]/V_w_mean)/(2*dx))*CO2_a[diff_bnd2] \
            - self.Lambda_a[diff_bnd]*(CO2_a[diff_bnd] - CO2_w[diff_bnd])#last term gets added to boundary cond.
#        else:
#            self.bCO2_a[1:] = CO2_a[1:-1]*(T-dt/(Pe_a[1:]*dx**2.)) + CO2_a[0:-2]*(np.sign(V_a[1:])*dt/(4.*dx) + dt/(2.*Pe_a[1:]*dx**2.)) \
#                            + CO2_a[2:]*(-np.sign(V_a[1:])*dt/(4.*dx) + dt/(2.*Pe_a[1:]*dx**2.))\
#                            - dt*Lambda_a*(CO2_a[1:-1] - C_w[1:-1]) #Last line here is added to previous C-N solution to include reaction
#            self.bCO2_a[-1] += dt*(1./(4.*dx) + 1./(2.*Pe_a[-1]*dx**2.))*CO2_a_upstream
#            self.bCO2_a[0] = (T-dt/(2.*dx))*CO2_a[n,0] + (dt/(2*dx))*CO2_a[n,1] - dt*Lambda_a*(CO2_a[n,0] - C_w[n,0])#last term gets added to boundary cond.


    def update_bCO2_w(self):
        #Set up shorter variable names for readability
        CO2_a = self.CO2_a
        CO2_w = self.CO2_w
        dt = self.dt_ad
        dx = self.dx_ad
        #T = self.T
        V_w = self.V_w
        V_w_mean = self.V_w_mean
        Pe_w = self.Pe_w
        if self.V_w[0]>0:
            core_beg = 0
            core_end = -1
            conc_bnd = 0
            diff_bnd = -1
            diff_bnd2 = -2
        else:
            core_beg = 1
            core_end = self.n_nodes-1
            conc_bnd = -1
            diff_bnd = 0
            diff_bnd2 = 1

        self.bCO2_w[core_beg:core_end] = CO2_w[1:-1]*(1./dt-1./(Pe_w*dx**2.)) \
           + CO2_w[0:-2]*((V_w[core_beg:core_end]/V_w_mean)/(4.*dx) + 1./(2.*Pe_w*dx**2.)) \
           + CO2_w[2:]* ((-V_w[core_beg:core_end]/V_w_mean)/(4.*dx) + 1./(2.*Pe_w*dx**2.)) \
           + self.Lambda_w[core_beg:core_end]*(CO2_a[1:-1] - CO2_w[1:-1])  \
           - self.R_CO2[core_beg:core_end]
        self.bCO2_w[conc_bnd] += ((np.sign(V_w[0])*V_w[conc_bnd]/V_w_mean)/(4.*dx) + 1./(2.*Pe_w*dx**2.))*self.CO2_w_upstream
        self.bCO2_w[diff_bnd] = (1./dt - np.sign(V_w[0])*(V_w[diff_bnd]/V_w_mean)/(2.*dx))*CO2_w[diff_bnd] \
            + np.sign(V_w[0])*((V_w[diff_bnd]/V_w_mean)/(2*dx))*CO2_w[diff_bnd2] \
            + self.Lambda_w[diff_bnd]*(CO2_a[diff_bnd] - CO2_w[diff_bnd])\
            - self.R_CO2[diff_bnd]

    def update_bCa(self):
        #Set up shorter variable names for readability
        Ca = self.Ca
        dt = self.dt_ad
        dx = self.dx_ad
        V_w = self.V_w
        V_w_mean = self.V_w_mean
        Pe_w = self.Pe_w
        if self.V_w[0]>0:
            core_beg = 0
            core_end = -1
            conc_bnd = 0
            diff_bnd = -1
            diff_bnd2 = -2
        else:
            core_beg = 1
            core_end = self.n_nodes-1
            conc_bnd = -1
            diff_bnd = 0
            diff_bnd2 = 1

        self.bCa[core_beg:core_end] = Ca[1:-1]*(1./dt-1./(Pe_w*dx**2.)) \
           + Ca[0:-2]*((V_w[core_beg:core_end]/V_w_mean)/(4.*dx) + 1./(2.*Pe_w*dx**2.)) \
           + Ca[2:]* ((-V_w[core_beg:core_end]/V_w_mean)/(4.*dx) + 1./(2.*Pe_w*dx**2.)) \
           + self.R_Ca[core_beg:core_end]
        self.bCa[conc_bnd] += (np.sign(V_w[0])*(V_w[conc_bnd]/V_w_mean)/(4.*dx) + 1./(2.*Pe_w*dx**2.))*self.Ca_upstream
        self.bCa[diff_bnd] = (1./dt - (np.sign(V_w[0])*V_w[diff_bnd]/V_w_mean)/(2.*dx))*Ca[diff_bnd] \
            + (np.sign(V_w[0])*(V_w[diff_bnd]/V_w_mean)/(2*dx))*Ca[diff_bnd2] \
            + self.R_Ca[diff_bnd]

    """ old b array code
                    bC_Ca[0:-1] = C_Ca[n,1:-1]*(1.-dt/(Pe_w[0:-1]*dx**2.)) + C_Ca[n,0:-2]*(dt/(4.*dx) + dt/(2.*Pe_w[0:-1]*dx**2.))\
                                        + C_Ca[n,2:]*(-dt/(4.*dx) + dt/(2.*Pe_w[0:-1]*dx**2.))\
                                        + dt*R_Ca[0:-1]
                    bC_Ca[0] += dt*(1./(4.*dx) + 1./(2.*Pe_w[0]*dx**2.))*C_Ca_upstream
                    bC_Ca[-1] = (1.-dt/(2.*dx))*C_Ca[n,-1] + (dt/(2*dx))*C_Ca[n,-2] + dt*R_Ca[-1]
    """


    def update_dimnless_params(self):
        L_tot = self.L
        self.V_w_mean = V_w_mean = np.abs(self.V_w.mean())
        self.V_a_mean = V_a_mean = np.abs(self.V_a.mean())
        #Time conversion parameter (between air and water)
        if V_a_mean>0:
            self.T = np.abs(V_w_mean/V_a_mean)
        self.tau = L_tot.sum()/V_w_mean #Total flowthrough time in secs
        self.Pe_a = L_tot*V_a_mean/self.D_a
        self.Pe_w = L_tot*V_w_mean/self.D_w
        openchan = self.A_a > 0
        self.Lambda_w[openchan] = \
            (self.gas_transf_vel*L_tot/V_w_mean)*self.W[openchan]/self.A_w[openchan]
        self.Lambda_a[openchan] = \
            (self.gas_transf_vel*L_tot/V_w_mean)*self.W[openchan]/self.A_a[openchan]
        self.Lambda_w[~openchan] = 0.
        self.Lambda_a[~openchan] = 0.
        self.dt_ad = self.dt_ad_dim/self.tau
        self.dx_ad = self.L_arr[0]/self.L #assume const grid

    def initialize_conc_arrays(self):
        #Create concentration arrays
        self.CO2_a = np.zeros(self.n_nodes)
        self.CO2_w = np.zeros(self.n_nodes)
        self.Ca = np.zeros(self.n_nodes)

        #Set upstream boundary concentrations for air
        if self.V_a[0]>0:
            CO2_a_upstream = self.pCO2_outside/self.pCO2_high
        elif self.V_a[0]==0:
            CO2_a_upstream = self.CO2_w_upstream
        else:
            CO2_a_upstream = 0.9

        #Set initial conditions for all species
        self.CO2_a[:] = CO2_a_upstream
        self.CO2_w[:] = self.CO2_w_upstream
        self.Ca[:] = self.Ca_upstream
        self.CO2_a_upstream = CO2_a_upstream

    def calc_steady_adv_disp_reaction(self):
        """
            D_H_w, D_H_a,
                            ntimes=1000, endtime=2., nx=1000, xmax=1,
                            L=1000, D_w=30., D_a=30, Q_a=1., Q_w=0.1,
                            pCO2_high=5000*1e-6, pCO2_outside=500*1e-6,
                            T_C=10, Lambda_w=0.5, tol=1e-5, rel_tol=1e-5,
                            C_w_upstream=1., C_Ca_upstream=0.5):
        """
        abs_tol = self.abs_tol
        rel_tol = self.rel_tol
        self.update_dimnless_params()
        if self.V_a_mean>0:
            self.update_adv_disp_M_air()
        self.update_adv_disp_M_water()
        self.initialize_conc_arrays()

        mm_yr_to_mols_sec = 100.*rho_limestone/g_mol_CaCO3/secs_per_year/100./(self.D_H_w/2.)

        air_water_converged = False
        air_water_Ca_converged = False
        Ca_initialized = False
        self.n_iter=0
        while not air_water_Ca_converged:
            self.n_iter+=1
            print('Timestep=',self.n_iter)
            if air_water_converged:
                if not Ca_initialized:
                    #For first iteration we will calculate Ca values based on steady state
                    #with no interaction between dissolution and CO2 drawdown.
                    F = np.zeros(self.n_nodes)
                    for i in np.arange(self.n_nodes-1):
                        Ca_in = self.Ca[i]
                        #print(i, Ca_in)
                        Ca_in_mol_L = self.Ca_eq_0*Ca_in
                        pCO2_in_atm = self.pCO2_high*self.CO2_w[i]
                        sol_in = solutionFromCaPCO2(Ca_in_mol_L, pCO2_in_atm, T_C=self.T_cave)
                        F[i+1] = palmerFromSolution(sol_in, PCO2=pCO2_in_atm)
                        R = F[i+1]*mm_yr_to_mols_sec[i]
                        dC_mol = R*self.dx_ad*self.L/self.V_w[i]
                        Ca_out_mol = Ca_in_mol_L + dC_mol
                        self.Ca[i+1] = Ca_out_mol/self.Ca_eq_0
                    Ca_initialized = True
            #        print(assf)
                else:
                    #Calculate calcite dissolution rates in
                    Ca_mol_L = self.Ca_eq_0*self.Ca
                    pCO2_atm = self.pCO2_high*self.CO2_w
                    sols = solutionFromCaPCO2(Ca_mol_L, pCO2_atm, T_C=self.T_cave)
                    F = palmerFromSolution(sols, PCO2=pCO2_atm)
            #print('done calculating palmer rates')
                self.F = F
                #Convert to mols/sec for conduit segment
                self.R = F[1:]*mm_yr_to_mols_sec
                #Convert to dimensionless Ca
                self.R_Ca = self.R*self.tau/self.Ca_eq_0
                #Convert to dimensionless pCO2
                self.R_CO2 = self.R*self.tau/self.K_H/self.pCO2_high
            else:
                self.R_Ca=self.R_CO2=np.zeros(self.n_nodes-1)

#            if air_water_converged:
                #Calculate b matrix for C_Ca
            if self.V_a_mean>0:
                self.update_bCO2_a()
            self.update_bCO2_w()
            if self.CO2_w.max()>2:
                print(asf)
            CO2_w_new = solve_banded((1,1), self.M_water, self.bCO2_w)
            if self.V_a_mean>0:
                CO2_a_new = solve_banded((1,1), self.M_air, self.bCO2_a)
            else:
                CO2_a_new = CO2_w_new
            if air_water_converged:
                self.update_bCa()
                Ca_new = solve_banded((1,1), self.M_water, self.bCa)

            abs_tol_CO2_w = max(abs(CO2_w_new - self.CO2_w[:-1]))
            rel_tol_CO2_w = max(abs((CO2_w_new - self.CO2_w[:-1])/self.CO2_w[:-1]) )
            if self.V_a[0]>0:
                abs_tol_CO2_a = max(abs(CO2_a_new - self.CO2_a[1:]))
                rel_tol_CO2_a = max(abs((CO2_a_new - self.CO2_a[1:])/self.CO2_a[1:]) )
            else:
                abs_tol_CO2_a = max(abs(CO2_a_new - self.CO2_a[:-1]))
                rel_tol_CO2_a = max(abs((CO2_a_new - self.CO2_a[:-1])/self.CO2_a[:-1]) )
#            print('n=',n)
            print('rel tol CO2_w=', rel_tol_CO2_w, '  abs_tol_CO2_w=',abs_tol_CO2_w)
            print('rel tol CO2_a=', rel_tol_CO2_a, '  abs_tol_CO2_a=',abs_tol_CO2_a)

            #Overwrite old solutions
            if self.V_a[0]>0:
                self.CO2_a[1:] = CO2_a_new
            else:
                self.CO2_a[:-1] = CO2_a_new
            self.CO2_w[:-1] = CO2_w_new

            if not air_water_converged:
                if (abs_tol_CO2_a < abs_tol and abs_tol_CO2_w < abs_tol and rel_tol_CO2_w<rel_tol and rel_tol_CO2_a<rel_tol):
                    air_water_converged = True
                    print("Air-water solution converged, beginning dissolution calculations: n=",self.n_iter)
            else:
                abs_tol_Ca = max(abs(Ca_new - self.Ca[:-1]))
                rel_tol_Ca = max(abs((Ca_new - self.Ca[:-1])/self.Ca[:-1]) )
                print('rel tol Ca=', rel_tol_Ca, '  abs_tol_Ca=',abs_tol_Ca)
                self.Ca[:-1] = Ca_new
                if (abs_tol_CO2_a < abs_tol and abs_tol_CO2_w < abs_tol and abs_tol_Ca<abs_tol and rel_tol_CO2_w<rel_tol and rel_tol_CO2_a<rel_tol and rel_tol_Ca<rel_tol):
                    print("Full solution converged: n=",self.n_iter)
                    air_water_Ca_converged = True
