import numpy as np
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
Sc_Ca = nu/D_Ca
g_mol_CaCO3 = 100.09
L_per_m3 = 1000.
secs_per_year =  3.154e7
secs_per_hour = 60.*60.
secs_per_day = secs_per_hour*24.
cm_m = 100.

###
## gas trasfer vel, typical values ~10 cm/hr for small streams (Wanningkhof 1990)
####

class CO2_1D:

    def __init__(self, x_arr, z_arr, Q_w=0.1,
    pCO2_high=5000*1e-6, pCO2_outside=500*1e-6, f=0.1,
    T_cave=10, T_outside=20., gas_transf_vel=0.1/secs_per_hour,
    abs_tol=1e-5, rel_tol=1e-5, CO2_err_rel_tol=0.001,
    CO2_w_upstream=1., CO2_a_upstream = 0.9, Ca_upstream=0.5, h0=0., rho_air_cave = 1.225, dH=50.,
    init_shape = 'circle', init_radii = 0.5, init_offsets = 0., xc_n=1000,
    impure=True,reduction_factor=0.01, dt_erode=1.,
    downstream_bnd_type='normal', trim=True, variable_gas_transf=False,
    subdivide_factor = 0.2):
        self.n_nodes = x_arr.size
        self.L = x_arr.max() - x_arr.min()
        self.x_arr = x_arr
        self.z_arr = z_arr#z is zero of xc coords
        self.L_arr = x_arr[1:]- x_arr[:-1]
        #Calculate stable timestep
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
        self.gas_transf_vel = np.ones(self.n_nodes-1)*gas_transf_vel
        self.variable_gas_transf = variable_gas_transf
        self.subdivide_factor = subdivide_factor
        #self.n_subdivide = n_subdivide

        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.CO2_err_rel_tol = CO2_err_rel_tol
        self.CO2_w_upstream = CO2_w_upstream
        self.CO2_a_upstream = CO2_a_upstream
        self.Ca_upstream = Ca_upstream
        self.reduction_factor = reduction_factor
        self.dt_erode = dt_erode
        self.xc_n = xc_n
        self.downstream_bnd_type = downstream_bnd_type
        self.trim = trim

        self.V_w = np.zeros(self.n_nodes - 1)
        self.V_a = np.zeros(self.n_nodes - 1)
        self.A_w = np.zeros(self.n_nodes - 1)
        self.A_a = np.zeros(self.n_nodes - 1)
        self.P_w = np.zeros(self.n_nodes - 1)
        self.P_a = np.zeros(self.n_nodes - 1)
        self.D_H_w = np.zeros(self.n_nodes - 1)
        self.D_H_a = np.zeros(self.n_nodes - 1)
        self.W = np.zeros(self.n_nodes - 1)

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

        #Create concentration arrays
        self.CO2_a = np.zeros(self.n_nodes)
        self.CO2_w = np.zeros(self.n_nodes)
        self.Ca = np.zeros(self.n_nodes)


    def run_one_step(self, T_outside_arr = []):
        self.calc_flow_depths()
        if len(T_outside_arr) == 0:
            self.calc_air_flow()
            self.calc_steady_state_transport()
            self.erode_xcs()
        else:
            dt_frac = 1./len(T_outside_arr)
            cum_dz = np.zeros(self.n_nodes-1)
            avg_CO2_w = np.zeros(self.n_nodes)
            avg_CO2_a = np.zeros(self.n_nodes)
            avg_Ca = np.zeros(self.n_nodes)
            for this_T in T_outside_arr:
                self.set_T_outside(this_T)
                self.calc_air_flow()
                self.calc_steady_state_transport()
                self.erode_xcs(dt_frac=dt_frac)
                cum_dz += self.dz
                avg_CO2_w += dt_frac*self.CO2_w
                avg_CO2_a += dt_frac*self.CO2_a
                avg_Ca += dt_frac*self.Ca
            self.dz = cum_dz
            self.CO2_w = avg_CO2_w
            self.CO2_a = avg_CO2_a
            self.Ca = avg_Ca
            print('dz_cum =',self.dz)

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
                    if y_in>0 and (y_out + y_in)/2. < xc.ymax - xc.ymin:# or could use fraction of y_in
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


    def set_concentration_bnd_conditions(self):
        #Determine upstream boundary concentration for air
        if self.V_a[0]>0:
            CO2_a_boundary = self.pCO2_outside/self.pCO2_high
            air_upstream_idx = 0
        elif self.V_a[0]==0:
            CO2_a_boundary = self.CO2_w_upstream
            air_upstream_idx = -1
        else:
            CO2_a_boundary = self.CO2_a_upstream
            air_upstream_idx = -1

        #Asign upstream boundary conditions into concentration arrays
        self.CO2_a[air_upstream_idx] = CO2_a_boundary
        self.CO2_w[-1] = self.CO2_w_upstream
        self.Ca[-1] = self.Ca_upstream


    def calc_steady_state_transport(self, palmer=False):
        self.set_concentration_bnd_conditions()

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
                rel_CO2_err = np.abs(self.CO2_a[0] - self.pCO2_outside/self.pCO2_high)/(self.pCO2_outside/self.pCO2_high)
                if not hasattr(self, 'CO2_err_rel_tol'):
                    self.CO2_err_rel_tol = 0.001
                if rel_CO2_err > self.CO2_err_rel_tol or True in np.isnan(self.CO2_a):
                    print("Linear shooting method failed...residual=",self.CO2_a[0] - self.pCO2_outside/self.pCO2_high)
                    CO2_a_upstream_brent = brentq(self.downstream_CO2_residual, self.pCO2_outside/self.pCO2_high, 1., rtol=self.CO2_err_rel_tol)
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
            if self.variable_gas_transf == True:
                self.calc_gas_transf_vel_from_eD()
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
                eps = 5*nu*Sc_Ca**(-1./3.)/np.sqrt(T_b/rho_w)
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
                else:
                    #For zero airflow, CO2_a goes to CO2_w
                    dCO2_a = 0.#(this_CO2_w - this_CO2_a)
                    this_CO2_a = this_CO2_w
                    self.CO2_a[i] = self.CO2_w[i]
            else:
                dCO2_a = 0.
                this_CO2_a = this_CO2_w
                self.CO2_a[i] = self.CO2_w[i]
            dCO2_w_exc = self.L_arr[i-1]*K_w[i-1]/self.V_w[i-1]*(this_CO2_w - this_CO2_a)
            #Check whether air or water CO2 changes too much
            if np.abs(dCO2_a) > self.subdivide_factor*np.abs(this_CO2_w - this_CO2_a)\
                or np.abs(dCO2_w_exc) > self.subdivide_factor*np.abs(this_CO2_w - this_CO2_a):
                Q_f = (-1./self.Q_w + 1./self.Q_a)
                lambda_co2 = 1./(self.gas_transf_vel[i-1]*self.W[i-1]*Q_f)
                CO2_w_out = this_CO2_w + ( (this_CO2_w - this_CO2_a)/(-self.Q_w*Q_f))*(np.exp(self.L_arr[i-1]/lambda_co2)-1. )
                CO2_a_out = CO2_w_out - (this_CO2_w - this_CO2_a)*np.exp(self.L_arr[i-1]/lambda_co2)
                dCO2_w_exc = CO2_w_out - this_CO2_w
                dCO2_a = CO2_a_out - this_CO2_a
            dCO2_w = dCO2_w_exc - R_CO2/self.Q_w/L_per_m3
            dCa = R/self.Q_w/L_per_m3
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

    def erode_xcs(self, dt_frac = 1.):
        F_to_m_yr = g_mol_CaCO3*secs_per_year/rho_limestone/cm_m**3
        old_ymins = self.ymins.copy()
        for i,xc in enumerate(self.xcs):
            xc.dr = F_to_m_yr*xc.F_xc*self.dt_erode*dt_frac
            #print('i=',i,'  max dr=', xc.dr.max(), '  max F_xc=',xc.F_xc.max())
            #xc.dr = savgol_filter(xc.dr,15,3,mode='wrap')
            xc.erode(xc.dr, trim=self.trim)
            self.ymins[i]= xc.ymin
        #Adjust slopes
        dz = self.ymins - old_ymins
        self.dz = dz
        print('dt_frac=',dt_frac)
        print('dz=',dz)
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

    def calc_gas_transf_vel_from_eD(self):
        #Relationship from Ulseth et al. (2019), Nat Geosci.
        eD = g*self.slopes*np.abs(self.V_w)
        k_600_m_d = np.exp(3.10 + 0.35*np.log(eD))
        if eD.max()>0.02:
            k_600_m_d[eD>0.02] = np.exp(6.43 + 1.18*np.log(eD[eD>0.02]))
        k_600 = k_600_m_d/secs_per_day
        Sc_CO2 = self.calc_Sc_CO2()
        k_CO2 = k_600*(600/Sc_CO2)**0.5
        self.gas_transf_vel = k_CO2

    def calc_Sc_CO2(self):
        A = 1742
        B= -91.24
        C=2.208
        D=-0.0219
        T = self.T_cave
        Sc_CO2 = A + B*T + C*T**2 + D*T**3
        return Sc_CO2
