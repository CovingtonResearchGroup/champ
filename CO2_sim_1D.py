import numpy as np
from scipy.linalg import solve_banded
from olm.calcite import concCaEqFromPCO2, createPalmerInterpolationFunctions, palmerRate, calc_K_H,\
                        solutionFromCaPCO2, palmerFromSolution
from olm.general import CtoK

from crossSection import CrossSection
from ShapeGen import genCirc, genEll
from numpy.random import rand,seed

#Constants
g=9.8#m/s^2

class CO2_1D:

    def __init__(self, x_arr, z_arr, D_w=30., D_a=30, Q_w=0.1,
    pCO2_high=5000*1e-6, pCO2_outside=500*1e-6, f=0.1,
    T_cave=10, T_outside=20., Lambda_w=0.5, abs_tol=1e-5, rel_tol=1e-5,
    CO2_w_upstream=1., Ca_upstream=0.5, h0=0., rho_air_cave = 1.225, dH=50.,
    init_shape = 'circle', init_radii = 0.5, offsets = 0., xc_n=1000):
        self.n_nodes = x_arr.size
        self.L = x_arr.max() - x_arr.min()
        self.x_arr = x_arr
        self.z_arr = z_arr
        self.D_w = D_w
        self.D_a = D_a
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
        self.Lambda_w = Lambda_w
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.CO2_w_upstream = CO2_w_upstream
        self.Ca_upstream = Ca_upstream

        self.V_w = np.zeros(self.n_nodes - 1)
        self.V_a = np.zeros(self.n_nodes - 1)
        self.A_w = np.zeros(self.n_nodes - 1)
        self.A_a = np.zeros(self.n_nodes - 1)
        self.P_w = np.zeros(self.n_nodes - 1)
        self.P_a = np.zeros(self.n_nodes - 1)
        self.D_H_w = np.zeros(self.n_nodes - 1)
        self.D_H_a = np.zeros(self.n_nodes - 1)
        self.W = np.zeros(self.n_nodes - 1)

        self.slopes = (z_arr[1:] - z_arr[:-1])/(x_arr[1:] - x_arr[:-1])
        self.L_arr = x_arr[1:]- x_arr[:-1]
        self.fd_mids = np.zeros(self.n_nodes-1)
        self.offsets = np.ones(self.n_nodes-1) * offsets
        self.h = np.zeros(self.n_nodes)
        self.h0 = h0
        self.f=f
        self.flow_type = np.zeros(self.n_nodes-1,dtype=object)

        #Initialize cross-sections
        self.xcs = []
        self.maxdepths = np.zeros(self.n_nodes-1)
        self.radii = init_radii*np.ones(self.n_nodes-1)
        for i in np.arange(self.n_nodes-1):
            x, y = genCirc(self.radii[i],n=xc_n)
            y = y + self.offsets[i]
            this_xc = CrossSection(x,y)
            self.xcs.append(this_xc)
            self.maxdepths[i] = this_xc.ymax - this_xc.ymin

    def calc_flow_depths(self):
        # Loop through cross-sections and solve for flow depths,
        # starting at downstream end
        for i, xc in enumerate(self.xcs):
            #Try calculating flow depth
            norm_fd = xc.calcNormalFlowDepth(self.Q_w,self.slopes[i],f=self.f)
            backflooded= (self.h[i]-self.z_arr[i+1])>self.maxdepths[i]
            if norm_fd==-1:
                over_normal_capacity=True
            else:
                over_normal_capacity=False

            if over_normal_capacity or backflooded:
                self.flow_type[i] = 'full'
                if i==0:
                    #if downstream boundary set head to top of pipe
                    self.h[0]=self.maxdepths[0]
                #We have a full pipe, calculate head gradient instead
                delh = xc.calcPipeFullHeadGrad(self.Q_w,self.slopes[i],f=self.f)
                self.h[i+1] = self.h[i] + delh * self.L_arr[i]
                self.fd_mids[i] = xc.ymax - xc.ymin
            else:
                crit_fd = xc.calcCritFlowDepth(self.Q_w)
                y_star = min([crit_fd,norm_fd])
                y_out = self.h[i] - self.z_arr[i]
                downstream_critical = y_star>y_out and y_star>0# and i>0
                partial_backflood = norm_fd < self.h[i] - self.z_arr[i+1]
                downstream_less_normal = norm_fd>y_out
                if partial_backflood: #upstream node is flooded above normal depth
                    self.flow_type[i] = 'pbflood'
                    y_in = xc.calcUpstreamHead(self.Q_w,self.slopes[i],y_out,self.L_arr[i],f=self.f)
                    self.h[i+1] = self.z_arr[i+1] + y_in
                    self.fd_mids[i] = (y_out + y_in)/2.
                elif downstream_critical:
                    self.flow_type[i] = 'dwnscrit'
                    #Use minimum of critical or normal depth for downstream y
                    y_in = xc.calcUpstreamHead(self.Q_w,self.slopes[i],y_star,self.L_arr[i],f=self.f)
                    self.fd_mids[i] = (y_in + y_star)/2.
                    self.h[i+1] = self.z_arr[i+1] + y_in
                    if i==0:
                        self.h[0]=y_star
                elif downstream_less_normal:
                    self.flow_type[i] = 'dwnslessnorm'
                    y_in = xc.calcUpstreamHead(self.Q_w,self.slopes[i],y_out,self.L_arr[i],f=self.f)
                    self.h[i+1] = self.z_arr[i+1] + y_in
                    self.fd_mids[i] = (y_out+y_in)/2.
                else:
                    self.flow_type[i] = 'norm'
                    if i==0:
                        self.h[i] = norm_fd + self.z_arr[i]
                    #dz = slopes[i]*(x[i+1] - x[i])
                    self.h[i+1] = self.z_arr[i+1] + norm_fd
                    self.fd_mids[i] = norm_fd
            # Calculate flow areas, wetted perimeters, hydraulic diameters,
            # free surface widths, and velocities
            wetidx = (xc.y - xc.ymin) < self.fd_mids[i]
            self.A_w[i] = xc.calcA(wantidx=wetidx)
            print(self.A_w[i])
            self.P_w[i] = xc.calcP(wantidx=wetidx)
            self.V_w[i] = self.Q_w/self.A_w[i]
            self.D_H_w[i] = 4*self.A_w[i]/self.P_w[i]
            if self.flow_type[i] != 'full':
                L,R = xc.findLR(self.fd_mids[i])
                self.W[i] = xc.x[R] - xc.x[L]
            else:
                self.W[i] = 0.

    def calc_air_flow(self):
        dT = self.T_outside - self.T_cave
        dP_tot = self.rho_air_cave*g*self.dH*dT/self.T_outside_K
        R_air = np.zeros(self.n_nodes-1)
        for i, xc in enumerate(self.xcs):
            dryidx = xc.y>self.fd_mids[i]
            self.A_a[i] = xc.calcA(wantidx=dryidx)
            self.P_a = xc.calcP(wantidx=dryidx)
            if self.A_a[i]>0:
                self.D_H_a[i] = 4.*self.A_a[i]/self.P_a
                R_air[i] = self.rho_air_cave*self.f*self.L_arr[i]/(2.*self.D_H_a[i]*self.A_a[i]**2.)
            else:
                R_air[i] = np.inf
        self.Q_a = np.sqrt(abs(dP_tot/R_air.sum()))*np.sign(dP_tot)
        print("Air discharge = ",self.Q_a, ' m^3/s')







    def calc_steady_adv_reaction(D_H_w, D_H_a,
                    ntimes=1000, endtime=2., nx=1000, xmax=1,
                    L=1000, D_w=30., D_a=30, Q_a=1., Q_w=0.1,
                    pCO2_high=5000*1e-6, pCO2_outside=500*1e-6,
                    T_C=10, Lambda_w=0.5, tol=1e-5, rel_tol=1e-5,
                    C_w_upstream=1., C_Ca_upstream=0.5):
        #Define parameters for discretization
        dt = endtime/(ntimes-1)
        dx = xmax/(nx-1.)
        T_K = CtoK(T_C)
        K_H = calc_K_H(T_K) #Henry's law constant mols dissolved per atm
        Ca_eq_0 = concCaEqFromPCO2(pCO2_high, T_C=T_C)
        palmer_interp_funcs = createPalmerInterpolationFunctions(impure=True)
        rho = 2.6#g/cm^3
        g_mol_CaCO3 = 100.09
        secs_per_year =  3.154e7

        #Arrays of diameters, velocities, and Pe
        D_H_w = D_H_w*np.ones(nx-1)
        D_H_a = D_H_a*np.ones(nx-1)
        P_w = D_H_w*np.pi/2.#assumes semi-circular xc
        P_a = D_H_a*np.pi/2.
        V_w = Q_w/(D_H_w/P_w/4.)#-1
        V_a = Q_a/(D_H_a/P_a/4.)#0.1

        #Time conversion parameter (between air and water)
        T = V_w.mean()/np.abs(V_a.mean())
        tau = L/V_w.mean()#Flowthrough time in secs
        Pe_a = L*np.abs(V_a)/D_a
        Pe_w = L*V_w/D_w

        #Reaction/exchange parameters
        #Lambda_w = sim_dict['Lambda_w']
        Lambda_a = Lambda_w*T

        #Construct A matrix
        A_upper_air = dt*(np.sign(V_a)*1./(4.*dx) - 1./(2.*Pe_a*dx**2.))*np.ones(nx-1)
        A_lower_air = dt*(-np.sign(V_a)*1./(4.*dx) - 1./(2.*Pe_a*dx**2.))*np.ones(nx-1)
        A_mid_air = (T+dt/(Pe_a*dx**2.))*np.ones(nx-1)
        A_upper_water = dt*(1./(4.*dx) - 1./(2.*Pe_w*dx**2.))*np.ones(nx-1)
        A_lower_water = dt*(-1./(4.*dx) - 1./(2.*Pe_w*dx**2.))*np.ones(nx-1)
        A_mid_water = (1.+dt/(Pe_w*dx**2.))*np.ones(nx-1)

        A_upper_air[0] = 0.
        A_lower_air[-1] = 0.
        if V_a[0]>0:
            A_lower_air[-2] = -dt/(2.*dx)
            A_mid_air[-1] = T + dt/(2*dx)
        else:
            A_upper_air[1] = -dt/(2.*dx)
            A_mid_air[0] = T + dt/(2*dx)

        A_upper_water[0] = 0.
        A_lower_water[-1] = 0.
        A_lower_water[-2] = -dt/(2.*dx)
        A_mid_water[-1] = 1. + dt/(2*dx)

        A_air = np.vstack((A_upper_air, A_mid_air, A_lower_air))
        A_water = np.vstack((A_upper_water, A_mid_water, A_lower_water))

        #Create two concentration arrays
        C_a = np.zeros([ntimes,nx])
        C_w = np.zeros([ntimes,nx])
        C_Ca = np.zeros([ntimes,nx])

        #Set upstream boundary concentrations
        C_a_upstream = pCO2_outside/pCO2_high

        #Set initial conditions for both species
        C_a[0,:] = C_a_upstream
        C_w[0,:] = C_w_upstream
        C_Ca[0,:] = C_Ca_upstream

        if V_a[0]>0:
            C_a[:,0] = C_a_upstream
        else:
            C_a[:,-1] = C_a_upstream

        C_w[:,0] = C_w_upstream
        C_Ca[:,0] = C_Ca_upstream

        #Create b arrays for each concentration variable
        bC_a = np.zeros(nx-1)
        bC_w = np.zeros(nx-1)
        bC_Ca = np.zeros(nx-1)

        air_water_converged = False
        air_water_Ca_converged = False
        Ca_initialized = False
        for n in np.arange(ntimes-1):
            #print('Timestep=',n)
            mm_yr_to_mols_sec = 100.*rho/g_mol_CaCO3/secs_per_year/100./(D_H_w/2.)
            if air_water_converged:
                if not Ca_initialized:
                    #For first iteration we will calculate Ca values based on steady state
                    #with no interaction between dissolution and CO2 drawdown.
                    F = np.zeros(nx)
                    for i in np.arange(nx-1):
                        Ca_in = C_Ca[n,i]
                        print(i, Ca_in)
                        Ca_in_mol_L = Ca_eq_0*C_Ca[n,i]
                        pCO2_in_atm = pCO2_high*C_w[n,i]
                        sol_in = solutionFromCaPCO2(Ca_in_mol_L, pCO2_in_atm, T_C=T_C)
                        F[i+1] = palmerFromSolution(sol_in, PCO2=pCO2_in_atm)
                        R = F[i+1]*mm_yr_to_mols_sec[i]
                        dC_mol = R*dx*L/V_w[i]
                        Ca_out_mol = Ca_in_mol_L + dC_mol
                        C_Ca[n,i+1] = Ca_out_mol/Ca_eq_0
                    Ca_initialized = True
                else:
                    #Calculate calcite dissolution rates in
                    Ca_mol_L = Ca_eq_0*C_Ca[n,:]
                    pCO2_atm = pCO2_high*C_w[n,:]
                    sols = solutionFromCaPCO2(Ca_mol_L, pCO2_atm, T_C=T_C)
                    F = palmerFromSolution(sols, PCO2=pCO2_atm)
            #print('done calculating palmer rates')
                #Convert to mols/sec for conduit segment
                R = F[1:]*mm_yr_to_mols_sec
                #Convert to dimensionless Ca
                R_Ca = R*tau/Ca_eq_0
                #Convert to dimensionless pCO2
                R_CO2 = R*tau/K_H/pCO2_high
            else:
                R_Ca=R_CO2=np.zeros(nx-1)

            #Calculate b matrix for C_a
            if V_a[0]>0:
                bC_a[0:-1] = C_a[n,1:-1]*(T-dt/(Pe_a[0:-1]*dx**2.)) + C_a[n,0:-2]*(np.sign(V_a[0:-1])*dt/(4.*dx) + dt/(2.*Pe_a[0:-1]*dx**2.)) \
                                + C_a[n,2:]*(-np.sign(V_a[0:-1])*dt/(4.*dx) + dt/(2.*Pe_a[0:-1]*dx**2.))\
                                - dt*Lambda_a*(C_a[n,1:-1] - C_w[n,1:-1]) #Last line here is added to previous C-N solution to include reaction
                bC_a[0] += dt*(1./(4.*dx) + 1./(2.*Pe_a[0]*dx**2.))*C_a_upstream
                bC_a[-1] = (T-dt/(2.*dx))*C_a[n,-1] + (dt/(2*dx))*C_a[n,-2] - dt*Lambda_a*(C_a[n,-1] - C_w[n,-1])#last term gets added to boundary cond.
            else:
                bC_a[1:] = C_a[n,1:-1]*(T-dt/(Pe_a[1:]*dx**2.)) + C_a[n,0:-2]*(np.sign(V_a[1:])*dt/(4.*dx) + dt/(2.*Pe_a[1:]*dx**2.)) \
                                + C_a[n,2:]*(-np.sign(V_a[1:])*dt/(4.*dx) + dt/(2.*Pe_a[1:]*dx**2.))\
                                - dt*Lambda_a*(C_a[n,1:-1] - C_w[n,1:-1]) #Last line here is added to previous C-N solution to include reaction
                bC_a[-1] += dt*(1./(4.*dx) + 1./(2.*Pe_a[-1]*dx**2.))*C_a_upstream
                bC_a[0] = (T-dt/(2.*dx))*C_a[n,0] + (dt/(2*dx))*C_a[n,1] - dt*Lambda_a*(C_a[n,0] - C_w[n,0])#last term gets added to boundary cond.

            #Calculate b matrix for C_w
            bC_w[0:-1] = C_w[n,1:-1]*(1.-dt/(Pe_w[0:-1]*dx**2.)) + C_w[n,0:-2]*(dt/(4.*dx) + dt/(2.*Pe_w[0:-1]*dx**2.))\
                                + C_w[n,2:]*(-dt/(4.*dx) + dt/(2.*Pe_w[0:-1]*dx**2.))\
                                + dt*Lambda_w*(C_a[n,1:-1] - C_w[n,1:-1])\
                                - dt*R_CO2[0:-1]
            bC_w[0] += dt*(1./(4.*dx) + 1./(2.*Pe_w[0]*dx**2.))*C_w_upstream
            bC_w[-1] = (1.-dt/(2.*dx))*C_w[n,-1] + (dt/(2*dx))*C_w[n,-2] + dt*Lambda_w*(C_a[n,-1] - C_w[n,-1]) - dt*R_CO2[-1]
            if air_water_converged:
                #Calculate b matrix for C_Ca
                bC_Ca[0:-1] = C_Ca[n,1:-1]*(1.-dt/(Pe_w[0:-1]*dx**2.)) + C_Ca[n,0:-2]*(dt/(4.*dx) + dt/(2.*Pe_w[0:-1]*dx**2.))\
                                    + C_Ca[n,2:]*(-dt/(4.*dx) + dt/(2.*Pe_w[0:-1]*dx**2.))\
                                    + dt*R_Ca[0:-1]
                bC_Ca[0] += dt*(1./(4.*dx) + 1./(2.*Pe_w[0]*dx**2.))*C_Ca_upstream
                bC_Ca[-1] = (1.-dt/(2.*dx))*C_Ca[n,-1] + (dt/(2*dx))*C_Ca[n,-2] + dt*R_Ca[-1]


            #Solve systems of equations
            if V_a[0]>0:
                C_a[n+1,1:] = solve_banded((1,1), A_air, bC_a)
            else:
                C_a[n+1,:-1] = solve_banded((1,1), A_air, bC_a)
            C_w[n+1,1:] = solve_banded((1,1), A_water, bC_w)
            if air_water_converged:
                C_Ca[n+1,1:] = solve_banded((1,1), A_water, bC_Ca)

            abs_tol_C_w = max(abs(C_w[n+1] - C_w[n]))
            abs_tol_C_a = max(abs(C_a[n+1] - C_a[n]))
            rel_tol_C_w = max(abs((C_w[n+1] - C_w[n])/C_w[n]) )
            rel_tol_C_a = max(abs((C_a[n+1] - C_a[n])/C_a[n]) )
            print('n=',n)
            print('rel tol C_w=', rel_tol_C_w, '  abs_tol_C_w=',abs_tol_C_w)
            print('rel tol C_a=', rel_tol_C_a, '  abs_tol_C_a=',abs_tol_C_a)

            if not air_water_converged:
                if (abs_tol_C_a < tol and abs_tol_C_w < tol and rel_tol_C_w<rel_tol and rel_tol_C_a<rel_tol):
                    air_water_converged = True
                    print("Air-water solution converged, beginning dissolution calculations: n=",n)
            else:
                abs_tol_C_Ca = max(abs(C_Ca[n+1] - C_Ca[n]))
                rel_tol_C_Ca = max(abs((C_Ca[n+1] - C_Ca[n])/C_Ca[n]) )
                print('rel tol C_Ca=', rel_tol_C_Ca, '  abs_tol_C_Ca=',abs_tol_C_Ca)

                if (abs_tol_C_a < tol and abs_tol_C_w < tol and abs_tol_C_Ca<tol and rel_tol_C_w<rel_tol and rel_tol_C_a<rel_tol and rel_tol_C_Ca<rel_tol):
                    print("Full solution converged: n=",n)
                    air_water_Ca_converged = True
                    return C_w, C_a, C_Ca, R
        return C_w, C_a, C_Ca, R
