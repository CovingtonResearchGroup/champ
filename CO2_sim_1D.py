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
R_da = 287.058 #Specific gas constant for dry air, J/(kg*K)
R_wv = 461.495 #Specific gas constant for water vapor, J/(kg*K)
p_atm = 101325. # 1 atm in Pa
D_Ca = 10**-9#m^2/s
nu = 1.3e-6#m^2/s at 10 C
Sc_Ca = nu/D_Ca
g_mol_CaCO3 = 100.09
L_per_m3 = 1000.
secs_per_year =  3.154e7
secs_per_hour = 60.*60.
secs_per_day = secs_per_hour*24.
cm_m = 100.
cm_per_mm = 0.1
cm2_per_m2 = 100.*100.

###
## gas trasfer vel, typical values ~10 cm/hr for small streams (Wanningkhof 1990)
####

class CO2_1D:
    """Simulation object for a coupled waterflow, airflow, CO2 exchange, and
    cave cross-section evolution algorithm

    The simulation object implements an algorithm for simulating the evolution
    of a 1D cave channel with an arbitrary number of defined cross-sections.

    Key functionality includes:

    1. Calculating steady discharge within a mixture of open channel and
    full pipe conditions with arbitrary channel cross-sectional shapes.

    2. Calculating airflow within the air-filled portion of the cave passage.

    3. Calculating CO2 transport and exchange between air and water.

    4. Calculating calcite dissolution rates.

    5. Evolving channel cross-section according to calculated dissolution rates.

    Examples
    --------
    >>> import numpy as np
    >>> from CO2_sim_1D import CO2_1D
    >>> n=5
    >>> x = np.linspace(0, 5000,n) # m
    >>> slope = 0.001
    >>> z = x*slope
    >>> r = 1.*np.ones(n-1) # m
    >>> Q = 1. # m^3/s
    >>> sim = CO2_1D(x, z, init_radii=r, Q_w = Q, T_outside=20., T_cave=10.,
    ...              dt_erode=1., Ca_upstream=0.5, CO2_w_upstream =1.,
    ...              CO2_a_upstream=0.9, pCO2_high=5e-3)
    >>> nsteps=10
    >>> for t in np.arange(nsteps):
    >>>     sim.run_one_step()

    """

    def __init__(self, x_arr, z_arr, Q_w=0.1, f=0.1,
        init_radii = 0.5, init_offsets = 0., xc_n=1000,
        pCO2_high=5000*1e-6, pCO2_outside=500*1e-6,
        CO2_w_upstream=1., CO2_a_upstream = 0.9, Ca_upstream=0.5,
        gas_transf_vel=0.1/secs_per_hour, variable_gas_transf=True,
        T_cave=10, T_outside=20., dH=50.,
        reduction_factor=0.01, dt_erode=1.,impure=True,
        CO2_err_rel_tol=0.001, trim=True, subdivide_factor = 0.2):

        """
        Parameters
        ----------
        x_arr : ndarray
            Array of distances in meters along the channel for the node locations.
        z_arr: ndarray
            Array of elevations in meters for nodes along the channel. Minimum y
            values for each cross-section will be added to these elevations
            during initialization, so that z_arr will represent the channel bottom.
        Q_w : float, optional
            Discharge in the channel (m^3/s). Default is 0.1 m^3/s.
        f : float, optional
            Darcy-Weisbach friction factor (unitless), used in both water flow and air
            flow calculations. Default is 0.1.
        init_radii : float or ndarray, optional
            Initial cross-section radii (meters). If a float then all cross-sections
            will be assigned the same radius. If an array then each element
            represents the radius of a single cross-section (length should be n-1
            where n is the number of nodes). Default is 0.5 m.
        init_offsets : float or ndarray, optional
            These offsets will be added to y-values within initial cross-sections.
            By default, y will be zero at the centroid of the initial cross-section.
            Default value is zero. Should have length of n-1, where n is number of nodes.
        xc_n : int, optional
            Number of points that will define the cave passage shape within a cross-section.
            Default is 1000.
        pCO2_high : float, optional
            pCO2 value, in atm, by which all others are normalized. Normally this will
            be the highest pCO2 value in the simulation, such as the upstream
            water pCO2. Boundary values of pCO2 are defined as fractions of this
            value. Default is 5e-3 atm (5000 ppm).
        pCO2_outside : float, optional
            pCO2 of outside atmosphere in atm. This is used for cave air boundary
            condition when airflow is in winter direction. Default is 5e-4 atm (500 ppm).
        CO2_w_upstream : float, optional
            The pCO2 within the water at the upstream boundary expressed as a fraction
            of pCO2_high. Default value is 1.
        CO2_a_upstream : float, optional
            The pCO2 of the air at the upstream boundary expressed as a fraction
            of pCO2_high. Default value is 0.9.
        Ca_upstream : float, optional
            Default boundary value for the upstream Ca concentration expressed
            as a fraction of saturation at a pCO2 of pCO2_high. Default is 0.5.
        gas_transf_vel : float or ndarray
            Gas transfer velocity  in m/s. If a constant value, then that value
            is used for all cross-sections. If an array of length n-1, where n
            is the number of nodes, then each element sets the gas transfer
            velocity for a single cross-section. This variable only used if
            variable_gas_transf=False. Otherwise, gas transfer velocities are
            set by empirical relationship that uses channel geometry. Default
            value is 10 cm/hr.
        variable_gas_transf : boolean, optional
            Whether gas transfer velocities will be adjusted during the simulation
            according to empirical relationship based on energy dissipation (Ulseth et al., 2019).
            If True, then transfer velocities will be adjusted. If False, then
            transfer velocities will be constant (set by gas_transf_vel).
            Default value is True.
        T_cave : float, optional
            Cave temperature (air and water) in degrees C. Default is 10 C.
        T_outside: float, optional
            Initial outside air temperature in degrees C. Simulations run with
            multiple outside air temperatures use keyword argument within run_one_step().
            Default value is 20 C.
        dH : float, optional
            Elevation difference (m) driving chimney effect airflow. For now this is
            a constant, but might think about setting it in a more physical way.
            Default is 50 m.
        reduction_factor : float, optional
            Factor by which pure transport-limited dissolution rate will be multiplied.
            This is used to reduce rates because the transport-limited equation creates
            unrealistically high rates, even if field evidence suggests that rates are
            to some extent transport-limited. Default value 0.01.
        dt_erode : float, optional
            Erosional time step in years. Default value is 1 year.
        impure : boolean, optional
            If the Palmer dissolution rate equation is used, this determines
            whether to use the equation for impure or pure calcite. Default is True.
        CO2_err_rel_tol : float, optional
            Acceptable relative tolerance for match of boundary value for pCO2
            in air when air and water flow in different directions. This is
            used to determine if linear shooting method was sucessful. If this
            tolerance is not met then Brent method is used with this same
            relative tolerance as the convergence criterium. Default is 0.001.
        trim : boolean, optional
            Whether or not cross-sections should be trimmed as much of the
            cross-section becomes dry. This enables maintenance of a high
            resolution of the wet portion of the cross-section for simulations
            with substantial incision. If this is set to False, long-term
            simulations are likely to become unstable. Default is true.
        subdivide_factor : float, optional
            If the change in pCO2 of air or water within a single segment is
            more than the difference between air and water pCO2 times
            this factor, then use the analytical solution to calculate the
            downstream concentration of CO2 for this segment. This avoids
            too large of a change of CO2 within a single conduit segment, which
            can lead to unstable solutions. Default is 0.2.


        References
        ----------
        Ulseth, A.J., Hall, R.O. Jr, Canada, M.B., Madinger, H.L., Niayfar, A.,
        and T.J. Battin (2019). Distinct air-water gas exchange regimes in low-
        and high-energy streams. Nature Geoscience, 12, 259-263.
        https://doi.org/10.1038/s41561-019-0324-8

        """

        self.n_nodes = x_arr.size
        self.L = x_arr.max() - x_arr.min()
        self.x_arr = x_arr
        self.z_arr = z_arr#z is zero of xc coords
        self.L_arr = x_arr[1:]- x_arr[:-1]

        self.Q_w = Q_w
        self.Q_a = 0.
        self.pCO2_high = pCO2_high
        self.pCO2_outside = pCO2_outside
        self.dH = dH
        self.T_cave = T_cave
        self.T_cave_K = CtoK(T_cave)
        self.T_outside = T_outside
        self.T_outside_K = CtoK(T_outside)
        self.set_rho_air_cave()

        self.gas_transf_vel = np.ones(self.n_nodes-1)*gas_transf_vel
        self.variable_gas_transf = variable_gas_transf
        self.subdivide_factor = subdivide_factor

        self.CO2_err_rel_tol = CO2_err_rel_tol
        self.CO2_w_upstream = CO2_w_upstream
        self.CO2_a_upstream = CO2_a_upstream
        self.Ca_upstream = Ca_upstream
        self.reduction_factor = reduction_factor
        self.dt_erode = dt_erode
        self.xc_n = xc_n
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

        #Create concentration arrays
        self.CO2_a = np.zeros(self.n_nodes)
        self.CO2_w = np.zeros(self.n_nodes)
        self.Ca = np.zeros(self.n_nodes)

        self.h = np.zeros(self.n_nodes)
        self.f=f
        self.flow_type = np.zeros(self.n_nodes-1,dtype=object)

        self.K_H = calc_K_H(self.T_cave_K) #Henry's law constant mols dissolved per atm
        self.Ca_eq_0 = concCaEqFromPCO2(self.pCO2_high, T_C=T_cave)
        self.palmer_interp_funcs = createPalmerInterpolationFunctions(impure=impure)

        #Initialize cross-sections
        self.xcs = []
        self.radii = init_radii*np.ones(self.n_nodes-1)
        ymins = []
        for i in np.arange(self.n_nodes-1):
            x, y = genCirc(self.radii[i],n=xc_n)
            y = y + self.init_offsets[i]
            this_xc = CrossSection(x,y)
            self.xcs.append(this_xc)
            ymins.append(this_xc.ymin)
        self.ymins = np.array(ymins)
        #Reset z to bottom of cross-sections
        self.z_arr[1:] = self.z_arr[1:] + self.ymins
        self.z_arr[0] = self.z_arr[0] + self.ymins[0]
        self.slopes = (self.z_arr[1:] - self.z_arr[:-1])/(self.x_arr[1:] - self.x_arr[:-1])

    def run_one_step(self, T_outside_arr = []):
        """Run one time step of simulation.

        Calculates flow depths, air flow, transport, and erosion for
        a single time step and updates geometry and chemistry.

        Parameters
        ----------
        T_outside_arr : ndarray
            Array of outside air temperatures to use for this timestep.
            Erosion is calculated for steady state transport for
            each of these air temperature values and averaged evenly
            among them. If set to zero length, then current value
            of T_outside is used instead. Default is [] (use current
            value ot T_outside).

        """

        self.calc_flow_depths()
        if len(T_outside_arr) == 0:
            #Use single T_outside value
            self.calc_air_flow()
            self.calc_steady_state_transport()
            self.erode_xcs()
        else:
            #Run erosion for multiple outside air temps
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

    def calc_flow_depths(self):
        """Calculates flow depths and hydraulic head values along channel.

        Notes
        -----
        Starts at downstream end and propagates solution upstream. Flow can
        be full-pipe, backflooded, partially backflooeded (i.e. deeper than
        required for normal flow because of downstream conditions), or normal.
        If full pipe flow occurs in the furthest downstream segment, downstream
        head is set to the elevation of top of the downstream cross-section.
        Otherwise, the downstream head is set assuming that flow is normal
        within the downstream cross-section.

        """
        # Loop through cross-sections and solve for flow depths,
        # starting at downstream end
        for i, xc in enumerate(self.xcs):
            old_fd = self.fd_mids[i]
            if old_fd <=0:
                old_fd = xc.ymax - xc.ymin
            xc.create_A_interp()
            xc.create_P_interp()
            #Try calculating flow depth
            backflooded= (self.h[i]-self.z_arr[i+1]-xc.ymax+xc.ymin)>0
            over_normal_capacity=False
            if not backflooded:
                norm_fd = xc.calcNormalFlowDepth(self.Q_w,self.slopes[i],f=self.f, old_fd=old_fd)
                if (norm_fd<xc.ymax-xc.ymin) and i==0:
                    #Transition downstream head boundary to normal flow depth
                    # If we don't do this, we can get stuck in full-pipe conditions
                    # because of downstream boundary head.
                    self.h[0] = self.z_arr[0] + norm_fd
                if norm_fd==-1:
                    over_normal_capacity=True
            if over_normal_capacity or backflooded:
                self.flow_type[i] = 'full'
                if i==0:
                    #if downstream boundary set head to top of cross-section
                    self.h[0]= self.z_arr[0] + xc.ymax - xc.ymin
                #We have a full pipe, calculate head gradient instead
                delh = xc.calcPipeFullHeadGrad(self.Q_w,f=self.f)
                self.h[i+1] = self.h[i] + delh * self.L_arr[i]
                self.fd_mids[i] = xc.ymax - xc.ymin
            else:
                #crit_fd = xc.calcCritFlowDepth(self.Q_w)
                y_star = norm_fd#min([crit_fd,norm_fd])
                y_out = self.h[i] - self.z_arr[i]
                downstream_critical = y_star>y_out and y_star>0# and i>0
                partial_backflood = norm_fd < self.h[i] - self.z_arr[i+1]
                downstream_less_normal = norm_fd>y_out
                if partial_backflood: #upstream node is flooded above normal depth
                    self.flow_type[i] = 'pbflood'
                    y_in = xc.calcUpstreamHead(self.Q_w,self.slopes[i],y_out,self.L_arr[i],f=self.f)
                    if y_in>0 and (y_out + y_in)/2. < xc.ymax - xc.ymin:# or could use fraction of y_in
                        self.h[i+1] = self.z_arr[i+1] + y_in
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
                        self.h[i] = norm_fd + self.z_arr[i]
                    #dz = slopes[i]*(x[i+1] - x[i])
                    self.h[i+1] = self.z_arr[i+1] + norm_fd
                    self.fd_mids[i] = norm_fd
            # Calculate flow areas, wetted perimeters, hydraulic diameters,
            # free surface widths, and velocities
            wetidx = (xc.y - xc.ymin) < self.fd_mids[i]
            self.A_w[i] = xc.calcA(wantidx=wetidx)
            self.P_w[i] = xc.calcP(wantidx=wetidx)
            self.V_w[i] = -self.Q_w/self.A_w[i]
            self.D_H_w[i] = 4*self.A_w[i]/self.P_w[i]
            if self.flow_type[i] != 'full':
                L,R = xc.findLR(self.fd_mids[i])
                self.W[i] = xc.x[R] - xc.x[L]
            else:
                self.W[i] = 0.
            #Set water line in cross-section object
            xc.setFD(self.fd_mids[i])


    def calc_air_flow(self):
        """Calculates airflow (velocity and discharge) within dry portion of
        cave passage.
        """
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
        """Sets boundary conditions for CO2 (air and water) and Ca.
        """
        #Determine upstream boundary concentration for air
        if self.V_a[0]>0:
            CO2_a_boundary = self.pCO2_outside/self.pCO2_high
            air_bnd_cond_idx = 0
        elif self.V_a[0]==0:
            CO2_a_boundary = self.CO2_w_upstream
            air_bnd_cond_idx = -1
        else:
            CO2_a_boundary = self.CO2_a_upstream
            air_bnd_cond_idx = -1

        #Asign upstream boundary conditions into concentration arrays
        self.CO2_a[air_bnd_cond_idx] = CO2_a_boundary
        self.CO2_w[-1] = self.CO2_w_upstream
        self.Ca[-1] = self.Ca_upstream


    def calc_steady_state_transport(self, palmer=False):
        """Calculates transport of carbonate species and calcite dissolution.

        Parameters
        ----------
        palmer : boolean, optional
            Determines whether the Palmer dissolution rate equation should
            be used rather than the transport-limited equation. Default
            value is False (use transport-limited equation).

        Notes
        -----
        For case of summer airflow direction, CO2 in the air and water and
        Ca concentration in the water are calculated starting from upstream
        boundary. For winter airflow direction, a shooting method is used
        to find proper upstream air pCO2 value that produces the required
        downstream (atmospheric) value of pCO2. First a linear shooting
        method is applied. If this does not meet the tolerance specified by
        CO2_err_rel_tol, then Brent's method is used.

        """
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
        """Calculates CO2 and Ca concentrations and dissolution rates starting
        from values at upstream boundary.
        """
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
        for i in np.arange(self.n_nodes-1, 0, -1):
            this_CO2_w = self.CO2_w[i]*self.pCO2_high
            this_CO2_a = self.CO2_a[i]*self.pCO2_high
            this_Ca = self.Ca[i]*self.Ca_eq_0
            if palmer:
                SA = self.P_w[i-1]*self.L_arr[i-1]
                mm_yr_to_mols_per_m2_sec = (cm_per_mm/secs_per_year)*(rho_limestone/g_mol_CaCO3)*cm2_per_m2
                sol = solutionFromCaPCO2(this_Ca, this_CO2_w, T_C=self.T_cave)
                F[i-1] = palmerFromSolution(sol, PCO2=this_CO2_w)*mm_yr_to_mols_per_m2_sec
                R = F[i-1]*SA
            else:
                this_xc = self.xcs[i-1]
                if self.flow_type[i-1] == 'norm':
                    #use bed slope for energy slope in this case
                    eSlope = self.slopes[i-1]
                else:
                    eSlope = (self.h[i] - self.h[i-1])/self.L_arr[i-1]
                this_xc.setEnergySlope(eSlope)
                this_xc.setMaxVelPoint(self.fd_mids[i-1])
                this_xc.calcUmax(self.Q_w)
                T_b = this_xc.calcT_b()
                if T_b.min()<0:
                    print(asdf)
                eps = 5*nu*Sc_Ca**(-1./3.)/np.sqrt(T_b/rho_w)
                Ca_Eq = concCaEqFromPCO2(this_CO2_w, T_C=self.T_cave)
                F_xc = self.reduction_factor*D_Ca/eps*(Ca_Eq - this_Ca)*L_per_m3
                #Smooth F_xc with savgol_filter
                window = int(np.ceil(len(F_xc)/5)//2*2+1)
                F_xc = savgol_filter(F_xc,window,3)
                if F_xc.min() < 0:
                    #Don't allow precipitation
                    F_xc = F_xc*0.0
                this_xc.set_F_xc(F_xc)
                F[i-1] = np.sum(F_xc*this_xc.wet_ls)/self.P_w[i-1] #Units of F are mols/m^2/sec
                R = F[i-1]*self.P_w[i-1]*self.L_arr[i-1]
            R_CO2 = R/self.K_H
            #dx is negative, so signs on dC terms flip
            if self.A_a.min()>0:
                if self.V_a[i-1] != 0:
                    dCO2_a = -self.L_arr[i-1]*K_a[i-1]/self.V_a[i-1]*(this_CO2_w - this_CO2_a)
                else:
                    #For zero airflow, CO2_a goes to CO2_w
                    dCO2_a = 0.
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
        """Erode the cross-sections using results from transport calculation.

        Parameters
        ----------
        dt_frac : float, optional
            The fraction of the timestep for which erosion is being calculated.
            This is used by run_one_step() for calculations of dissolution
            rates for multiple outside air temperatures within a single
            timestep.
        """
        F_to_m_yr = g_mol_CaCO3*secs_per_year/rho_limestone/cm_m**3
        old_ymins = self.ymins.copy()
        for i,xc in enumerate(self.xcs):
            xc.dr = F_to_m_yr*xc.F_xc*self.dt_erode*dt_frac
            xc.erode(xc.dr, trim=self.trim)
            self.ymins[i]= xc.ymin
        #Adjust slopes
        dz = self.ymins - old_ymins
        self.dz = dz
        Celerity_times_dt = np.abs(max(dz/self.slopes))
        CFL = Celerity_times_dt/min((self.x_arr[1:] - self.x_arr[:-1]))
        print('CFL=',CFL)
        self.z_arr[1:] = self.z_arr[1:] + dz
        self.slopes = (self.z_arr[1:] - self.z_arr[:-1])/(self.x_arr[1:] - self.x_arr[:-1])

    def set_T_outside(self, T_outside_C):
        """Set outside temperature to a different value.

        Parameters
        ----------
        T_outside_C : float
            New outside temperature.
        """
        self.T_outside = T_outside_C
        self.T_outside_K = CtoK(T_outside_C)

    def calc_gas_transf_vel_from_eD(self):
        """Calculate gas transfer velocities from energy dissipation.

        Notes
        -----
        Uses relaitonship from Ulseth et al. (2019) with a break in
        power law relationship that separates low and high energy
        streams.
        """
        eD = g*self.slopes*np.abs(self.V_w)
        k_600_m_d = np.exp(3.10 + 0.35*np.log(eD))
        if eD.max()>0.02:
            k_600_m_d[eD>0.02] = np.exp(6.43 + 1.18*np.log(eD[eD>0.02]))
        k_600 = k_600_m_d/secs_per_day
        Sc_CO2 = self.calc_Sc_CO2()
        k_CO2 = k_600*(600/Sc_CO2)**0.5
        self.gas_transf_vel = k_CO2

    def calc_Sc_CO2(self):
        """Calculate Schmidt Number for CO2 at cave temperature.

        Returns
        -------
        Sc_CO2 : float
            Schmidt Number for CO2.
        """
        A = 1742
        B= -91.24
        C=2.208
        D=-0.0219
        T = self.T_cave
        Sc_CO2 = A + B*T + C*T**2 + D*T**3
        return Sc_CO2

    def set_rho_air_cave(self):
        """Calculate and set cave air density.

        Notes
        -----
        Assumes air is saturated with water vapor. Calculates water vapor
        saturation pressure based on Tetens equation and then calculates
        density for resulting mixture of ideal gases at 1 atm total pressure.

        """
        #Calculate saturation water vapor pressure from Tetens equation
        p_wv = 1000. *0.61078 *np.exp(17.27*self.T_cave/(self.T_cave + 237.3))#Pa
        p_da = p_atm - p_wv
        self.rho_air_cave = (p_da/R_da + p_wv/R_wv)/self.T_cave_K
