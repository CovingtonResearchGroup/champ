import numpy as np
from scipy.optimize import root_scalar, minimize_scalar, shgo

from champ.crossSection import CrossSection
from champ.utils.ShapeGen import name_to_function, genCirc
from champ.utils import ShapeGen
# import debugpy

SMALL = 1e-5
WARN_ERR = (
    0.05  # Print warning if flow depth solver has residual larger than this value.
)
alpha = 1.1  # Coriolis coefficient, assumed constant but could vary with velocity
# distribution.


class sim:
    """Base simulation class, for inheritance by more specific simulation types"""
    def __init__(self):
        self.elapsed_time = 0.0
        self.timestep = 0
        self.singleXC = False
        self.uplift = 0.0
        self.uplift_times = None

    def update_params(self, sim_params):
        """
        Update parameters of simulation.

        Takes dictionary and sets simulation parameters
        using keys and values from dictionary.

        Parameters
        ----------
        sim_params : dict
            Dictionary of key values pairs to set into object variables.
        """
        for param in sim_params.keys():
            setattr(self, param, sim_params[param])

    def run_one_step(self):
        """Run one time step of simulation.

        Calculates flow depth and erosion for
        a single time step and updates geometry.
        """

        self.elapsed_time += self.dt_erode
        self.timestep += 1
        self.calc_flow()
        self.erode()
        if not self.singleXC:
            self.apply_uplift()

    # Dummy functions that will be defined in child classes
    def calc_flow(self):
        pass

    def erode(self):
        pass

    def apply_uplift(self):
        """Apply one timestep of uplift"""
        if isinstance(self.uplift, list):
            if self.uplift_idx < len(self.uplift) - 1:
                if self.elapsed_time >= self.uplift_times[self.uplift_idx]:
                    self.uplift_idx += 1
            uplift = self.uplift[self.uplift_idx]
        else:
            uplift = self.uplift
        self.z_arr[0] -= uplift * self.dt_erode
        # Recalculate downstream slope after uplift
        self.slopes[0] = (self.z_arr[1] - self.z_arr[0]) / (
            self.x_arr[1] - self.x_arr[0]
        )

    def set_layers(self, layer_elevs):
        """Set elevations of rock layers"""
        if layer_elevs is not None:
            # Check that number of contacts and K's match
            n_layers = len(self.K)
            n_transitions = len(layer_elevs)
            if n_layers != n_transitions + 1:
                print(
                    (
                        "Number of K values specified must be one more than number of "
                        "transition elevations!"
                    )
                )
                raise IndexError
            else:
                # Check that layer elevations are in correct order
                old_elev = -1e10  # Big negative number
                for i, elev in enumerate(layer_elevs):
                    if i != 0:
                        if elev <= old_elev:
                            raise RuntimeError("Layer elevations in wrong order!")
                        old_elev = elev

                self.layer_elevs = np.array(layer_elevs)
                self.layered_sim = True
        else:
            self.layered_sim = False


class singleXC(sim):
    """Simulation with a single channel cross-section
    
    Parameters
        ----------
        Q_w : float, optional
            Discharge in the channel (m^3/s). Default is 0.1 m^3/s.
        f : float, optional
            Darcy-Weisbach friction factor (unitless), used in both water flow and air
            flow calculations. Default is 0.1.
        n_mann: float or ndarray, optional
            Manning's n. If specified, then f will be calculated from n_mann
            and R_h during flow calculations (which will still use the Darcy-
            Weisbach equation). Default is None.
        init_radius : float, optional
            Initial cross-section radius (meters). Default is 1 m.
        xc_n : int, optional
            Number of points that will define the channel cross-section shape.
            Higher numbers of points can produce numerical instability, requiring
            smaller values of dt_erode or max_frac_erode. Default number is 500.
        slope : float
            Prescribed channel slope. Default is 0.001.
        dt_erode : float, optional
            Erosional time step in years. Default value is 1 year.
        adaptive_step : boolean, optional
            Whether or not to adjust timestep dynamically. Default is False.
        max_frac_erode : float, optional
            Maximum fraction of radial distance to erode within a single timestep
            under adaptive time-stepping. If erosion exceeds this fraction, then
            the timestep will be reduced. If erosion is much less than this fraction,
            then the timestep will be increased. We have not conducted a detailed
            stability analysis. However, initial tests show 0.01 leads to instability,
            whereas the default value is stable. If instabilities occur, and adaptive
            time-stepping is enabled, decreasing this fraction may help.
            Default = 0.005.
        trim : boolean, optional
            Whether or not cross-sections should be trimmed as much of the
            cross-section becomes dry. This enables maintenance of a high
            resolution of the wet portion of the cross-section for simulations
            with substantial incision. If this is set to False, long-term
            simulations are likely to become unstable. Default is True.
        a : float, optional
            Exponent in power law erosion rule (default=1).
        K : float or list, optional
            Erodibility in power law erosion rule (default = 1e-5).
            If multiple layers are specified, then this is a list of
            erodibilities listed from lowest to highest elevation.
        layer_elevs : list of floats, optional
            Specifies a list of elevations (from low to high), where rock
            erodibility changes. If specified, K should be a list with
            one more item than this list.

        Notes
        -----
        To maximize efficiency, use adapative time-stepping. Our tests of stability
        suggest that increasing the number of points in the cross-section (xc_n)
        decreases numerical stability, though it also increases accuracy with which
        the cross-sectional shape is represented. Our default values of xc_n=500 and
        max_frac_erode=0.005 are near the stability threshold for the example cases
        we have run. In our judgment, these values are near optimum for balancing
        fidelity and stability. Increasing xc_n requires a decrease in max_frac_erode.
        Similarly, if the precise shape of the cross-section is not of much concern,
        one could decrease xc_n and increase max_frac_erode, while still maintaining
        numerical stability. Note that this will speed up the simulations for two
        reasons: 1) It decreases the number of points for which erosion must be
        calculated, and 2) The timestep will adjust to a larger value, enabling
        faster simulation of a certain duration of time.
    """

    def __init__(
        self,
        init_radius=1.0,
        Q_w=1.0,
        slope=0.001,
        dt_erode=1.0,
        adaptive_step=False,
        max_frac_erode=0.005,
        f=0.1,
        n_mann=None,
        xc_n=500,
        trim=True,
        a=1.0,
        K=1e-5,
        layer_elevs=None,
    ):
        super(singleXC, self).__init__()
        self.singleXC = True
        self.init_radius = init_radius
        self.Q_w = Q_w
        self.slope = slope
        self.dt_erode = dt_erode
        self.old_dt = dt_erode
        self.adaptive_step = adaptive_step
        self.max_frac_erode = max_frac_erode
        self.f = f
        self.xc_n = xc_n

        x, y = genCirc(init_radius, n=xc_n)
        self.xc = CrossSection(x, y, f=f, n_mann=n_mann)

        self.trim = trim
        self.a = a
        self.K = K
        self.set_layers(layer_elevs)

    def calc_flow(self):
        """Calculate flow depth.
        
        Parameters
        ----------
        """
        old_fd = self.xc.fd
        self.xc.create_A_interp()
        self.xc.create_P_interp()
        norm_fd = self.xc.calcNormalFlowDepth(self.Q_w, self.slope, old_fd=old_fd)
        if norm_fd == -1:
            # pipefull
            delh = self.xc.calcPipeFullHeadGrad(self.Q_w)
            self.xc.setEnergySlope(delh)
        else:
            self.xc.setEnergySlope(self.slope)

    def erode(self):
        """Erode the cross-section.
        """
        if not self.layered_sim:
            self.xc.erode_power_law(a=self.a, K=self.K, dt=self.dt_erode)
        else:
            self.xc.erode_power_law_layered(
                a=self.a, K=self.K, layer_elevs=self.layer_elevs, dt=self.dt_erode
            )
        if self.adaptive_step:
            # Check for percent change in radial distance
            frac_erode = self.xc.dr / self.xc.r_l
            if frac_erode.max() > self.max_frac_erode:
                # Timestep is too big, reduce it
                self.dt_erode = self.dt_erode / 1.5
                print("Reducing timestep to " + str(self.dt_erode))
            elif frac_erode.max() < 0.5 * self.max_frac_erode:
                # Timestep is too small, increase it
                self.dt_erode = self.dt_erode * 1.5
                print("Increasing timestep to " + str(self.dt_erode))


class multiXC(sim):
    """
    Simulation with multiple channel cross-sections.

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
        f : float or ndarray, optional
            Darcy-Weisbach friction factor (unitless). If an array is provided with
            a length equal to the number of cross-sections, then independent values
            will be asigned to each cross-section. Default is 0.1.
        n_mann: float or ndarray, optional
            Manning's n. If specified, then f will be calculated from n_mann
            and R_h during flow calculations (which will still use the Darcy-
            Weisbach equation). Default is None.
        init_radii : float or ndarray, optional
            Initial cross-section radii (meters). If a float then all cross-sections
            will be assigned the same radius. If an array then each element
            represents the radius of a single cross-section (length should be n-1
            where n is the number of nodes). Default is 0.5 m.
        shape_dict : dict, optional
            A dictionary of cross-sectional shape parameters, including name and
            keyword parameters for function in ShapeGen. If this is provided,
            then init_radii is ignored.
        init_offsets : float or ndarray, optional
            These offsets will be added to y-values within initial cross-sections.
            By default, y will be zero at the centroid of the initial cross-section.
            Default value is zero. Should have length of n-1, where n is number of
            nodes.
        xc_n : int, optional
            Number of points that will define the cave passage shape within a
            cross-section. Default is 1000.
        dt_erode : float, optional
            Erosional time step in years. Default value is 1 year.
        uplift : float or list of floats
            Rate of change of baselevel. This distance is subtracted
            from the elevation of the downstream boundary node during
            each timestep.
        uplift_times : list
            Times in years at which uplift rates change. This argument is included if
            uplift is a list of different uplift rates.
        adaptive_step : boolean, optional
            Whether or not to adjust timestep dynamically. Default is False.
        max_frac_erode : float, optional
            Maximum fraction of radial distance to erode within a single timestep
            under adaptive time-stepping. If erosion exceeds this fraction, then
            the timestep will be reduced. If erosion is much less than this fraction,
            then the timestep will be increased. We have not conducted a detailed
            stability analysis. However, initial tests show 0.01 leads to instability,
            whereas the default value is stable. If instabilities occur, and adaptive
            time-stepping is enabled, decreasing this fraction may help.
            Default = 0.005.
        trim : boolean, optional
            Whether or not cross-sections should be trimmed as much of the
            cross-section becomes dry. This enables maintenance of a high
            resolution of the wet portion of the cross-section for simulations
            with substantial incision. If this is set to False, long-term
            simulations are likely to become unstable. Default is True.
        a : float, optional
            Exponent in power law erosion rule (default=1).
        K : float or list, optional
            Erodibility in power law erosion rule (default = 1e-5).
            If multiple layers are specified, then this is a list of
            erodibilities listed from lowest to highest elevation.
        layer_elevs : list of floats, optional
            Specifies a list of elevations (from low to high), where rock
            erodibility changes. If specified, K should be a list with
            one more item than this list.

        Notes
        -----
        To maximize efficiency, use adapative time-stepping. Our tests of stability
        suggest that increasing the number of points in the cross-section (xc_n)
        decreases numerical stability, though it also increases accuracy with which
        the cross-sectional shape is represented. Our default values of xc_n=500 and
        max_frac_erode=0.005 are near the stability threshold for single cross-section
        simulations we have run. Surprisingly, multiXC simulations seem somewhat more
        stable. That is, a larger value of max_frac_erode will still be numerically
        stable (up to 5x for a n=10, xc_n=500 simulation). Increases in the number
        of cross-sections can enhance instability, though normally large numbers of
        cross-sections are needed to see this effect.
        Increasing xc_n requires a decrease in max_frac_erode.
        Similarly, if the precise shape of the cross-section is not of much concern,
        one could decrease xc_n and increase max_frac_erode, while still maintaining
        numerical stability. Note that this will speed up the simulations for two
        reasons: 1) It decreases the number of points for which erosion must be
        calculated, and 2) The timestep will adjust to a larger value, enabling
        faster simulation of a certain duration of time.
    """
    def __init__(
        self,
        x_arr,
        z_arr,
        Q_w=0.1,
        f=0.1,
        n_mann=None,
        init_radii=0.5,
        shape_dict=None,
        init_offsets=0.0,
        xc_n=500,
        dt_erode=1.0,
        uplift=0.0,
        uplift_times=None,
        adaptive_step=False,
        max_frac_erode=0.005,
        trim=True,
        a=1.0,
        K=1e-5,
        layer_elevs=None,
    ):
        super(multiXC, self).__init__()
        self.singleXC = False
        self.n_nodes = x_arr.size
        self.L = x_arr.max() - x_arr.min()
        self.x_arr = x_arr
        self.z_arr = z_arr  # z is zero of xc coords
        # Store initial z values so that absolute elevation can be calculated
        # from XC y values
        self.init_z = np.copy(z_arr)
        # print('init_z=',self.init_z)
        self.L_arr = x_arr[1:] - x_arr[:-1]

        self.Q_w = Q_w

        self.dt_erode = dt_erode
        self.old_dt = dt_erode
        self.uplift = uplift
        self.uplift_times = uplift_times
        self.uplift_idx = 0
        self.adaptive_step = adaptive_step
        self.max_frac_erode = max_frac_erode
        self.xc_n = xc_n
        self.trim = trim
        self.a = a
        self.K = K
        self.set_layers(layer_elevs)

        self.V_w = np.zeros(self.n_nodes - 1)
        self.A_w = np.zeros(self.n_nodes - 1)
        self.P_w = np.zeros(self.n_nodes - 1)
        self.D_H_w = np.zeros(self.n_nodes - 1)
        self.W = np.zeros(self.n_nodes - 1)
        self.fd_mids = np.zeros(self.n_nodes - 1)
        self.init_offsets = np.ones(self.n_nodes - 1) * init_offsets

        self.h = np.zeros(self.n_nodes)
        self.f = f
        self.n_mann = n_mann
        self.flow_type = np.zeros(self.n_nodes - 1, dtype=object)
        self.shape_dict = shape_dict
        self.init_radii = init_radii
        self.initialize_XCs(self.n_nodes - 1)

    def initialize_XCs(self, n_xcs):
        """Initial channel cross-sections.
        
        Parameters
        ----------
        n_xcs : int
            Number of channel cross-sections.
        """

        
        # Initialize cross-sections
        self.xcs = []
        self.radii = self.init_radii * np.ones(n_xcs)
        ymins = []
        for i in np.arange(n_xcs):
            if self.shape_dict is None:
                x, y = genCirc(self.radii[i], n=self.xc_n)
            else:
                if i == 0:
                    shape_name = self.shape_dict["name"]
                    self.shape_dict.pop("name")
                    shape_func_name = name_to_function(shape_name)
                    func_string = "ShapeGen." + shape_func_name + "(**self.shape_dict)"
                x, y = eval(func_string)

            y = y + self.init_offsets[i]
            if self.n_mann is None:
                if np.size(self.f) == 1:
                    xc_f = self.f
                else:
                    xc_f = self.f[i]
                this_xc = CrossSection(x, y, f=xc_f)
            else:
                if np.size(self.n_mann) == 1:
                    xc_n_mann = self.n_mann
                else:
                    xc_n_mann = self.n_mann[i]
                this_xc = CrossSection(x, y, n_mann=xc_n_mann)

            self.xcs.append(this_xc)
            ymins.append(this_xc.ymin)

        self.ymins = np.array(ymins)
        # Reset z to bottom of cross-sections
        if len(self.z_arr) == len(self.ymins):
            self.z_arr = self.z_arr + self.ymins
        else:
            self.z_arr[1:] = self.z_arr[1:] + self.ymins
            self.z_arr[0] = self.z_arr[0] + self.ymins[0]
        self.slopes = (self.z_arr[1:] - self.z_arr[:-1]) / (
            self.x_arr[1:] - self.x_arr[:-1]
        )

    def calc_flow(self, h0=None):
        """Calculates flow depths and hydraulic head values along channel.

        Parameters
        ----------
        h0 : float, optional
            Head value ad downstream node. By default, it is set to existing head at downstream node.

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
        # Set downstream head if provided
        if h0 is not None:
            self.h[0] = h0
        # Loop through cross-sections and solve for flow depths,
        # starting at downstream end
        for i, xc in enumerate(self.xcs):
            old_fd = self.fd_mids[i]
            if old_fd <= 0:
                old_fd = xc.ymax - xc.ymin
            # Initial test showed interpolation takes the longest, flow calc is next,
            # remainder is 10x less.
            # tic = time.perf_counter()
            xc.create_A_interp()
            xc.create_P_interp()
            # toc = time.perf_counter()
            # print(f"Interpolation took {toc - tic:0.4f} seconds.")
            # tic = toc
            # Try calculating flow depth
            backflooded = (self.h[i] - self.z_arr[i + 1] - xc.ymax + xc.ymin) > 0
            over_normal_capacity = False
            if not backflooded:
                norm_fd = xc.calcNormalFlowDepth(
                    self.Q_w, self.slopes[i], old_fd=old_fd
                )
                if (norm_fd < xc.ymax - xc.ymin) and i == 0 and h0 is None:
                    # Transition downstream head boundary to normal flow depth
                    # If we don't do this, we can get stuck in full-pipe conditions
                    # because of downstream boundary head.
                    self.h[0] = self.z_arr[0] + norm_fd
                if norm_fd == -1:
                    over_normal_capacity = True
            if over_normal_capacity or backflooded:
                self.flow_type[i] = "full"
                if i == 0 and h0 is None:
                    # if downstream boundary set head to top of cross-section
                    self.h[0] = self.z_arr[0] + xc.ymax - xc.ymin
                # We have a full pipe, calculate head gradient instead
                delh = xc.calcPipeFullHeadGrad(self.Q_w)
                self.h[i + 1] = self.h[i] + delh * self.L_arr[i]
                self.fd_mids[i] = xc.ymax - xc.ymin
            else:
                # crit_fd = xc.calcCritFlowDepth(self.Q_w)
                # y_star = norm_fd  # min([crit_fd,norm_fd])
                y_out = self.h[i] - self.z_arr[i]
                # downstream_critical = y_star > y_out and y_star > 0  # and i>0
                partial_backflood = norm_fd < self.h[i] - self.z_arr[i + 1]
                # downstream_less_normal = norm_fd > y_out
                if partial_backflood:  # upstream node is flooded above normal depth
                    self.flow_type[i] = "pbflood"
                    y_in = xc.calcUpstreamHead(
                        self.Q_w, self.slopes[i], y_out, self.L_arr[i]
                    )
                    if (
                        y_in > 0 and (y_out + y_in) / 2.0 < xc.ymax - xc.ymin
                    ):  # or could use fraction of y_in
                        self.h[i + 1] = self.z_arr[i + 1] + y_in
                        self.fd_mids[i] = (y_out + y_in) / 2.0
                    else:
                        # We need full pipe to push needed Q
                        delh = xc.calcPipeFullHeadGrad(self.Q_w)
                        self.h[i + 1] = self.h[i] + delh * self.L_arr[i]
                        self.fd_mids[i] = xc.ymax - xc.ymin
                        self.flow_type[i] = "full"
                # elif downstream_critical:
                #    self.flow_type[i] = 'dwnscrit'
                #    #Use minimum of critical or normal depth for downstream y
                #    y_in = xc.calcUpstreamHead(self.Q_w,self.slopes[i],
                #                               y_star,self.L_arr[i],f=self.f)
                #    self.fd_mids[i] = (y_in + y_star)/2.
                #    self.h[i+1] = self.z_arr[i+1] + y_in
                #    if i==0:
                #        self.h[0]=self.z_arr[0] + y_star#norm_fd #y_star
                # elif downstream_less_normal:
                #    self.flow_type[i] = 'dwnslessnorm'
                #    y_in = xc.calcUpstreamHead(self.Q_w,self.slopes[i],
                #                               y_out,self.L_arr[i],f=self.f)
                #    self.h[i+1] = self.z_arr[i+1] + y_in
                #    self.fd_mids[i] = (y_out+y_in)/2.
                else:
                    self.flow_type[i] = "norm"
                    if i == 0 and h0 is None:
                        self.h[i] = norm_fd + self.z_arr[i]
                    # dz = slopes[i]*(x[i+1] - x[i])
                    self.h[i + 1] = self.z_arr[i + 1] + norm_fd
                    self.fd_mids[i] = norm_fd
            # toc = time.perf_counter()
            # print(f"Calculating flow depth took {toc - tic:0.4f} seconds.")
            # tic = toc
            # Calculate flow areas, wetted perimeters, hydraulic diameters,
            # free surface widths, and velocities
            self.A_w[i] = xc.calcA(depth=self.fd_mids[i])
            self.P_w[i] = xc.calcP(depth=self.fd_mids[i])
            self.V_w[i] = -self.Q_w / self.A_w[i]
            self.D_H_w[i] = 4 * self.A_w[i] / self.P_w[i]
            if self.flow_type[i] != "full":
                L, R = xc.findLR(self.fd_mids[i])
                self.W[i] = xc.x[R] - xc.x[L]
            else:
                self.W[i] = 0.0
            # Set water line in cross-section object
            xc.setFD(self.fd_mids[i])
            if self.flow_type[i] == "norm":
                # use bed slope for energy slope in this case
                eSlope = self.slopes[i]
            else:
                eSlope = (self.h[i + 1] - self.h[i]) / self.L_arr[i]
            xc.setEnergySlope(eSlope)
            # toc = time.perf_counter()
            # print(f"Remainder took {toc - tic:0.4f} seconds.")

    def erode(self):
        """Erode the cross-sections.
        """
        old_ymins = self.ymins.copy()
        for i, xc in enumerate(self.xcs):
            if not self.layered_sim:
                xc.erode_power_law(a=self.a, K=self.K, dt=self.dt_erode)
            else:
                # print('layer_elevs=',self.layer_elevs)
                # print('init_z=',self.init_z[i+1])
                if len(self.init_z == len(self.xcs)):
                    absolute_layer_elevs = self.layer_elevs - self.init_z[i]
                else:
                    absolute_layer_elevs = self.layer_elevs - self.init_z[i + 1]
                # print('i=',i, ' abs_layer_elevs=',absolute_layer_elevs)
                xc.erode_power_law_layered(
                    a=self.a,
                    K=self.K,
                    layer_elevs=absolute_layer_elevs,
                    dt=self.dt_erode,
                )
            self.ymins[i] = xc.ymin

        # Adjust slopes
        dz = self.ymins - old_ymins
        self.dz = dz
        # Celerity_times_dt = np.abs(max(dz / self.slopes))
        # CFL = Celerity_times_dt / min((self.x_arr[1:] - self.x_arr[:-1]))
        # print('CFL=',CFL)
        if len(dz) == len(self.z_arr) - 1:
            # Average upstream and downstream erosion rates
            # to calculate rates on nodes
            # Seems more stable than applying erosion to upstream node
            # self.z_arr[1:-1] = self.z_arr[1:-1] + (dz[1:] + dz[:-1]) / 2
            # Asymmetric weighting Much more stable that weighting the downstream
            # erosion, which is surprising
            # self.z_arr[1:-1] = self.z_arr[1:-1] + (2 * dz[1:] + dz[:-1]) / 3
            # Still eventially unstable
            # self.z_arr[1:-1] = self.z_arr[1:-1] + dz[1:]
            # self.z_arr[-1] = self.z_arr[-1] + dz[-1]
            # Original formulation, but quite unstable for GVF midXCs solver
            self.z_arr[1:] = self.z_arr[1:] + dz

            # Highly unstable
            # if isinstance(self.uplift, list):
            #    if self.uplift_idx < len(self.uplift) - 1:
            #        if self.elapsed_time >= self.uplift_times[self.uplift_idx]:
            #            self.uplift_idx += 1
            #    uplift = self.uplift[self.uplift_idx]
            # else:
            #    uplift = self.uplift

            # dz_tot = np.zeros(self.n_nodes)
            # dz_tot[1:] = dz
            # dz_tot[0] = -uplift * self.dt_erode
            # self.z_arr[1:] = self.z_arr[1:] + (dz_tot[1:] + dz_tot[:-1]) / 2
        else:
            # Somewhat more stable? Instability creeps in from upstream?
            # self.z_arr[1:-1] = self.z_arr[1:-1] + (dz[1:-1] + dz[2:]) / 2
            # self.z_arr[-1] = self.z_arr[-1] + dz[-1]
            self.z_arr[1:] = self.z_arr[1:] + dz[1:]
        self.slopes = (self.z_arr[1:] - self.z_arr[:-1]) / (
            self.x_arr[1:] - self.x_arr[:-1]
        )

        # Set old_dt for use in plots that calculate erosion rates
        self.old_dt = self.dt_erode
        if self.adaptive_step:
            # Check for percent change in radial distance
            sim_max_frac_erode = 0.0
            for xc in self.xcs:
                frac_erode = xc.dr / xc.r_l
                xc_max_frac_erode = frac_erode.max()
                if xc_max_frac_erode > sim_max_frac_erode:
                    sim_max_frac_erode = xc_max_frac_erode

            if sim_max_frac_erode > self.max_frac_erode:
                # Timestep is too big, reduce it
                self.dt_erode = self.dt_erode / 1.5
                print("Reducing timestep to " + str(self.dt_erode))
            elif sim_max_frac_erode < 0.5 * self.max_frac_erode:
                # Timestep is too small, increase it
                self.dt_erode = self.dt_erode * 1.5
                print("Increasing timestep to " + str(self.dt_erode))


class multiXCNormalFlow(multiXC):
    """Simulation with multiple cross-sections that assumes normal flow conditions.
    
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
        f : float or ndarray, optional
            Darcy-Weisbach friction factor (unitless). If an array is provided with
            a length equal to the number of cross-sections, then independent values
            will be asigned to each cross-section. Default is 0.1.
        n_mann: float or ndarray, optional
            Manning's n. If specified, then f will be calculated from n_mann
            and R_h during flow calculations (which will still use the Darcy-
            Weisbach equation). Default is None.
        init_radii : float or ndarray, optional
            Initial cross-section radii (meters). If a float then all cross-sections
            will be assigned the same radius. If an array then each element
            represents the radius of a single cross-section (length should be n-1
            where n is the number of nodes). Default is 0.5 m.
        shape_dict : dict, optional
            A dictionary of cross-sectional shape parameters, including name and
            keyword parameters for function in ShapeGen. If this is provided,
            then init_radii is ignored.
        init_offsets : float or ndarray, optional
            These offsets will be added to y-values within initial cross-sections.
            By default, y will be zero at the centroid of the initial cross-section.
            Default value is zero. Should have length of n-1, where n is number of
            nodes.
        xc_n : int, optional
            Number of points that will define the cave passage shape within a
            cross-section. Default is 1000.
        dt_erode : float, optional
            Erosional time step in years. Default value is 1 year.
        uplift : float or list of floats
            Rate of change of baselevel. This distance is subtracted
            from the elevation of the downstream boundary node during
            each timestep.
        uplift_times : list
            Times in years at which uplift rates change. This argument is included if
            uplift is a list of different uplift rates.
        adaptive_step : boolean, optional
            Whether or not to adjust timestep dynamically. Default is False.
        max_frac_erode : float, optional
            Maximum fraction of radial distance to erode within a single timestep
            under adaptive time-stepping. If erosion exceeds this fraction, then
            the timestep will be reduced. If erosion is much less than this fraction,
            then the timestep will be increased. We have not conducted a detailed
            stability analysis. However, initial tests show 0.01 leads to instability,
            whereas the default value is stable. If instabilities occur, and adaptive
            time-stepping is enabled, decreasing this fraction may help.
            Default = 0.005.
        trim : boolean, optional
            Whether or not cross-sections should be trimmed as much of the
            cross-section becomes dry. This enables maintenance of a high
            resolution of the wet portion of the cross-section for simulations
            with substantial incision. If this is set to False, long-term
            simulations are likely to become unstable. Default is True.
        a : float, optional
            Exponent in power law erosion rule (default=1).
        K : float or list, optional
            Erodibility in power law erosion rule (default = 1e-5).
            If multiple layers are specified, then this is a list of
            erodibilities listed from lowest to highest elevation.
        layer_elevs : list of floats, optional
            Specifies a list of elevations (from low to high), where rock
            erodibility changes. If specified, K should be a list with
            one more item than this list.

        Notes
        -----
        To maximize efficiency, use adapative time-stepping. Our tests of stability
        suggest that increasing the number of points in the cross-section (xc_n)
        decreases numerical stability, though it also increases accuracy with which
        the cross-sectional shape is represented. Our default values of xc_n=500 and
        max_frac_erode=0.005 are near the stability threshold for single cross-section
        simulations we have run. Surprisingly, multiXC simulations seem somewhat more
        stable. That is, a larger value of max_frac_erode will still be numerically
        stable (up to 5x for a n=10, xc_n=500 simulation). Increases in the number
        of cross-sections can enhance instability, though normally large numbers of
        cross-sections are needed to see this effect.
        Increasing xc_n requires a decrease in max_frac_erode.
        Similarly, if the precise shape of the cross-section is not of much concern,
        one could decrease xc_n and increase max_frac_erode, while still maintaining
        numerical stability. Note that this will speed up the simulations for two
        reasons: 1) It decreases the number of points for which erosion must be
        calculated, and 2) The timestep will adjust to a larger value, enabling
        faster simulation of a certain duration of time.
"""

    def calc_flow(self):
        """Calculates flow depths assuming normal flow.

        Notes
        -----
        Starts at downstream end and propagates solution upstream. Flow is assumed
        to be normal.

        """
        # Loop through cross-sections and solve for flow depths,
        # starting at downstream end
        for i, xc in enumerate(self.xcs):
            old_fd = self.fd_mids[i]
            if old_fd <= 0:
                old_fd = xc.ymax - xc.ymin
            xc.create_A_interp()
            xc.create_P_interp()
            norm_fd = xc.calcNormalFlowDepth(self.Q_w, self.slopes[i], old_fd=old_fd)
            self.flow_type[i] = "norm"
            if i == 0:
                self.h[i] = norm_fd + self.z_arr[i]
            self.h[i + 1] = self.z_arr[i + 1] + norm_fd
            self.fd_mids[i] = norm_fd
            # Calculate flow areas, wetted perimeters, hydraulic diameters,
            # free surface widths, and velocities
            self.A_w[i] = xc.calcA(depth=self.fd_mids[i])
            self.P_w[i] = xc.calcP(depth=self.fd_mids[i])
            self.V_w[i] = -self.Q_w / self.A_w[i]
            self.D_H_w[i] = 4 * self.A_w[i] / self.P_w[i]
            L, R = xc.findLR(self.fd_mids[i])
            self.W[i] = xc.x[R] - xc.x[L]
            # Set water line in cross-section object
            xc.setFD(self.fd_mids[i])
            # use bed slope for energy slope in this case
            eSlope = self.slopes[i]
            xc.setEnergySlope(eSlope)


class multiXCGVF(multiXC):
    """Simulation with multiple cross-sections that assumes gradually varied flow.
    
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
        f : float or ndarray, optional
            Darcy-Weisbach friction factor (unitless). If an array is provided with
            a length equal to the number of cross-sections, then independent values
            will be asigned to each cross-section. Default is 0.1.
        n_mann: float or ndarray, optional
            Manning's n. If specified, then f will be calculated from n_mann
            and R_h during flow calculations (which will still use the Darcy-
            Weisbach equation). Default is None.
        init_radii : float or ndarray, optional
            Initial cross-section radii (meters). If a float then all cross-sections
            will be assigned the same radius. If an array then each element
            represents the radius of a single cross-section (length should be n-1
            where n is the number of nodes). Default is 0.5 m.
        shape_dict: dict, optional
            A dictionary of cross-sectional shape parameters, including name and
            keyword parameters for function in ShapeGen. If this is provided,
            then init_radii is ignored.
        init_offsets : float or ndarray, optional
            These offsets will be added to y-values within initial cross-sections.
            By default, y will be zero at the centroid of the initial cross-section.
            Default value is zero. Should have length of n-1, where n is number of
            nodes.
        xc_n : int, optional
            Number of points that will define the cave passage shape within a
            cross-section. Default is 1000.
        dt_erode : float, optional
            Erosional time step in years. Default value is 1 year.
        uplift : float or list of floats
            Rate of change of baselevel. This distance is subtracted
            from the elevation of the downstream boundary node during
            each timestep.
        uplift_times : list
            Times in years at which uplift rates change. This argument is included if
            uplift is a list of different uplift rates.
        adaptive_step : boolean, optional
            Whether or not to adjust timestep dynamically. Default is False.
        max_frac_erode : float, optional
            Maximum fraction of radial distance to erode within a single timestep
            under adaptive time-stepping. If erosion exceeds this fraction, then
            the timestep will be reduced. If erosion is much less than this fraction,
            then the timestep will be increased. We have not conducted a detailed
            stability analysis. However, initial tests show 0.01 leads to instability,
            whereas the default value is stable. If instabilities occur, and adaptive
            time-stepping is enabled, decreasing this fraction may help.
            Default = 0.005.
        trim : boolean, optional
            Whether or not cross-sections should be trimmed as much of the
            cross-section becomes dry. This enables maintenance of a high
            resolution of the wet portion of the cross-section for simulations
            with substantial incision. If this is set to False, long-term
            simulations are likely to become unstable. Default is True.
        a : float, optional
            Exponent in power law erosion rule (default=1).
        K : float or list, optional
            Erodibility in power law erosion rule (default = 1e-5).
            If multiple layers are specified, then this is a list of
            erodibilities listed from lowest to highest elevation.
        layer_elevs : list of floats, optional
            Specifies a list of elevations (from low to high), where rock
            erodibility changes. If specified, K should be a list with
            one more item than this list.
        abs_tol : float
            Maximum allowed error for flow solver.
        max_iterations: int
            Maximum number of allowed iterations for flow solver.

        Notes
        -----
        To maximize efficiency, use adapative time-stepping. Our tests of stability
        suggest that increasing the number of points in the cross-section (xc_n)
        decreases numerical stability, though it also increases accuracy with which
        the cross-sectional shape is represented. Our default values of xc_n=500 and
        max_frac_erode=0.005 are near the stability threshold for single cross-section
        simulations we have run. Surprisingly, multiXC simulations seem somewhat more
        stable. That is, a larger value of max_frac_erode will still be numerically
        stable (up to 5x for a n=10, xc_n=500 simulation). Increases in the number
        of cross-sections can enhance instability, though normally large numbers of
        cross-sections are needed to see this effect.
        Increasing xc_n requires a decrease in max_frac_erode.
        Similarly, if the precise shape of the cross-section is not of much concern,
        one could decrease xc_n and increase max_frac_erode, while still maintaining
        numerical stability. Note that this will speed up the simulations for two
        reasons: 1) It decreases the number of points for which erosion must be
        calculated, and 2) The timestep will adjust to a larger value, enabling
        faster simulation of a certain duration of time.
        """

    def __init__(
        self,
        x_arr,
        z_arr,
        Q_w=0.1,
        f=0.1,
        n_mann=None,
        init_radii=0.5,
        shape_dict=None,
        init_offsets=0.0,
        xc_n=500,
        dt_erode=1.0,
        uplift=0.0,
        uplift_times=None,
        adaptive_step=False,
        max_frac_erode=0.005,
        trim=True,
        a=1.0,
        K=1e-5,
        layer_elevs=None,
        abs_tol=0.001,
        max_iterations=50,
    ):
        super(multiXC, self).__init__()
        self.singleXC = False
        self.n_nodes = x_arr.size
        self.L = x_arr.max() - x_arr.min()
        self.x_arr = x_arr
        self.z_arr = z_arr  # z is zero of xc coords
        # Store initial z values so that absolute elevation can be calculated
        # from XC y values
        self.init_z = np.copy(z_arr)
        # print('init_z=',self.init_z)
        self.L_arr = x_arr[1:] - x_arr[:-1]

        self.Q_w = Q_w

        self.dt_erode = dt_erode
        self.old_dt = dt_erode
        self.uplift = uplift
        self.uplift_times = uplift_times
        self.uplift_idx = 0
        self.adaptive_step = adaptive_step
        self.max_frac_erode = max_frac_erode
        self.xc_n = xc_n
        self.trim = trim
        self.a = a
        self.K = K
        self.set_layers(layer_elevs)

        self.V_w = np.zeros(self.n_nodes)
        self.A_w = np.zeros(self.n_nodes)
        self.P_w = np.zeros(self.n_nodes)
        self.D_H_w = np.zeros(self.n_nodes)
        self.W = np.zeros(self.n_nodes)
        self.fd = np.zeros(self.n_nodes)
        self.init_offsets = np.ones(self.n_nodes) * init_offsets

        self.h = np.zeros(self.n_nodes)
        self.flow_type = np.zeros(self.n_nodes, dtype=object)

        self.abs_tol = abs_tol
        self.max_iterations = max_iterations
        self.f = f
        self.n_mann = n_mann
        self.shape_dict = shape_dict
        self.init_radii = init_radii
        self.initialize_XCs(self.n_nodes)

    def calc_flow(self, h0=None):
        """Calculates flow depths assuming gradually varied open channel flow.

        Notes
        -----
        Starts at downstream end and propagates solution upstream. Flow is assumed
        to be subcritical and gradually varied. If flow depths go below critical
        depth, then depth is assumed to be critical.

        """
        for i, xc in enumerate(self.xcs[:-1]):
            xc_up = self.xcs[i + 1]
            # Renew interpolation functions
            xc_up.create_A_interp()
            xc_up.create_P_interp()
            xc_up.Q = self.Q_w  # Otherwise this is done by calcNormalFlowDepth.
            # We need to do it explictly here for other XCs
            # that aren't the first one.
            if i == 0:
                xc.create_A_interp()
                xc.create_P_interp()
                norm_fd = xc.calcNormalFlowDepth(self.Q_w, self.slopes[i])
                if h0 is None:
                    self.h[i] = norm_fd + self.z_arr[i]
                    self.fd[i] = norm_fd
                else:
                    self.h[i] = h0
                    self.fd[i] = h0 - self.z_arr[i]

            A_down = xc.calcA(depth=self.fd[i])
            P_down = xc.calcP(depth=self.fd[i])
            D_H_down = 4 * A_down / P_down
            # K_down = xc.calcConvey(self.fd[i], f=self.f)
            V_down = self.Q_w / A_down
            V_head_down = alpha * V_down**2 / (2 * xc.g)
            H_down = self.h[i] + V_head_down
            if xc.n_mann is not None:
                xc.set_f_from_n_mann(D_H_down)
            S_f_down = xc.f * V_down**2 / (2 * xc.g * D_H_down)
            dx = self.x_arr[i + 1] - self.x_arr[i]
            if self.fd[i + 1] > 0:
                fd_guess = self.fd[i + 1]
            else:
                # Use depth from previous XC if available
                fd_guess = self.fd[i]
            norm_fd = xc.calcNormalFlowDepth(self.Q_w, self.slopes[i])
            fd_crit = xc_up.calcCritFlowDepth(self.Q_w)
            # print(fd_guess)
            try:
                # Search for best bracket
                n_search = 10
                fd_search = np.linspace(
                    fd_guess * 1.5, 0.8 * min([fd_crit, norm_fd]), n_search
                )
                bracket_found = False
                sign_this_res = None
                sign_old_res = None
                j = 0
                while not bracket_found and j + 1 < len(fd_search):
                    this_res = self.fd_residual(
                        fd_search[j], i + 1, H_down, S_f_down, dx
                    )
                    # print("this_res =", this_res)
                    sign_this_res = np.sign(this_res)
                    if sign_old_res is not None:
                        if sign_this_res * sign_old_res == -1:
                            # We have a sign change in residual
                            low_bracket = fd_search[j]
                            high_bracket = fd_search[j - 1]
                            bracket_found = True
                    sign_old_res = sign_this_res
                    j += 1
                # print("bracket found =", bracket_found)
                if not bracket_found:
                    low_bracket = fd_crit
                    high_bracket = fd_guess * 1.2

                sol = root_scalar(
                    self.fd_residual,
                    args=(i + 1, H_down, S_f_down, dx),
                    method="brenth",
                    x0=fd_guess,
                    bracket=(low_bracket, high_bracket),
                    xtol=0.00001,
                    rtol=0.00005,
                )
                is_converged = sol.converged
            except ValueError:
                print("Falling back on minimization solver.")
                is_converged = False
            #            sol = root_scalar(
            #                self.fd_residual,
            #                args=(i + 1, H_down, S_f_down, dx),
            #                x0=fd_guess,
            #                x1=0.9 * fd_guess,
            #            )

            if is_converged:
                fd_sol = sol.root
                flag = sol.flag
                converged = sol.converged
            else:
                # Try minimization of abs error
                # res = minimize_scalar(
                #    self.fd_residual_abs,
                #    bracket=(fd_crit, 1.1 * fd_guess),
                #    args=(i + 1, H_down, S_f_down, dx),
                # )
                fd_max = xc.ymax - xc.ymin
                res = shgo(
                    self.fd_residual_abs,
                    [
                        (fd_crit, fd_max),
                    ],
                    n=32,
                    sampling_method="sobol",
                    args=(i + 1, H_down, S_f_down, dx),
                )
                # converged = res.success
                # print("converged =", converged, "  fun=", res.fun)
                # print(res)
                # flag = "used minimization solver"
                fd_sol = res.x[0]
            # Calculate actual flow depth residual
            err = self.fd_residual(fd_sol, i + 1, H_down, S_f_down, dx)
            # print("i=", i, "  err=", err, " fd=", fd_sol)
            if abs(err) > WARN_ERR:
                print("*******************************************")
                print("Warning! Flow depth solution is inaccurate. Error is", err)
                print("*******************************************")

            xc_up.setFD(fd_sol)
            """print(
                "fd_sol =",
                fd_sol,
                "  fd_crit =",
                fd_crit,
                " converged =",
                converged,
                "  flag =",normal flow conditions
                flag,
            )"""
            if fd_sol < fd_crit:
                # Force critical flow
                fd_sol = fd_crit

            self.h[i + 1] = self.z_arr[i + 1] + fd_sol
            self.fd[i + 1] = fd_sol
            xc_up.setFD(fd_sol)

        # Calculate flow areas, wetted perimeters, hydraulic diameters,
        # free surface widths, and velocities
        for i, xc in enumerate(self.xcs):
            # print("i =", i, "  fd=", self.fd[i])
            self.A_w[i] = xc.calcA(depth=self.fd[i])
            self.P_w[i] = xc.calcP(depth=self.fd[i])
            self.V_w[i] = -self.Q_w / self.A_w[i]
            self.D_H_w[i] = 4 * self.A_w[i] / self.P_w[i]
            if self.fd[i] < xc.ymax - xc.ymin:
                L, R = xc.findLR(self.fd[i])
                self.W[i] = xc.x[R] - xc.x[L]
            else:
                self.W[i] = 0.0
            if xc.n_mann is not None:
                xc.set_f_from_n_mann(self.D_H_w[i])
            S_f = xc.f * self.V_w[i] ** 2 / (2 * xc.g * self.D_H_w[i])
            xc.setEnergySlope(S_f)

    def fd_residual(self, fd_guess, xc_up_idx, H_down, S_f_down, dx):
        """Calculate residual between guessed upstream flow depth and energy
           equation flow depth.

        Parameters
        ----------
        fd_guess : float
            Guessed upstream flow depth.
        xc_idx : int
            Index of current downstream cross-section.
        H_down : float
            Head at downstream cross-section.
        S_f_down : float
            Friction slope at downstream cross-section
        dx : float
            Distance between cross-sections.
        """
        xc_up = self.xcs[xc_up_idx]
        A_up = xc_up.calcA(depth=fd_guess)
        P_up = xc_up.calcP(depth=fd_guess)
        if A_up < SMALL:
            A_up = SMALL
        if P_up < SMALL:
            P_up = SMALL
        V_up = self.Q_w / A_up
        V_head_up = alpha * V_up**2 / (2 * xc_up.g)
        D_H_up = 4 * A_up / P_up
        if xc_up.n_mann is not None:
            xc_up.set_f_from_n_mann(D_H_up)
        S_f_up = xc_up.f * V_up**2 / (2 * xc_up.g * D_H_up)
        H_up_energy = H_down + 0.5 * (S_f_down + S_f_up) * dx
        fd_up_energy = H_up_energy - V_head_up - self.z_arr[xc_up_idx]
        err = fd_up_energy - fd_guess
        return err

    def fd_residual_abs(self, fd_guess, xc_up_idx, H_down, S_f_down, dx):
        fd_res = self.fd_residual(fd_guess, xc_up_idx, H_down, S_f_down, dx)
        return abs(fd_res)


class multiXCGVF_midXCs(multiXC):
    """Simulation with multiple cross-sections that assumes gradually varied flow and has cross-sections at conduit midpoints.
    
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
        f : float or ndarray, optional
            Darcy-Weisbach friction factor (unitless). If an array is provided with
            a length equal to the number of cross-sections, then independent values
            will be asigned to each cross-section. Default is 0.1.
        n_mann: float or ndarray, optional
            Manning's n. If specified, then f will be calculated from n_mann
            and R_h during flow calculations (which will still use the Darcy-
            Weisbach equation). Default is None.
        init_radii : float or ndarray, optional
            Initial cross-section radii (meters). If a float then all cross-sections
            will be assigned the same radius. If an array then each element
            represents the radius of a single cross-section (length should be n-1
            where n is the number of nodes). Default is 0.5 m.
        shape_dict: dict, optional
            A dictionary of cross-sectional shape parameters, including name and
            keyword parameters for function in ShapeGen. If this is provided,
            then init_radii is ignored.
        init_offsets : float or ndarray, optional
            These offsets will be added to y-values within initial cross-sections.
            By default, y will be zero at the centroid of the initial cross-section.
            Default value is zero. Should have length of n-1, where n is number of
            nodes.
        xc_n : int, optional
            Number of points that will define the cave passage shape within a
            cross-section. Default is 1000.
        dt_erode : float, optional
            Erosional time step in years. Default value is 1 year.
        uplift : float or list of floats
            Rate of change of baselevel. This distance is subtracted
            from the elevation of the downstream boundary node during
            each timestep.
        uplift_times : list
            Times in years at which uplift rates change. This argument is included if
            uplift is a list of different uplift rates.
        adaptive_step : boolean, optional
            Whether or not to adjust timestep dynamically. Default is False.
        max_frac_erode : float, optional
            Maximum fraction of radial distance to erode within a single timestep
            under adaptive time-stepping. If erosion exceeds this fraction, then
            the timestep will be reduced. If erosion is much less than this fraction,
            then the timestep will be increased. We have not conducted a detailed
            stability analysis. However, initial tests show 0.01 leads to instability,
            whereas the default value is stable. If instabilities occur, and adaptive
            time-stepping is enabled, decreasing this fraction may help.
            Default = 0.005.
        trim : boolean, optional
            Whether or not cross-sections should be trimmed as much of the
            cross-section becomes dry. This enables maintenance of a high
            resolution of the wet portion of the cross-section for simulations
            with substantial incision. If this is set to False, long-term
            simulations are likely to become unstable. Default is True.
        a : float, optional
            Exponent in power law erosion rule (default=1).
        K : float or list, optional
            Erodibility in power law erosion rule (default = 1e-5).
            If multiple layers are specified, then this is a list of
            erodibilities listed from lowest to highest elevation.
        layer_elevs : list of floats, optional
            Specifies a list of elevations (from low to high), where rock
            erodibility changes. If specified, K should be a list with
            one more item than this list.
        abs_tol : float
            Maximum allowed error for flow solver.
        max_iterations: int
            Maximum number of allowed iterations for flow solver.

        Notes
        -----
        To maximize efficiency, use adapative time-stepping. Our tests of stability
        suggest that increasing the number of points in the cross-section (xc_n)
        decreases numerical stability, though it also increases accuracy with which
        the cross-sectional shape is represented. Our default values of xc_n=500 and
        max_frac_erode=0.005 are near the stability threshold for single cross-section
        simulations we have run. Surprisingly, multiXC simulations seem somewhat more
        stable. That is, a larger value of max_frac_erode will still be numerically
        stable (up to 5x for a n=10, xc_n=500 simulation). Increases in the number
        of cross-sections can enhance instability, though normally large numbers of
        cross-sections are needed to see this effect.
        Increasing xc_n requires a decrease in max_frac_erode.
        Similarly, if the precise shape of the cross-section is not of much concern,
        one could decrease xc_n and increase max_frac_erode, while still maintaining
        numerical stability. Note that this will speed up the simulations for two
        reasons: 1) It decreases the number of points for which erosion must be
        calculated, and 2) The timestep will adjust to a larger value, enabling
        faster simulation of a certain duration of time.
        """

    def __init__(
        self,
        x_arr,
        z_arr,
        Q_w=0.1,
        f=0.1,
        n_mann=None,
        init_radii=0.5,
        shape_dict=None,
        init_offsets=0.0,
        xc_n=500,
        dt_erode=1.0,
        uplift=0.0,
        uplift_times=None,
        adaptive_step=False,
        max_frac_erode=0.005,
        trim=True,
        a=1.0,
        K=1e-5,
        layer_elevs=None,
        abs_tol=0.001,
        max_iterations=50,
    ):

        """
        """
        super(multiXC, self).__init__()
        self.singleXC = False
        self.n_nodes = x_arr.size
        self.L = x_arr.max() - x_arr.min()
        self.x_arr = x_arr
        self.z_arr = z_arr  # z is zero of xc coords
        # Store initial z values so that absolute elevation can be calculated
        # from XC y values
        self.init_z = np.copy(z_arr)
        # print('init_z=',self.init_z)
        self.L_arr = x_arr[1:] - x_arr[:-1]

        self.Q_w = Q_w

        self.dt_erode = dt_erode
        self.old_dt = dt_erode
        self.uplift = uplift
        self.uplift_times = uplift_times
        self.uplift_idx = 0
        self.adaptive_step = adaptive_step
        self.max_frac_erode = max_frac_erode
        self.xc_n = xc_n
        self.trim = trim
        self.a = a
        self.K = K
        self.set_layers(layer_elevs)

        self.V_w = np.zeros(self.n_nodes - 1)
        self.A_w = np.zeros(self.n_nodes - 1)
        self.P_w = np.zeros(self.n_nodes - 1)
        self.D_H_w = np.zeros(self.n_nodes - 1)
        self.W = np.zeros(self.n_nodes - 1)
        self.fd_mids = np.zeros(self.n_nodes - 1)
        self.init_offsets = np.ones(self.n_nodes - 1) * init_offsets

        self.h = np.zeros(self.n_nodes)
        self.flow_type = np.zeros(self.n_nodes - 1, dtype=object)

        self.abs_tol = abs_tol
        self.max_iterations = max_iterations
        self.f = f
        self.n_mann = n_mann
        self.shape_dict = shape_dict
        self.init_radii = init_radii
        self.initialize_XCs(self.n_nodes - 1)

    def calc_flow(self, h0=None):
        """Calculates flow depths assuming gradually varied open channel flow.

        Notes
        -----
        Starts at downstream end and propagates solution upstream. Flow is assumed
        to be subcritical and gradually varied. If flow depths go below critical
        depth, then depth is assumed to be critical.

        """
        for i, xc in enumerate(self.xcs[:-1]):
            # print("i=", i)
            xc_up = self.xcs[i + 1]
            # Renew interpolation functions
            xc_up.create_A_interp()
            xc_up.create_P_interp()
            xc_up.Q = self.Q_w  # Otherwise this is done by calcNormalFlowDepth.
            # We need to do it explictly here for other XCs
            # that aren't the first one.
            if i == 0:
                xc.create_A_interp()
                xc.create_P_interp()
                norm_fd = xc.calcNormalFlowDepth(self.Q_w, self.slopes[i])
                if h0 is None:
                    self.h[i] = norm_fd + self.z_arr[i]
                    self.fd_mids[i] = norm_fd
                else:
                    self.h[i] = h0
                    if self.h[i + 1] == 0:
                        # If we haven't solved for flow yet, use downstream flow depth
                        self.fd_mids[i] = h0 - self.z_arr[i]
                    else:
                        # Otherwise, average flow depth between adjacent nodes
                        self.fd_mids[i] = (
                            (h0 - self.z_arr[i]) + (self.h[i + 1] - self.z_arr[i + 1])
                        ) / 2

            A_down = xc.calcA(depth=self.fd_mids[i])
            P_down = xc.calcP(depth=self.fd_mids[i])
            D_H_down = 4 * A_down / P_down
            V_down = self.Q_w / A_down
            V_head_down = alpha * V_down**2 / (2 * xc.g)
            zmid = (self.z_arr[i] + self.z_arr[i + 1]) / 2
            H_down = self.fd_mids[i] + zmid + V_head_down
            if xc.n_mann is not None:
                xc.set_f_from_n_mann(D_H_down)
            S_f_down = xc.f * V_down**2 / (2 * xc.g * D_H_down)
            dx = self.x_arr[i + 1] - self.x_arr[i]
            if self.fd_mids[i + 1] > 0:
                # Use depth from previous timestep if available
                fd_guess = self.fd_mids[i + 1]
            else:
                # Otherwise, use downstream flow depth
                fd_guess = self.fd_mids[i]
            norm_fd = xc.calcNormalFlowDepth(self.Q_w, self.slopes[i])
            fd_crit = xc_up.calcCritFlowDepth(self.Q_w)
            # debugpy.breakpoint()
            # print(fd_guess)
            try:
                # Search for best bracket
                n_search = 10
                fd_search = np.linspace(
                    fd_guess * 1.5, 0.8 * min([fd_crit, norm_fd]), n_search
                )
                bracket_found = False
                sign_this_res = None
                sign_old_res = None
                j = 0
                while not bracket_found and j + 1 < len(fd_search):
                    this_res = self.fd_residual(
                        fd_search[j], i + 1, H_down, S_f_down, dx
                    )
                    # print("this_res =", this_res)
                    sign_this_res = np.sign(this_res)
                    if sign_old_res is not None:
                        if sign_this_res * sign_old_res == -1:
                            # We have a sign change in residual
                            low_bracket = fd_search[j]
                            high_bracket = fd_search[j - 1]
                            bracket_found = True
                    sign_old_res = sign_this_res
                    j += 1
                # print("bracket found =", bracket_found)
                if not bracket_found:
                    low_bracket = fd_crit
                    high_bracket = fd_guess * 1.2

                sol = root_scalar(
                    self.fd_residual,
                    args=(i + 1, H_down, S_f_down, dx),
                    method="brenth",
                    x0=fd_guess,
                    bracket=(low_bracket, high_bracket),
                    xtol=0.00001,
                    rtol=0.00005,
                )
                is_converged = sol.converged
            except ValueError:
                print("Falling back on minimization solver.")
                is_converged = False
            #            sol = root_scalar(
            #                self.fd_residual,
            #                args=(i + 1, H_down, S_f_down, dx),
            #                x0=fd_guess,
            #                x1=0.9 * fd_guess,
            #            )

            if is_converged:
                fd_sol = sol.root
                flag = sol.flag
                converged = sol.converged
            else:
                # Try minimization of abs error
                # res = minimize_scalar(
                #    self.fd_residual_abs,
                #    bracket=(fd_crit, 1.1 * fd_guess),
                #    args=(i + 1, H_down, S_f_down, dx),
                # )
                fd_max = xc.ymax - xc.ymin
                res = shgo(
                    self.fd_residual_abs,
                    [
                        (fd_crit, fd_max),
                    ],
                    n=32,
                    sampling_method="sobol",
                    args=(i + 1, H_down, S_f_down, dx),
                )
                # converged = res.success
                # print("converged =", converged, "  fun=", res.fun)
                # print(res)
                # flag = "used minimization solver"
                fd_sol = res.x[0]
            # Calculate actual flow depth residual
            err = self.fd_residual(fd_sol, i + 1, H_down, S_f_down, dx)
            # print("i=", i, "  err=", err, " fd=", fd_sol)
            if abs(err) > WARN_ERR:
                print("*******************************************")
                print("Warning! Flow depth solution is inaccurate. Error is", err)
                print("*******************************************")

            xc_up.setFD(fd_sol)
            """print(
                "fd_sol =",
                fd_sol,
                "  fd_crit =",
                fd_crit,
                " converged =",
                converged,
                "  flag =",
                flag,
            )"""
            if fd_sol < fd_crit:
                # Force critical flow
                print("******************************")
                print("Enforcing critical flow at XC ", i)
                print("******************************")
                fd_sol = fd_crit

            self.h[i + 1] = self.z_arr[i + 1] + (fd_sol + self.fd_mids[i]) / 2
            if self.h[i + 1] < self.h[i]:
                print("Reversed head gradient!")
                # debugpy.breakpoint()
            self.fd_mids[i + 1] = fd_sol
            xc_up.setFD(fd_sol)
            # Extrapolate head at final upstream node
            if i == (len(self.xcs) - 2):
                zmid = (self.z_arr[i + 1] + self.z_arr[i + 2]) / 2
                dh_dx = ((self.fd_mids[i + 1] + zmid) - self.h[i + 1]) / (dx / 2)
                self.h[i + 2] = self.h[i + 1] + dh_dx * dx

        # Calculate flow areas, wetted perimeters, hydraulic diameters,
        # free surface widths, and velocities
        for i, xc in enumerate(self.xcs):
            # print("i =", i, "  fd=", self.fd[i])
            self.A_w[i] = xc.calcA(depth=self.fd_mids[i])
            self.P_w[i] = xc.calcP(depth=self.fd_mids[i])
            self.V_w[i] = -self.Q_w / self.A_w[i]
            self.D_H_w[i] = 4 * self.A_w[i] / self.P_w[i]
            if self.fd_mids[i] < xc.ymax - xc.ymin:
                L, R = xc.findLR(self.fd_mids[i])
                self.W[i] = xc.x[R] - xc.x[L]
            else:
                self.W[i] = 0.0
            if xc.n_mann is not None:
                xc.set_f_from_n_mann(self.D_H_w[i])
            S_f = xc.f * self.V_w[i] ** 2 / (2 * xc.g * self.D_H_w[i])
            xc.setEnergySlope(S_f)

    def fd_residual(self, fd_guess, xc_up_idx, H_down, S_f_down, dx):
        """Calculate residual between guessed upstream flow depth and energy
           equation flow depth.

        Parameters
        ----------
        fd_guess : float
            Guessed upstream flow depth.
        xc_idx : int
            Index of current downstream cross-section.
        H_down : float
            Head at downstream cross-section.
        S_f_down : float
            Friction slope at downstream cross-section
        dx : float
            Distance between cross-sections.
        """
        xc_up = self.xcs[xc_up_idx]
        A_up = xc_up.calcA(depth=fd_guess)
        P_up = xc_up.calcP(depth=fd_guess)
        if A_up < SMALL:
            A_up = SMALL
        if P_up < SMALL:
            P_up = SMALL
        V_up = self.Q_w / A_up
        V_head_up = alpha * V_up**2 / (2 * xc_up.g)
        D_H_up = 4 * A_up / P_up
        if xc_up.n_mann is not None:
            xc_up.set_f_from_n_mann(D_H_up)
        S_f_up = xc_up.f * V_up**2 / (2 * xc_up.g * D_H_up)
        H_up_energy = H_down + 0.5 * (S_f_down + S_f_up) * dx

        fd_up_energy = (
            H_up_energy
            - V_head_up
            - (self.z_arr[xc_up_idx] + self.z_arr[xc_up_idx + 1]) / 2
        )
        err = fd_up_energy - fd_guess
        return err

    def fd_residual_abs(self, fd_guess, xc_up_idx, H_down, S_f_down, dx):
        fd_res = self.fd_residual(fd_guess, xc_up_idx, H_down, S_f_down, dx)
        return abs(fd_res)


class spim(sim):
    """Simulation for channel profile evolution using the stream power incision model.
    
    Parameters
        ----------
        x_arr : ndarray
            Array of distances in meters along the channel for the node locations.
        z_arr : ndarray
            Array of elevations in meters for nodes along the channel. Minimum y
            values for each cross-section will be added to these elevations
            during initialization, so that z_arr will represent the channel bottom.
        uplift : float or list of floats
            Rate of change of baselevel. This distance is subtracted
            from the elevation of the downstream boundary node during
            each timestep.
        uplift_times : list
            Times in years at which uplift rates change. This argument is included if
            dz0_dt is a list of different uplift rates.
        Q_w : float, optional
            Discharge in the channel (m^3/s). Default is 0.1 m^3/s.
        dt_erode : float, optional
            Erosional time step in years. Default value is 1 year.
        a : float, optional
            Exponent in power law erosion rule (default=1).
        K : float or list, optional
            Erodibility in power law erosion rule (default = 1e-5).
            If multiple layers are specified, then this is a list of
            erodibilities listed from lowest to highest elevation.
        layer_elevs : list of floats, optional
            Specifies a list of elevations (from low to high), where rock
            erodibility changes. If specified, K should be a list with
            one more item than this list.
        """

    def __init__(
        self,
        x_arr,
        z_arr,
        uplift=0.0,
        uplift_times=None,
        Q_w=0.1,
        dt_erode=1.0,
        a=1.0,
        K=1e-5,
        layer_elevs=None,
        MIN_SLOPE=1e-8,
    ):

        super(spim, self).__init__()
        self.n_nodes = x_arr.size
        self.L = x_arr.max() - x_arr.min()
        self.x_arr = x_arr
        self.dx = x_arr[1] - x_arr[0]
        self.z_arr = z_arr
        self.MIN_SLOPE = MIN_SLOPE
        self.updateSlopes()
        self.Q_w = Q_w
        self.dt_erode = dt_erode
        self.old_dt = dt_erode
        self.a = a
        self.n = (2.0 / 3.0) * a
        self.K = K
        self.uplift = uplift
        self.uplift_times = uplift_times
        if uplift_times is not None:
            self.uplift_idx = 0
        self.dz = 0.0
        if layer_elevs is not None:
            self.set_layers(layer_elevs)
            self.K_arr = np.zeros(self.n_nodes)
            self.updateKs()
        else:
            self.layered_sim = False
            self.K_arr = K * np.ones(self.n_nodes)

    def run_one_step(self):
        """Run one time step of simulation.

        Calculates erosion for a single time step and updates geometry.

        """

        self.elapsed_time += self.dt_erode
        self.timestep += 1
        self.erode()
        self.apply_uplift()
        self.updateSlopes()
        if self.layered_sim:
            self.updateKs()
        # Set old_dt for plotting erosion rates
        self.old_dt = self.dt_erode

    def erode(self):
        """Erode the cross-section.
        """
        self.dz = -self.K_arr[1:] * self.slopes**self.n * self.dt_erode
        # erosion = self.dt_erode * self.dz
        self.z_arr[1:] += self.dz  # erosion

    def updateKs(self):
        """Update bedrock erodibility based on new elevations."""
        old_elev = None
        for i, elev in enumerate(self.layer_elevs):
            if i == 0:
                layer_idx = self.z_arr < elev
            else:
                layer_idx = np.logical_and(self.z_arr < elev, self.z_arr >= old_elev)
            self.K_arr[layer_idx] = self.K[i]
            old_elev = elev
        final_layer_idx = self.z_arr > elev
        self.K_arr[final_layer_idx] = self.K[-1]

    def updateSlopes(self):
        "Update channel slopes based on new elevations"
        self.slopes = (self.z_arr[1:] - self.z_arr[:-1]) / self.dx
        self.slopes[self.slopes < 0] = self.MIN_SLOPE
