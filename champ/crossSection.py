from numpy import (
    sin,
    cos,
    fabs,
    sign,
    roll,
    logical_and,
    arange,
    where,
    linspace,
    sqrt,
)
import numpy as np
from scipy import interpolate
from scipy.optimize import root_scalar, brentq, minimize_scalar
from champ.utils.fastroutines import (
    calcA,
    calcP,
    rollm,
    rollp,
    fast1DCubicSpline as finterp1d,
)
import copy


# import debugpy

g = 9.8  # m/s^2
rho_w = 998.2  # kg/m^3
SMALL = 1e-6
use_centroid_fraction = (
    0.98  # switch to  max vel at centroid if over this fraction of ymax
)
trim_factor = 2.0  # Trim xc points with y above trim_factor*fd
add_factor = 1.75  # add xc points back in from total if ceiling less than add_factor*fd


class CrossSection:
    """Cross-section object that contains functions for calculating geometry
    and flow.

    """

    def __init__(self, x, y, z0=0.01 / 30.0, f=0.1, n_mann=None):
        """
        Parameters
        ----------
        x : ndarray
            Array of x values (horizontal position) of points that make up
            cross-section.
        y : ndarray
            Array of y values (vertical position) of points that make up cross-section.
        z0: float, optional
            Roughness height in meters. Default value is 1/30 of a cm.
        f: float, optional
            Darcy-Weisbach Friction factor. Default is 0.1.
        n_mann: float, optional
            Manning's n. If specified, then f will be calculated from n_mann
            and R_h during flow calculations (which will still use the Darcy-
            Weisbach equation). Default is None.
        """
        self.g = g
        self.n = len(x)
        self.x = x
        self.y = y
        self.rollXC()
        self.x_total = None
        self.y_total = None
        self.ymin = min(y)
        self.ymax = max(y)
        self.create_pm()
        self.z0 = z0
        self.f = f
        self.n_mann = n_mann
        self.setFD(self.ymax - self.ymin)
        self.setMaxVelPoint(self.fd)
        self.Q = 0.0
        self.back_to_total = False
        self.is_trimmed = False

    def rollXC(self):
        """
        Roll x and y points such that XC starts at maximum y value
        """
        ymaxidx = self.y.argmax()
        if self.x[ymaxidx] > 0:
            # Max point is on RHS and needs to be shifted to left
            idxs = arange(len(self.x))
            left_ys = self.y[self.x < 0]
            left_y_idxs = idxs[self.x < 0]
            max_left_y_idx = left_ys.argmax()
            ymaxidx = left_y_idxs[max_left_y_idx]

        y_roll = self.y.size - ymaxidx
        self.x = roll(self.x, y_roll)
        self.y = roll(self.y, y_roll)

    # Create arrays of x+1 (xp), y+1 (yp), x-1 (xm), x+1 (xp)
    def create_pm(self):
        """Create rolled arrays of adjacent points in cross-section."""
        self.xm = rollm(self.x)  # roll(self.x, 1)
        self.ym = rollm(self.y)  # roll(self.y, 1)
        self.xp = rollp(self.x)  # roll(self.x, self.x.size - 1)
        self.yp = rollp(self.y)  # roll(self.y, self.y.size - 1)
        if self.x_total is not None:
            self.xm_total = rollm(self.x_total)  # roll(self.x_total, 1)
            self.ym_total = rollm(self.y_total)  # roll(self.y_total, 1)
            self.xp_total = rollp(
                self.x_total
            )  # roll(self.x_total, self.x_total.size - 1)
            self.yp_total = rollp(
                self.y_total
            )  # roll(self.y_total, self.y_total.size - 1)

    def calcP(self, depth=-1.0, total=False):
        """Calculate perimeter of cross-section or subset. Calls cython for
        fast computation.

        Parameters
        ----------
        depth : float, optional
            Flow depth at which to calculate wetted perimeter. If set to -1
            then the whole cross-section is used.

        total: boolean, optional
            Whether to use the total or trimmed cross-section in the calc.
            Default is false.

        Returns
        -------
        P : float
            The perimeter of the selected portion of the cross-section.
        """
        # If we exceed depth of trimmed XC, then use total
        if (depth > self.ymax - self.ymin) and self.x_total is not None:
            total = True

        if total:
            return calcP(
                self.x_total, self.xp_total, self.y_total, self.yp_total, depth
            )
        else:
            return calcP(self.x, self.xp, self.y, self.yp, depth)

    # Calculates area of the cross-section
    def calcA(self, depth=-1.0, total=False):
        """Calculate area of cross-section or subset. Calls cython for
        fast computation.

        Parameters
        ----------
        depth : float, optional
            Flow depth at which to calculate area. If set to -1
            then the whole cross-section is used.

        total: boolean, optional
            Whether to use the total or trimmed cross-section in the calc.
            Default is false.


        Returns
        -------
        A : float
            The area of the selected portion of the cross-section.
        """
        # If we exceed depth of trimmed XC, then use total
        if (depth > self.ymax - self.ymin) and self.x_total is not None:
            total = True

        if total:
            return calcA(
                self.x_total, self.xp_total, self.y_total, self.yp_total, depth
            )
        else:
            return calcA(self.x, self.xm, self.y, self.ym, depth)

    def create_A_interp(self, n_points=30):
        """Create an interpolation function for area as a function of flow depth.

        Parameters
        ----------
        n_points : int, optional
            Number of points along which to interpolate. Default value is 30.
        """
        maxdepth = self.ymax - self.ymin
        max_interp = self.fd * 1.5
        if max_interp > maxdepth:
            max_interp = maxdepth

        num_xc_points = len(self.y[self.y - self.ymin < max_interp])
        if num_xc_points < n_points / 3.0:
            n_points = int(np.round(num_xc_points / 3.0))

        depth_arr = np.linspace(0, max_interp, n_points)

        As = np.array([self.calcA(depth=i) for i in depth_arr])

        A_interp = finterp1d(
            depth_arr, As, bounds_error=False, fill_value=(As[0], As[-1])
        )

        self.A_interp = A_interp

    def create_P_interp(self, n_points=30):
        """Create an interpolation function for perimeter as a function of flow depth.

        Parameters
        ----------
        n_points : int, optional
            Number of points along which to interpolate. Default value is 30.
        """
        maxdepth = self.ymax - self.ymin
        max_interp = self.fd * 1.5
        if max_interp > maxdepth:
            max_interp = maxdepth

        num_xc_points = len(self.y[self.y - self.ymin < max_interp])
        if num_xc_points < n_points / 3.0:
            n_points = int(np.round(num_xc_points / 3.0))

        depth_arr = np.linspace(0, max_interp, n_points)

        Ps = np.array([self.calcP(depth=i) for i in depth_arr])

        P_interp = finterp1d(
            depth_arr, Ps, bounds_error=False, fill_value=(Ps[0], Ps[-1])
        )

        self.P_interp = P_interp

    def findLR(self, h):
        """Find the indicies of the left and right points defining the wall
        at a certain height above the bottom of the cross-section.

        Parameters
        ----------
        h : float
            Height above the bottom of the cross-section at which to find the
            left and right walls.

        Returns
        -------
        L,R : int
            Indicies of the left and right wall points.

        """
        a_h = self.ymin + h
        condL = logical_and(self.y > a_h, a_h >= self.yp)
        condR = logical_and(self.y < a_h, a_h <= self.yp)
        L = where(condL)[0][0] + 1
        R = where(condR)[0][0]
        return L, R

    def setFD(self, fd):
        """Set the flow depth within the cross-section."""
        if self.y_total is None:
            maxdepth = self.ymax - self.ymin
        else:
            maxdepth = self.y_total.max() - self.ymin
        if fd > maxdepth:
            fd = maxdepth
        if fd > self.ymax - self.ymin:
            self.update_total_xc(self.x, self.y)
            self.switchToTotalXC()
        self.fd = fd
        self.wetidx = self.y - self.ymin <= fd

    def setMaxVelPoint(self, fd):
        """Set the maximum velocity point."""

        self.setFD(fd)
        if fd > (self.ymax - self.ymin) * use_centroid_fraction:
            # Treat as full pipe
            mx, my = self.findCentroid()
        else:
            # open channel
            L, R = self.findLR(fd)
            mx = 0
            my = self.y[R]

        self.xmaxVel = mx
        self.ymaxVel = my

    def findCentroid(self):
        """Find the centroid of the cross-section.

        Returns
        -------
        cx, cy : float
            The x and y coordinates of the centroid.
        """
        m = self.xm * self.y - self.x * self.ym
        A = self.calcA()
        cx = (1 / (6 * A)) * (
            (self.x + self.xm) * m
        ).sum()  # A was self.sA. not sure if this matters
        cy = (1 / (6 * A)) * ((self.y + self.ym) * m).sum()
        return cx, cy

    def calcR_l(self, wantidx=None):
        """Calculate radial distances between wall and max velocity point.

        Parameters
        ----------
        wantidx : ndarray of boolean
            An array of boolean values to select cross-section points for
            which to calculate distance. Normally, this will be the wet
            portion of the cross-section. If None, then distances are
            calculated for all cross-section wall points. Default is None.

        Returns
        -------
        r_l : ndarray
            Array of radial distances from max velocity point to wall.
        """
        if wantidx is not None:
            self.r_l = np.hypot(
                self.x[wantidx] - self.xmaxVel, self.y[wantidx] - self.ymaxVel
            )
        else:
            self.r_l = np.hypot(self.x - self.xmaxVel, self.y - self.ymaxVel)

        return self.r_l

    # Find the value of U_max by weighted law of the wall
    def calcUmax(self, Q, method="area"):
        """Calculate maximum velocity using law of the wall.

        Parameters
        ----------
        Q : float
            Discharge through cross-section.
        method : string, optional
            Valid options include 'area' (default) and 'line'. This
            argument determines whether velocities are weighted linearly
            along the rays or by area.
        """
        wetidx = self.wetidx  # self.y-self.ymin <self.fd
        r_l = self.calcR_l(wantidx=wetidx)
        z0 = self.z0
        Ax = self.xmaxVel
        Ay = self.ymaxVel
        Bx = roll(self.x[wetidx], 1)
        By = roll(self.y[wetidx], 1)
        Cx = self.x[wetidx]
        Cy = self.y[wetidx]

        a_top = Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By)
        a_i = fabs(a_top / 2.0)

        if method == "line":
            u_i = r_l / (r_l - z0) - 1 / np.log(r_l / z0)

        if method == "area":
            u_top = (
                r_l**2 * np.log(r_l / z0)
                - 3.0 / 2 * r_l**2
                + 2.0 * r_l * z0
                - z0**2 / 2.0
            )
            u_bottom = z0**2 + r_l**2 - 2 * z0 * r_l
            u_i = 1.0 / np.log(r_l / z0) * u_top / u_bottom

        self.umax = Q / (a_i * u_i).sum()

    def calcT_b(self):
        """Calculates boundary shear stress within wet cross-section."""
        vgrad2 = self.calcVGrad2()
        # print('max vgrad2=',vgrad2.max())
        psi = self.calcPsi()
        Awet = self.calcA(depth=self.fd)
        self.T_b = psi * rho_w * Awet * vgrad2
        return self.T_b

    def calcVGrad2(self):
        """Calculates the square of the velocity gradient at the roughness height."""
        wetidx = self.wetidx
        phi = np.arctan2(self.ymaxVel - self.y[wetidx], self.xmaxVel - self.x[wetidx])
        alpha = np.arctan2(
            self.yp[wetidx] - self.ym[wetidx], self.xp[wetidx] - self.xm[wetidx]
        )
        alpha[0] = alpha[1]
        alpha[-1] = alpha[-2]
        self.vgrad2 = (
            (self.umax / self.z0)
            * (1.0 / np.log(self.r_l / self.z0))
            * np.fabs(np.sin(phi - alpha))
        ) ** 2.0
        return self.vgrad2

    def calcPsi(self):
        """Calculates dimensionless force-balancing scale factor."""
        wetidx = self.wetidx
        l = np.hypot(self.x[wetidx] - self.xp[wetidx], self.y[wetidx] - self.yp[wetidx])
        sum = ((self.vgrad2) * l).sum()
        self.wet_ls = l
        self.psi = g * self.eSlope / sum
        return self.psi

    def setEnergySlope(self, slope):
        """Sets the energy slope within the cross-section."""
        self.eSlope = slope

    def erode_power_law(self, a=1.0, K=1e-5, dt=1.0):
        """Erode wall according to a power law function of shear stress.

        Parameters
        ----------
        a : float, optional
            Exponent in the erosion power law (default is a=1).
        K : float, optional
            Multiplicative constant in the power law (default is K=1e-5).
        dt : float, optional
            Timestep for erosion (in years). Default is 1 year.
        Notes
        -----
        Erodes the wall according to:
        .. math:: E = K \tau_b^a
        """
        self.setMaxVelPoint(self.fd)
        self.calcUmax(self.Q)
        T_b = self.calcT_b()
        self.dr = dt * K * T_b**a
        self.erode(self.dr)

    def erode_power_law_layered(self, a=1.0, dt=1.0, K=[1e-5, 2e-5], layer_elevs=[-2]):
        """Erode wall according to a power law function of shear stress with erodibility
        varying by elevation.

        Parameters
        ----------
        a : float, optional
            Exponent in the erosion power law (default is a=1).
        K : list, optional
            List containing multiplicative constants in the power law.
        layer_elevs : list, optional
            List containing elevations where layer erodibilities change.
        dt : float, optional
            Timestep for erosion (in years). Default is 1 year.
        Notes
        -----
        Erodes the wall according to:
        .. math:: E = K \tau_b^a
        """
        self.setMaxVelPoint(self.fd)
        self.calcUmax(self.Q)
        T_b = self.calcT_b()
        ywet = self.y[self.wetidx]
        self.dr = np.zeros(len(ywet))
        old_elev = None
        for i, elev in enumerate(layer_elevs):
            if i == 0:
                layer_idx = ywet < elev
            else:
                layer_idx = logical_and(ywet < elev, ywet >= old_elev)
            # print('i=',i, '  len(layer_idx)=',len(layer_idx[layer_idx==True]))
            self.dr[layer_idx] = dt * K[i] * T_b[layer_idx] ** a
            old_elev = elev
        final_layer_idx = ywet > elev
        # print('len(final_layer_idx)=',len(final_layer_idx[final_layer_idx==True]))

        self.dr[final_layer_idx] = dt * K[-1] * T_b[final_layer_idx] ** a
        # self.dr = K*T_b**a
        self.erode(self.dr)

    def update_total_xc(self, nx, ny):
        """Updates total cross-section to include newest part of the actively
        evolving cross-section.

        Parameters
        ----------
        nx, ny : ndarray
            The new x and y points within the evolved cross-section.

        """
        n = self.n
        # create new total xc arrays from old and wet portions
        self.x1 = x1 = self.x_total[
            np.logical_and(self.x_total < 0, self.y_total > ny.max())
        ]
        self.y1 = y1 = self.y_total[
            np.logical_and(self.x_total < 0, self.y_total > ny.max())
        ]
        # Slightly trim high-res XC to remove any connection across top
        self.x2 = x2 = nx[ny < ny.max() - 0.05 * (ny.max() - ny.min())]
        self.y2 = y2 = ny[ny < ny.max() - 0.05 * (ny.max() - ny.min())]
        self.x4 = x4 = self.x_total[
            np.logical_and(self.x_total > 0, self.y_total > ny.max())
        ]
        self.y4 = y4 = self.y_total[
            np.logical_and(self.x_total > 0, self.y_total > ny.max())
        ]
        x_total_tmp = np.concatenate([x1, x2, x4])
        y_total_tmp = np.concatenate([y1, y2, y4])
        # debugpy.breakpoint()
        tck, u = interpolate.splprep([x_total_tmp, y_total_tmp], u=None, k=1, s=0.0)
        un = linspace(u.min(), u.max(), n)  # if n!=nx.size else nx.size)
        self.x_total, self.y_total = interpolate.splev(un, tck, der=0)

    def erode(self, dr, resample=True, n=None, trim=True):
        """Erodes the cross-section radially by specified distance.

        Parameters
        ----------
        dr : ndarray of float
            Specifies the erosion distances for each cross-section point.
        resample : boolean, optional
            Whether to resample the x, y coordinates after erosion.
            Default is True.
        n : int
            Number of points that should be in resampled cross-section.
            Default value is None, in which case self.n is used.
        trim : boolean
            Whether to enable trimming of cross-section with substantial
            fraction above the water level. Default is True.
        """
        if n is None:
            n = self.n
        wetidx = self.wetidx
        theta = np.arctan2(self.xp - self.xm, self.yp - self.ym)
        nx = self.x
        ny = self.y
        nx[wetidx] = self.x[wetidx] + dr * cos(theta[wetidx])
        ny[wetidx] = self.y[wetidx] - dr * sin(theta[wetidx])

        # Once flow drops far enough below ceiling, trim XC
        tmp_ymin = min(ny)
        trim_y = self.fd * trim_factor + tmp_ymin

        if self.back_to_total:
            if self.fd < 0.1 * (max(ny) - min(ny)):
                self.back_to_total = False

        if trim and not self.back_to_total:
            if trim_y < max(ny):
                # Initialize total xc arrays if first trimming event
                if self.x_total is None:
                    first_trim = True
                    # Roll XC positions so that arrays start a top in upper left quad
                    max_y_idx = np.argmax(ny)
                    if nx[max_y_idx] > 0:
                        start_found = False
                        while not start_found:
                            max_y_idx += 1
                            if max_y_idx == len(ny):
                                max_y_idx = 0
                            if nx[max_y_idx] < 0:
                                start_found = True
                    # Roll top point in XC to start of array
                    nx = roll(nx, -max_y_idx)
                    ny = roll(ny, -max_y_idx)
                    self.x_total = nx
                    self.y_total = ny
                    self.is_trimmed = True
                else:
                    first_trim = False
                nx = nx[ny < trim_y]
                ny = ny[ny < trim_y]
                if not first_trim:
                    self.update_total_xc(nx, ny)
            elif not trim_y < max(ny) and self.is_trimmed:
                # Water level is increasing
                self.update_total_xc(nx, ny)
                if (max(ny) - min(ny)) < add_factor * self.fd:
                    self.switchToTotalXC()
                    resample = False
                    nx = self.x
                    ny = self.y

        # Resample points by fitting spline
        if resample:
            # s = nx.size#+np.sqrt(2*nx.size)
            tck, u = interpolate.splprep([nx, ny], u=None, k=1, s=0.0)
            un = linspace(u.min(), u.max(), n)  # if n!=nx.size else nx.size)
            nx, ny = interpolate.splev(un, tck, der=0)

        # Set new XC coordinates
        self.x = nx
        self.y = ny
        self.rollXC()
        self.create_pm()
        self.ymin = min(ny)
        # Use LHS ymax rather than total. This works because of roll.
        self.ymax = ny[0]  # max(ny)
        self.n = len(nx)

    def switchToTotalXC(self):
        # Switch to using total
        print("########################################")
        print("############ Switch! #############")
        print("########################################")
        self.x = copy.deepcopy(self.x_total)
        self.y = copy.deepcopy(self.y_total)
        self.x_total = None
        self.y_total = None
        # Set new XC coordinates
        self.rollXC()
        self.create_pm()
        self.ymin = min(self.x)
        # Use LHS ymax rather than total. This works because of roll.
        self.ymax = self.y[0]  # max(self.y)
        self.n = len(self.x)
        self.back_to_total = True

    def calcNormalFlow(self, depth, slope, use_interp=True):
        """Calculate normal discharge for given depth and slope.

        Parameters
        ----------
        depth : float
            Flow depth for which to calculate discharge.
        slope : float
            Channel slope for which to calculate discharge.
        use_interp : boolean, optional
            Whether to use the area and perimeter interpolation functions
            rather than the cross-section wall points in calculating normal
            flow. This helps smooth the root-finding. Default value is True.

        """
        if use_interp:
            Pw = self.P_interp(depth)
            A = self.A_interp(depth)
        else:
            Pw = self.calcP(depth=depth)
            A = self.calcA(depth=depth)
        if Pw > 0 and A > 0 and depth > 0:
            D_H = 4.0 * A / Pw
            if self.n_mann is not None:
                self.set_f_from_n_mann(D_H)
            Q = sign(slope) * A * sqrt(2.0 * g * abs(slope) * D_H / self.f)
        else:
            Q = 0.0
        return Q

    def set_f_from_n_mann(self, D_H):
        R_H = D_H / 4
        f = 8 * g * self.n_mann**2 / (R_H ** (1 / 3))
        self.f = f

    def normal_discharge_residual(self, depth, slope, desiredQ):
        return desiredQ - self.calcNormalFlow(depth, slope)

    def abs_normal_discharge_residual(self, depth, slope, desiredQ):
        return np.abs(desiredQ - self.calcNormalFlow(depth, slope))

    def head_discharge_residual(self, y_in, y_out, L, slope, desiredQ):
        avg_flow_depth = (y_out + y_in) / 2.0
        head_slope = (y_in - y_out) / L + slope
        return desiredQ - self.calcNormalFlow(avg_flow_depth, head_slope)

    def crit_flow_depth_residual(self, depth, Q):
        A = self.A_interp(depth)
        if A < SMALL:
            A = SMALL
        L, R = self.findLR(depth)
        W = self.x[R] - self.x[L]
        if W < SMALL:
            W = SMALL
        return A**3 / W - Q**2 / g

    def abs_crit_flow_depth_residual(self, depth, Q):
        A = self.A_interp(depth)
        if A < SMALL:
            A = SMALL
        L, R = self.findLR(depth)
        W = self.x[R] - self.x[L]
        if W < SMALL:
            W = SMALL
        return abs(A**3 / W - Q**2 / g)

    def calcNormalFlowDepth(self, Q, slope, old_fd=None):
        """Calculate flow depth for a prescribed discharge.

        Parameters
        ----------
        Q : float
            Prescribed discharge.
        slope : float
            Slope of channel bed.
        old_fd : float
            Previous flow depth. This will be used to restrict upper bound
            on calculated flow depth. Default is None, for which the max depth
            will be used as upper range.

        Returns
        -------
        fd : float
            Flow depth required for prescribed discharge. -1 is returned if
            the flow depth exceeds the maximum possible depth in the
            cross-section.

        Notes
        -----
        This function will also set the cross-section flow depth to the
        calculated value.

        """
        self.Q = Q
        maxdepth = self.ymax - self.ymin
        if old_fd is None:
            upper_bound = maxdepth
        else:
            upper_bound = old_fd * 1.1  # 25
        calcFullFlow = self.calcNormalFlow(maxdepth, slope, use_interp=False)
        if Q >= calcFullFlow and not self.ymax > self.y.max():
            return -1
        else:
            SMALL_Q = self.normal_discharge_residual(SMALL, slope, Q)
            upper_bound_Q = self.normal_discharge_residual(upper_bound, slope, Q)
            if np.sign(SMALL_Q) != np.sign(upper_bound_Q):
                sol = brentq(
                    self.normal_discharge_residual,
                    a=SMALL,
                    b=upper_bound,
                    args=(slope, Q),
                    full_output=False,
                )
                fd = sol
            else:
                sol = minimize_scalar(
                    self.abs_normal_discharge_residual,
                    bounds=[SMALL, upper_bound],
                    args=(slope, Q),
                    method="bounded",
                )
                fd = sol.x

            if fd >= maxdepth:
                self.setFD(fd)
                return -1

        self.setFD(fd)
        return self.fd

    def calcCritFlowDepth(self, Q):
        """Calculate depth for critical flow at prescribed discharge.

        Parameters
        ----------
        Q : float
            Prescribed discharge.

        Returns
        -------
        crit_depth : float
            The critical flow depth for prescribed discharge.

        """
        maxdepth = self.ymax - self.ymin
        fd = self.fd
        upper_bound = fd * 1.25
        if upper_bound > 0.99 * maxdepth:
            upper_bound = 0.99 * maxdepth
        if np.sign(self.crit_flow_depth_residual(SMALL, Q)) != np.sign(
            self.crit_flow_depth_residual(
                upper_bound, Q
            )  # Edited Max's sign check, as it didn't make sense. Hopefully this is correct.
        ):
            sol = brentq(
                self.crit_flow_depth_residual,
                a=SMALL,
                b=upper_bound,
                args=(Q,),
                full_output=False,
            )
            crit_depth = sol
        else:
            sol = minimize_scalar(
                self.abs_crit_flow_depth_residual,
                bounds=[SMALL, upper_bound],
                args=(Q,),
                method="bounded",
            )
            crit_depth = sol.x

        return crit_depth

    def calcPipeFullHeadGrad(self, Q):
        """Calculate head gradient for prescribed discharge under pipe-full
        conditions.

        Parameters
        ----------
        Q : float
            Prescribed discharge.

        Returns
        -------
        delh : float
            Head gradient.

        """
        self.Q = Q
        Pw = self.calcP()
        A = self.calcA()
        D_H = 4.0 * A / Pw
        if self.n_mann is not None:
            self.set_f_from_n_mann(D_H)
        return (Q**2 / A**2) * self.f / (2.0 * g * D_H)

    def calcUpstreamHead(self, Q, slope, y_out, L):
        """Calculate upstream head to drive prescribed discharge under
        partially backflooded conditions.

        Parameters
        ----------
        Q : float
            Discharge.
        slope : float
            Channel bed slope.
        y_out : float
            Flow depth at outlet.
        L : float
            Length of channel segment.

        Returns
        -------
        y_in : float
            Upstream flow depth to drive prescribed discharge. Returns -1
            if valid flow depth is not found.

        """
        maxdepth = self.ymax - self.ymin
        head_res_a = self.head_discharge_residual(SMALL, y_out, L, slope, Q)
        head_res_b = self.head_discharge_residual(maxdepth, y_out, L, slope, Q)
        if np.sign(head_res_a) * np.sign(head_res_b) == -1:
            sol = root_scalar(
                self.head_discharge_residual,
                args=(y_out, L, slope, Q),
                bracket=(SMALL, maxdepth),
            )  # x0=y_out, x1=maxdepth)#, bracket=(SMALL,maxdepth) )
            y_in = sol.root
            return y_in
        else:
            return -1

    def calcConvey(self, depth):
        """Calculate conveyance, K, for provided flow depth.

        Parameters
        ----------
        depth : float
            Flow depth.

        Returns
        -------
        Convey : float
            Conveyance = A * sqrt(2*g*D_H/f).


        """
        A = self.calcA(depth=depth)
        P = self.calcP(depth=depth)
        if self.n_mann is not None:
            self.set_f_from_n_mann(4 * A / P)
        Convey = A * sqrt(2 * g * (4 * A / P) / self.f)
        return Convey
