"""Code to run simulations of single cross-sections."""

import numpy as np
from scipy.signal import savgol_filter

from olm.calcite import concCaEqFromPCO2, createPalmerInterpolationFunctions, palmerRate, calc_K_H,\
                        solutionFromCaPCO2, palmerFromSolution
from olm.general import CtoK

from crossSection import CrossSection
from ShapeGen import genCirc, genEll



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


class single_XC:
    def __init__(self, init_radius=1., Q_w=1., slope=0.001, impure=True,
        reduction_factor=0.01, dt_erode=1., pCO2_atm=5000*1e-6, Ca=0.5, f=0.1, T_cave = 10,
        xc_n = 1500):

        """
        Parameters
        ----------
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
        xc_n : int, optional
            Number of points that will define the cave passage shape within a cross-section.
            Default is 1000.
        slope : float
            Prescribed channel slope. Default is 0.001.
        pCO2_atm : float, optional
            Dissolved pCO2 value in atm. Default is 5e-3 atm (5000 ppm).
        Ca : float, optional
            Dissolved Ca concentration expressed
            as a fraction of saturation at a pCO2 of pCO2_atm. Default is 0.5.
        T_cave : float, optional
            Cave temperature (air and water) in degrees C. Default is 10 C.
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
        """

        self.init_radius = init_radius
        self.Q_w = Q_w
        self.slope = slope
        self.impure = impure
        self.reduction_factor = reduction_factor
        self.dt_erode = dt_erode
        self.pCO2_atm = pCO2_atm
        self.Ca = Ca
        self.f = f
        self.T_cave = T_cave
        self.T_cave_K = CtoK(T_cave)
        self.xc_n = xc_n

        self.K_H = calc_K_H(self.T_cave_K) #Henry's law constant mols dissolved per atm
        self.Ca_eq_0 = concCaEqFromPCO2(self.pCO2_atm, T_C=T_cave)
        self.palmer_interp_funcs = createPalmerInterpolationFunctions(impure=impure)

        x, y = genCirc(init_radius, n=xc_n)
        self.xc = CrossSection(x,y)

    def calc_flow_depth(self):
        """Calculate flow depth."""
        old_fd = self.xc.fd
        self.xc.create_A_interp()
        self.xc.create_P_interp()
        norm_fd = self.xc.calcNormalFlowDepth(self.Q_w, self.slope, f=self.f, old_fd=old_fd)
        if norm_fd == -1:
            #pipefull
            delh = self.xc.calcPipeFullHeadGrad(self.Q_w, f=self.f)
            self.xc.setEnergySlope(delh)
        else:
            self.xc.setEnergySlope(self.slope)

    def erode_xc(self):
        """Erode cross-section according to transport-limited equation."""
        #if palmer:
        #    this_Ca = self.Ca*self.Ca_eq_0
        #    sol = solutionFromCaPCO2(this_Ca, self.pCO2_atm, T_C=self.T_cave)
        #    E = palmerFromSolution(sol)
        #    xc.dr = np.ones(self.xc_n)*
        #else:
        self.xc.setMaxVelPoint(self.xc.fd)
        self.xc.calcUmax(self.Q_w)
        T_b = self.xc.calcT_b()
        eps = 5*nu*Sc**(-1./3.)/np.sqrt(T_b/rho_w)
        F_xc = self.reduction_factor*D_Ca/eps*(1. - self.Ca)*self.Ca_eq_0*L_per_m3
        #Smooth F_xc with savgol_filter
        window = int(np.ceil(len(F_xc)/5)//2*2+1)
        F_xc = savgol_filter(F_xc,window,3)
        self.xc.set_F_xc(F_xc)
        F_to_m_yr = g_mol_CaCO3*secs_per_year/rho_limestone/cm_m**3
        self.xc.dr = F_to_m_yr*F_xc*self.dt_erode
        self.xc.erode(self.xc.dr)

    def erode_power_law(self, a=1., K=1e-5):
        """Erode cross-section according to power law."""
        self.xc.erode_power_law(a=a, K=K)
#        self.xc.setMaxVelPoint(self.xc.fd)
#        self.xc.calcUmax(self.Q_w)
#        T_b = self.xc.calcT_b()
#        print('max T_b=', T_b.max())
#        self.xc.dr = K*T_b**a
#        self.xc.erode(self.xc.dr)

    def erode_power_law_layered(self, a=1., K=[1e-5,2e-5], layer_elevs=[-2]):
        """Erode cross-section according to power law."""
        self.xc.erode_power_law_layered(a=a, K=K, layer_elevs=layer_elevs)
