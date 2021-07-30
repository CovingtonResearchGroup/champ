import numpy as np
from numpy.testing import *

from co2_evo.CO2_sim_1D import CO2_1D
from olm.calcite import calc_K_H

g=9.8
L_per_m3 = 1000.

def test_calc_flow_depth():
    n=5
    x = np.linspace(0, 5000,n)
    slope = 0.001
    z = x*slope
    r = 1.*np.ones(n-1)
    Q = 0.98347
    sim = CO2_1D(x, z, init_radii=r, Q_w = Q, xc_n=1500)
    sim.calc_flow_depths()
    fds = sim.fd_mids
    fds_by_hand = 1.0*np.ones(n-1)
    assert_allclose(fds, fds_by_hand, rtol=0.011)

def test_calc_flow_depth_full_pipe():
    n=5
    x = np.linspace(0, 5000,n)
    slope = 0.001
    z = x*slope
    r = 1.*np.ones(n-1)
    Q = 5.
    f=0.1
    sim = CO2_1D(x, z, init_radii=r, Q_w = Q, f=f, xc_n=1500)
    sim.calc_flow_depths()
    del_h = Q**2./(np.pi*r[0]**2)**2 * (f/(2*g*2*r[0]))
    h_analytical = x*del_h + 1.
    assert_allclose(sim.h, h_analytical, rtol=0.001)

def test_calc_flow_depth_full_pipe_varied_r():
    n=5
    x = np.linspace(0, 5000,n)
    slope = 0.001
    z = x*slope
    r = 1.*np.linspace(0.5,1.5,n-1)
    Q = 5.
    f=0.1
    sim = CO2_1D(x, z, init_radii=r, Q_w = Q, f=f, xc_n=1500)
    sim.calc_flow_depths()
    del_h = Q**2./(np.pi*r**2)**2 * (f/(2*g*2*r))
    h_analytical = np.zeros(n)
    h_analytical[1:] = np.cumsum(del_h)*sim.L_arr
    h_analytical += 2*r[0] + sim.z_arr[0]
    assert_allclose(sim.h, h_analytical, rtol=0.001)


def test_calc_air_flow():
    n=5
    x = np.linspace(0, 5000,n)
    slope = 0.001
    z = x*slope
    r = 1.*np.ones(n-1)
    Q_low = 0.000001
    Q_half = 0.98347
    sim = CO2_1D(x, z, init_radii=r, Q_w = Q_half,
                T_cave = 10., dH=100., T_outside=20.,
                xc_n=1500)
    #Bring water up to half full (checked in prev test)
    sim.calc_flow_depths()
    sim.calc_air_flow()
    assert_approx_equal(sim.Q_a, -0.81233, significant=2)
    #Empty pipe
    sim.Q_w = Q_low
    sim.calc_flow_depths()
    sim.calc_air_flow()
    assert_approx_equal(sim.Q_a, -1.624661, significant=2)

def test_Ca_longitudinal_profile_open_system_limit():
    #Tests for exponential Ca profile in limit where PCO2~fixed
    #(that is, a short conduit)
    n=50
    L=1000.
    x = np.linspace(0, L,n)
    slope = 0.001
    z = x*slope
    r = 1.*np.ones(n-1)
    Q = 0.98347
    f=0.1
    Rf = 0.01
    sim = CO2_1D(x, z, init_radii=r, Q_w = Q, f=f, xc_n=1500,
                Ca_upstream=0.0, T_cave=10., T_outside=10.,
                reduction_factor = Rf)
    sim.calc_flow_depths()
    sim.calc_steady_state_transport()
    D_Ca = 10**-9#m^2/s
    nu = 1.3e-6#m^2/s at 10 C
    Sc = nu/D_Ca
    D_H = 4.*(np.pi*r[0]**2*0.5)/(np.pi*r[0])#half-filled circular conduit
    #Use process length scale from Covington et al. (2012) to calc solution
    lambda_d = np.sqrt(2)/2.*0.1**(-.5)*D_H*(5.*nu/(Rf*D_Ca*Sc**(1./3.)))
    Ca_ana = (1.-np.exp(-(L-x)/lambda_d))
    assert_allclose(sim.Ca, Ca_ana, rtol=0.0105)


def test_Ca_eq_closed_system_limit():
    #Tests whether downstream Ca concentration matches
    #analytical solution for closed system equilibrium
    #for a very long conduit that should be equilibrated.
    n=50
    L=200*1000.
    x = np.linspace(0, L,n)
    slope = 0.001
    z = x*slope
    r = 1.*np.ones(n-1)
    Q = 0.98347
    f=0.1
    Rf = 0.01
    sim = CO2_1D(x, z, init_radii=r, Q_w = Q, f=f, xc_n=1500,
                Ca_upstream=0.0, T_cave=10., T_outside=10.,
                reduction_factor = Rf)
    sim.calc_flow_depths()
    sim.calc_steady_state_transport()
    Ca_eq = sim.Ca[0]*sim.Ca_eq_0
    T_K = sim.T_cave_K
    K_H = calc_K_H(T_K)
    pCO2_in = sim.pCO2_high
    pCO2_closed = pCO2_in - Ca_eq/K_H #From dreybrodt et al. (2005)
    pCO2_downstream = sim.CO2_w[0]*sim.pCO2_high
    assert_allclose(pCO2_downstream, pCO2_closed, rtol=0.0001)

def test_cons_of_mass_in_carbonate_system():
    n=5
    L=1000.
    x = np.linspace(0, L,n)
    slope = 0.001
    z = x*slope
    r = 1.*np.ones(n-1)
    Q = 0.25
    f=0.1
    Rf = 0.01
    sim = CO2_1D(x, z, init_radii=r, Q_w = Q, f=f, xc_n=1500,
                Ca_upstream=0.0, T_cave=10., T_outside=20.,
                reduction_factor = Rf)
    sim.calc_flow_depths()
    #Test summer direction airflow
    sim.calc_air_flow()
    sim.calc_steady_state_transport()
    T_K = sim.T_cave_K
    K_H = calc_K_H(T_K)
    Ca_mol_L = sim.Ca*sim.Ca_eq_0
    dm_Ca = sim.Q_w*L_per_m3*(Ca_mol_L[:-1] - Ca_mol_L[1:])
    CO2_w_mol_L = sim.pCO2_high*sim.CO2_w*K_H
    CO2_a_mol_L = sim.pCO2_high*sim.CO2_a*K_H
    dm_a = sim.Q_a*L_per_m3*(CO2_a_mol_L[:-1] - CO2_a_mol_L[1:])
    dm_w = sim.Q_w*L_per_m3*(CO2_w_mol_L[:-1] - CO2_w_mol_L[1:])
    assert_allclose(dm_w, dm_a - dm_Ca, rtol=0.000001)
    #Test winter direction airflow
    sim.set_T_outside(0.)
    sim.calc_air_flow()
    sim.calc_steady_state_transport()
    T_K = sim.T_cave_K
    K_H = calc_K_H(T_K)
    Ca_mol_L = sim.Ca*sim.Ca_eq_0
    dm_Ca = sim.Q_w*L_per_m3*(Ca_mol_L[:-1] - Ca_mol_L[1:])
    CO2_w_mol_L = sim.pCO2_high*sim.CO2_w*K_H
    CO2_a_mol_L = sim.pCO2_high*sim.CO2_a*K_H
    dm_a = sim.Q_a*L_per_m3*(CO2_a_mol_L[:-1] - CO2_a_mol_L[1:])
    dm_w = sim.Q_w*L_per_m3*(CO2_w_mol_L[:-1] - CO2_w_mol_L[1:])
    assert_allclose(dm_w, dm_a - dm_Ca, rtol=0.000001)

def test_co2_water_profile():
    #Test limit of high airflow (constant air pco2)
    #Dissolved CO2 should be approximately exponential
    # with analytically derived length scale.
    n=50
    L=5.*1000.
    x = np.linspace(0, L,n)
    slope = 0.001
    z = x*slope+1.
    r = 1.*np.ones(n-1)
    Q = 0.25
    f=0.1
    Rf = 0.01
    sim = CO2_1D(x, z, init_radii=r, Q_w = Q, f=f, xc_n=1500,
                Ca_upstream=1.0, T_cave=10., T_outside=-270.,
                reduction_factor = Rf)
    sim.calc_flow_depths()
    sim.calc_air_flow()
    sim.calc_steady_state_transport()
    K_w = sim.gas_transf_vel.mean()*sim.W.mean()/sim.A_w.mean()
    lambda_co2 = -sim.V_w.mean()/K_w
    CO2_ana = (sim.CO2_a[0] -(sim.CO2_a[0] - sim.CO2_w[-1] )*np.exp(-(L-x)/lambda_co2))
    assert_allclose(sim.CO2_w,CO2_ana, rtol=0.01)

def test_coupled_analytical_solution_airflow_summer():
    n=50
    L=10.*1000.
    x = np.linspace(0, L,n)
    slope = 0.001
    z = x*slope+1.
    r = 1.*np.ones(n-1)
    Q = 0.25
    f=0.1
    Rf = 0.0
    sim = CO2_1D(x, z, init_radii=r, Q_w = Q, f=f, xc_n=1500,
            Ca_upstream=1.0, T_cave=10., T_outside=20.,
            reduction_factor = Rf, variable_gas_transf=False)
    sim.calc_flow_depths()
    sim.calc_air_flow()
    sim.calc_steady_state_transport()
    Q_f = (-1./sim.Q_w + 1./sim.Q_a)
    lambda_co2 = 1./(sim.gas_transf_vel.mean()*sim.W.mean()*Q_f)
    CO2_w_ana = sim.CO2_w[-1] +((sim.CO2_w[-1] - sim.CO2_a[-1])/(-sim.Q_w*Q_f))*(np.exp((L-x)/lambda_co2)-1. )
    CO2_a_ana = CO2_w_ana - (sim.CO2_w[-1] - sim.CO2_a[-1])*np.exp((L-x)/lambda_co2)
    assert_allclose(sim.CO2_w,CO2_w_ana, rtol=0.001)
    assert_allclose(sim.CO2_a,CO2_a_ana, rtol=0.001)

def test_coupled_analytical_solution_airflow_winter():
    n=50
    L=10.*1000.
    x = np.linspace(0, L,n)
    slope = 0.001
    z = x*slope+1.
    r = 1.*np.ones(n-1)
    Q = 0.25
    f=0.1
    Rf = 0.0
    sim = CO2_1D(x, z, init_radii=r, Q_w = Q, f=f, xc_n=1500,
                Ca_upstream=1.0, T_cave=10., T_outside=0.,
                reduction_factor = Rf, variable_gas_transf=False)
    sim.calc_flow_depths()
    sim.calc_air_flow()
    sim.calc_steady_state_transport()
    Q_f = (-1./sim.Q_w + 1./sim.Q_a)
    Q_r = -Q_f*sim.Q_w
    lambda_co2 = 1./(sim.gas_transf_vel.mean()*sim.W.mean()*Q_f)
    #Use linear shooting method to get analytical solution
    CO2_a_L_g1 = 0.5
    CO2_a_L_g2 = 0.
    CO2_w_g1 = sim.CO2_w[-1] +((sim.CO2_w[-1] - CO2_a_L_g1)/(-sim.Q_w*Q_f))*(np.exp((L-x)/lambda_co2)-1. )
    CO2_a_g1 = CO2_w_g1 - (sim.CO2_w[-1] - CO2_a_L_g1)*np.exp((L-x)/lambda_co2)
    CO2_w_g2 = sim.CO2_w[-1] +((sim.CO2_w[-1] - CO2_a_L_g2)/(-sim.Q_w*Q_f))*(np.exp((L-x)/lambda_co2)-1. )
    CO2_a_g2 = CO2_w_g2 - (sim.CO2_w[-1] - CO2_a_L_g2)*np.exp((L-x)/lambda_co2)

    f = (CO2_a_g1[0] - sim.CO2_a[0])/(CO2_a_g1[0] - CO2_a_g2[0])

    CO2_a_ana = (1-f)*CO2_a_g1 + f*CO2_a_g2
    CO2_w_ana = (1-f)*CO2_w_g1 + f*CO2_w_g2
    assert_allclose(sim.CO2_w,CO2_w_ana, rtol=0.01)
    assert_allclose(sim.CO2_a,CO2_a_ana, rtol=0.01)

def test_subdivide_solution_winter():
    n=6
    L=25.*1000.
    x = np.linspace(0, L,n)
    slope = 0.001
    z = x*slope+1.
    r = 1.*np.ones(n-1)
    Q = 0.25
    f=0.1
    Rf = 0.0
    sim = CO2_1D(x, z, init_radii=r, Q_w = Q, f=f, xc_n=1500,
                Ca_upstream=1.0, T_cave=10., T_outside=0.,
                reduction_factor = Rf, variable_gas_transf=False)
    sim.calc_flow_depths()
    sim.calc_air_flow()
    sim.calc_steady_state_transport()

    n=51
    x = np.linspace(0, L,n)
    z = x*slope+1.
    r = 1.*np.ones(n-1)
    sim2 = CO2_1D(x, z, init_radii=r, Q_w = Q, f=f, xc_n=1500,
                Ca_upstream=1.0, T_cave=10., T_outside=0.,
                reduction_factor = Rf, variable_gas_transf=False)
    sim2.calc_flow_depths()
    sim2.calc_air_flow()
    sim2.calc_steady_state_transport()

    assert_allclose(sim.CO2_w,sim2.CO2_w[::10], rtol=0.02)
    assert_allclose(sim.CO2_a,sim2.CO2_a[::10], rtol=0.02)


def test_subdivide_solution_summer():
    n=6
    L=25.*1000.
    x = np.linspace(0, L,n)
    slope = 0.001
    z = x*slope+1.
    r = 1.*np.ones(n-1)
    Q = 0.25
    f=0.1
    Rf = 0.0
    sim = CO2_1D(x, z, init_radii=r, Q_w = Q, f=f, xc_n=1500,
                Ca_upstream=1.0, T_cave=10., T_outside=20.,
                reduction_factor = Rf, variable_gas_transf=False)
    sim.calc_flow_depths()
    sim.calc_air_flow()
    sim.calc_steady_state_transport()

    n=51
    x = np.linspace(0, L,n)
    z = x*slope+1.
    r = 1.*np.ones(n-1)
    sim2 = CO2_1D(x, z, init_radii=r, Q_w = Q, f=f, xc_n=1500,
                Ca_upstream=1.0, T_cave=10., T_outside=20.,
                reduction_factor = Rf, variable_gas_transf=False)
    sim2.calc_flow_depths()
    sim2.calc_air_flow()
    sim2.calc_steady_state_transport()

    assert_allclose(sim.CO2_w,sim2.CO2_w[::10], rtol=0.02)
    assert_allclose(sim.CO2_a,sim2.CO2_a[::10], rtol=0.02)
