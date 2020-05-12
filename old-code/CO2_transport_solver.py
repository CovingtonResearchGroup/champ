import numpy as np
from scipy.linalg import solve_banded
from olm.calcite import concCaEqFromPCO2, createPalmerInterpolationFunctions, palmerRate, calc_K_H,\
                        solutionFromCaPCO2, palmerFromSolution
from olm.general import CtoK



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
