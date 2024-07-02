from champ.sim import multiXCGVF
import numpy as np
import matplotlib.pyplot as plt
ft_per_m = 3.28
width = 20.0 / ft_per_m
height = 15.0 / ft_per_m
Q_cfs = 500.0
cumecs_per_cfs = 0.0283
Q = Q_cfs * cumecs_per_cfs
Manning_n = 0.015
L = 3000 / ft_per_m

shape_dict = {
    "name": "rectangular",
    "width": width,    
    "height": height,
}
n=150
x = np.linspace(0, L, n)
z = np.zeros(n)
for i, this_z in enumerate(z):
    if x[i]>2500/ft_per_m:
        bed_slope = 0.01
    elif x[i]<=2500/ft_per_m and x[i]>1000/ft_per_m:
        bed_slope = 0.0004
    else:
        bed_slope = 0.00317     
    if i>0:
        z[i] = +bed_slope * (x[i] - x[i-1]) + z[i-1]
z += 70/ft_per_m - z[-1]

sim = multiXCGVF(x, z, 
                 shape_dict=shape_dict, 
                 n_mann=Manning_n, 
                 Q_w=Q, 
                 mixed_regime=True, 
                 upstream_bnd_type='Normal',
                 xc_n=1500)

sim.calc_flow(h0=66/ft_per_m, )



#f = interp1d(sim.x_arr * ft_per_m, sim.fd * ft_per_m)
# X and flow depth values from Chow textbook example case
#x_chow = np.array(
#    [155, 318, 493, 684, 898, 1155, 1314, 1515, 1641, 1797, 1917, 2075, 2214, 2401]
#)
#fd_chow = np.array(
#    [4.8, 4.6, 4.4, 4.2, 4, 3.8, 3.7, 3.6, 3.55, 3.5, 3.47, 3.44, 3.42, 3.4]
#)
# Interpolate model outputs to Chow x positions
#fd_mod = f(x_chow)

