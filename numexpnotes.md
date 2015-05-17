cyldim = 3, Nts = 4801
======================

Re 100
------
kappa = 1e-2, eps = 1e-3, Nts = 4.8e3+1
success

 to plot run the commands 

from dolfin_navier_scipy.data_output_utils import plot_outp_sig
plot_outp_sig("data/cylinderwake_Re100.0_NV19468NU3NY3_lqgbt__lqgbtcv0.01red_output_fbt00.0tE12.0Nts4801.0")

Re 150
------
kappa = 1e-2, eps = 1e-3, Nts = 4.8e3+1
success

Re 200
------
kappa = 1e-2, eps = 1e-3, Nts = 4.8e3+1
FAIL

kappa = 1e-3, eps = 1e-6, Nts = 4.8e3+1
FAIL 

kappa = 1e-3, eps = 1e-6, Nts = 9.6e3+1
success - note121 - (but with growing error)

from dolfin_navier_scipy.data_output_utils import plot_outp_sig
plot_outp_sig("data/cylinderwake_Re200.0_NV19468NU3NY3_lqgbt__lqgbtcv0.001red_output_fbt00.0tE12.0Nts9601.0inipert1e-06")


N=3, Re=200, conv history
---

# to compute stabilizing initial values for higher Re numbers
# relist = [None, 5.0e1, 1.0e2, 1.5e2, 2.0e2, 2.5e2, 3.0e2]  # , 3.5e2, 4.0e2]
relist = [None, 2.0e2]  # , 3.5e2, 4.0e2]

# mesh parameter for the cylinder meshes
cyldim = 3
# where to truncate the LQGBT characteristic values
trunclist = [1e-3]  # , 1e-3, 1e-2, 1e-1, 1e-0]
# dimension of in and output spaces
NU, NY = 3, 3
# to what extend we perturb the initial value
perturbpara = 1e-6
# closed loop def
closed_loop = 'red_output_fb'
# number of time steps -- also define the lag in the control application
t0, tE, Nts = 0.0, 12.0, 2*4.8e3+1

End: 12.0 -- now: 0.600000
norm of deviation 6.27768221557e-06
norm of actuation 1.22963184475e-11
End: 12.0 -- now: 0.601250
norm of deviation 6.27224837351e-06
norm of actuation 1.23687002708e-11

End: 12.0 -- now: 2.000000
norm of deviation 6.86865675143e-05
norm of actuation 8.56118848624e-10
End: 12.0 -- now: 2.001250
norm of deviation 6.86874445511e-05
norm of actuation 8.48798177641e-10
