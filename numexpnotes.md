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
