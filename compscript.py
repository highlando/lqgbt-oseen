import numpy as np

import lqgbt_lnse
# import sys
import datetime

# to compute stabilizing initial values for higher Re numbers
relist = [None, 5.0e1]  # , 1.0e2, 1.1e2]  # , 1.5e2]
pymess = False
pymess = True

# the input regularization parameter
gamma = 1.  # e5
# mesh parameter for the cylinder meshes
cyldim = 2
# whether to do bccontrol or distributed
bccontrol = True
palpha = 1e-5  # parameter for the Robin penalization
# where to truncate the LQGBT characteristic values
trunclist = [1e-3]  # , 1e-3, 1e-2, 1e-1, 1e-0]
# dimension of in and output spaces
NU, NY = 3, 3
# to what extend we perturb the initial value
perturbpara = 1e-6
# closed loop def
closed_loop = False  # 'red_output_fb'  # None, 'red_output_fb'
# number of time steps -- also define the lag in the control application
scaletest = 0.6  # for 1. we simulate till 12.
t0, tE, Nts = 0.0, scaletest*12.0, np.int(scaletest*1*2.4e3+1)

nwtn_adi_dict = dict(adi_max_steps=450,
                     adi_newZ_reltol=1e-7,
                     nwtn_max_steps=30,
                     nwtn_upd_reltol=1e-10,
                     nwtn_upd_abstol=1e-7,
                     ms=[-2.0, -1.5, -1.25, -1.1, -1.0, -0.9, -0.7, -0.5],
                     verbose=True,
                     full_upd_norm_check=False,
                     check_lyap_res=False)

pymess_dict = dict(verbose=True, maxit=45, nwtn_res2_tol=4e-8, linesearch=True)

logstr = 'logs/log_cyldim{0}NU{1}NY{2}gamma{3}'.format(cyldim, NU, NY, gamma) +\
    'closedloop{0}'.format(closed_loop) +\
    't0{0}tE{1}Nts{2}'.format(t0, tE, Nts) +\
    'Re{2}to{3}kappa{0}to{1}eps{4}'.format(trunclist[0], trunclist[-1],
                                           relist[0], relist[-1], perturbpara)

if bccontrol:
    logstr = logstr + '_bccontrol_palpha{0}'.format(palpha)

# import dolfin_navier_scipy.data_output_utils as dou
# dou.logtofile(logstr=logstr)

print('{0}'*10 + '\n log started at {1} \n' + '{0}'*10).\
    format('X', str(datetime.datetime.now()))

datadict = dict(problemname='cylinderwake', N=cyldim,
		bccontrol=bccontrol, palpha=palpha,
		gamma=gamma, NU=NU, NY=NY,
		t0=t0, tE=tE, Nts=Nts,
        pymess=pymess,
		nwtn_adi_dict=nwtn_adi_dict,
		plotit=False,
		paraoutput=True, multiproc=True, comp_freqresp=False,
		plain_bt=False, comp_stepresp=False, closed_loop=False,
		perturbpara=perturbpara)

# ## compute the projections
for ctrunc in trunclist:
    for cre in range(1, len(relist)):
        datadict.update(dict(use_ric_ini=relist[cre-1],
                             Re=relist[cre], trunc_lqgbtcv=ctrunc))
        lqgbt_lnse.lqgbt(**datadict)

# ## do the simulation(s)
for ctrunc in trunclist:
    datadict.update(dict(use_ric_ini=relist[-2], closed_loop=closed_loop,
		         Re=relist[-1], trunc_lqgbtcv=ctrunc))
    lqgbt_lnse.lqgbt(**datadict)
