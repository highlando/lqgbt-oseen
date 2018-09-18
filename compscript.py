import numpy as np

from lqgbt_oseen import lqgbt_lnse
# import sys
import datetime
import sys
import getopt

# to compute stabilizing initial values for higher Re numbers
pymess = True
pymess = False
# relist = [None, 5e1, 7.5e1, 1.e2]  # , 1.15e2, 1.25e2]  # 1.01e2]
relist = [None, 5e1, 7.5e1, 9.e1]  # , 1.e2]  # , 1.15e2, 1.25e2]  # 1.01e2]
# relist = [1.e2, 1.15e2, 1.25e2]  # 1.01e2]
max_re_only = False
max_re_only = True  # consider only the last Re for the simu

# the input regularization parameter
gamma = 1e-0  # e5
# mesh parameter for the cylinder meshes
# whether to do bccontrol or distributed
bccontrol = True
palpha = 1e-5  # parameter for the Robin penalization
cyldim = 3
# where to truncate the LQGBT characteristic values
trunclist = [1e-2]  # , 1e-2, 1e-1, 1e-0]
# dimension of in and output spaces
NU, NY = 3, 3
# to what extend we perturb the initial value
perturbpara = 1e-3
# whether we use a perturbed system
trytofail = False
trytofail = True
ttf_npcrdstps = 3
# whether to robustify the observer
robit = True
robit = False
robmrgnfac = 0.1
# whether to check the performance in the linear system
cl_linsys = True
cl_linsys = False
# closed loop def
closed_loop = 'redmod_sdre_fb'
closed_loop = 'red_sdre_fb'
closed_loop = False
closed_loop = 'full_state_fb'
closed_loop = None
closed_loop = 'red_output_fb'
closed_loop = 'hinf_red_output_fb'
# what inival
whichinival = 'sstokes'  # steady state Stokes solution
whichinival = 'sstokes++'  # a developed state starting from sstokes
whichinival = 'sstate+d'  # sstate plus perturbation
# number of time steps -- also define the lag in the control application
scaletest = 1.5  # .5  # for 1. we simulate till 12.
baset0, basetE, baseNts = 0.0, 12.0, 2.4e3+1
t0, tE, Nts = 0.0, scaletest*basetE, np.int(scaletest*baseNts)

# get command line input and overwrite standard paramters if necessary
options, rest = getopt.getopt(sys.argv[1:], '',
                              ['robit=',
                               'obsperturb=',
                               'ttf_npcrdstps=',
                               'robmrgnfac=',
                               'scaletest=',
                               'iniperturb=',
                               'closed_loop=',
                               'max_re_only='])
for opt, arg in options:
    if opt == '--robit':
        robit = int(arg)
        robit = np.bool(robit)
    elif opt == '--obsperturb':
        trytofail = int(arg)
        trytofail = np.bool(arg)
    elif opt == '--ttf_npcrdstps':
        ttf_npcrdstps = int(arg)
    elif opt == '--robmrgnfac':
        robmrgnfac = np.float(arg)
    elif opt == '--iniperturb':
        perturbpara = np.float(arg)
    elif opt == '--scaletest':
        scaletest = np.float(arg)
    elif opt == '--closed_loop':
        if np.int(arg) == -1:
                closed_loop = None
        elif np.int(arg) == 0:
                closed_loop = False
        elif np.int(arg) == 1:
                closed_loop = 'red_output_fb'
        elif np.int(arg) == 2:
                closed_loop = 'full_output_fb'
        elif np.int(arg) == 3:
                closed_loop = 'red_sdre_fb'
        elif np.int(arg) == 4:
                closed_loop = 'hinf_red_output_fb'
    elif opt == '--max_re_only':
            max_re_only = int(arg)
            max_re_only = np.bool(max_re_only)

print('max_re_only={0}'.format(max_re_only))
if max_re_only:
    relist = relist[-2:]

hinf = False  # hinf only need for reduced output feedback
if closed_loop == 'hinf_red_output_fb':
    closed_loop = 'red_output_fb'
    hinf = True

# print reynolds number and discretization lvl
infostring = ('Re             = {0}'.format(relist) +
              '\ncyldim         = {0}'.format(cyldim) +
              '\npymess         = {0}'.format(pymess) +
              '\nclosed_loop    = {0}'.format(closed_loop) +
              '\nH_infty        = {0}'.format(hinf) +
              '\ntrunc at       = {0}'.format(trunclist[0]) +
              '\nini_perturb    = {0}'.format(perturbpara) +
              '\nobs_perturb    = {0}'.format(trytofail) +
              '\nrobustification= {0}'.format(robit) +
              '\nrob margin fac = {0}'.format(robmrgnfac) +
              '\nttf_npcrdstps  = {0}'.format(ttf_npcrdstps) +
              '\nt0, tE, Nts    = {0}, {1}, {2}\n'.format(t0, tE, Nts) +
              '\nlinear cl sys  = {0}'.format(cl_linsys)
              )

print(infostring)

print(infostring)
if pymess:
    nwtn_adi_dict = dict(verbose=True, maxit=45, aditol=1e-8,
                         nwtn_res2_tol=4e-8, linesearch=True)
else:
    nwtn_adi_dict = dict(adi_max_steps=350,  # 450,
                         adi_newZ_reltol=2e-8,
                         nwtn_max_steps=30,
                         nwtn_upd_reltol=2e-8,
                         nwtn_upd_abstol=1e-7,
                         ms=[-100., -50., -10., -2.0, -1.3, -1.0, -0.9, -0.5],
                         verbose=True,
                         full_upd_norm_check=False,
                         check_lyap_res=False)

logstr = 'logs/log_cyldim{0}NU{1}NY{2}gamma{3}'.format(cyldim, NU, NY, gamma) +\
    'closedloop{0}'.format(closed_loop) +\
    't0{0}tE{1}Nts{2}'.format(t0, tE, Nts) +\
    'Re{2}to{3}kappa{0}to{1}eps{4}'.format(trunclist[0], trunclist[-1],
                                           relist[0], relist[-1], perturbpara)

# print 'log goes ' + logstr
# print 'how about \ntail -f '+logstr
# sys.stdout = open(logstr, 'a', 0)
print(('{0}'*10 + '\n log started at {1} \n' + '{0}'*10).
      format('X', str(datetime.datetime.now())))

for ctrunc in trunclist:
    for cre in range(1, len(relist)):
        import matplotlib.pyplot as plt
        plt.close('all')
        lqgbt_lnse.lqgbt(problemname='cylinderwake', N=cyldim,
                         use_ric_ini=relist[cre-1],
                         cl_linsys=cl_linsys,
                         NU=NU, NY=NY,
                         Re=relist[cre], plain_bt=False,
                         trunc_lqgbtcv=ctrunc,
                         t0=t0, tE=tE, Nts=Nts,
                         nwtn_adi_dict=nwtn_adi_dict,
                         paraoutput=False, multiproc=True,
                         comp_freqresp=False, comp_stepresp=False,
                         pymess=pymess,
                         gamma=gamma,
                         # closed_loop='red_output_fb',
                         # closed_loop=None,
                         plotit=False,
                         whichinival=whichinival,
                         hinf=hinf,
                         trytofail=trytofail, ttf_npcrdstps=ttf_npcrdstps,
                         robit=robit, robmrgnfac=robmrgnfac,
                         closed_loop=closed_loop,
                         perturbpara=perturbpara)

    print(infostring)
