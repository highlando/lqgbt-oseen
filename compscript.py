import lqgbt_lnse
# to compute stabilizing initial values for higher Re numbers
relist = [None, 5.0e1, 1.0e2, 1.5e2, 2.0e2]

# mesh parameter for the cylinder meshes
cyldim = 4
# where to truncate the LQGBT characteristic values
trunclist = [1e-2]  # , 1e-3, 1e-2, 1e-1, 1e-0]
# dimension of in and output spaces
NU, NY = 3, 3

for ctrunc in trunclist:
    for cre in range(1, len(relist)):
        lqgbt_lnse.lqgbt(problemname='cylinderwake', N=cyldim,
                         use_ric_ini=relist[cre-1],
                         NU=NU, NY=NY,
                         Re=relist[cre], plain_bt=False,
                         trunc_lqgbtcv=ctrunc,
                         t0=0.0, tE=12.0, Nts=2.4e3+1,
                         paraoutput=False,
                         comp_freqresp=True, comp_stepresp=False,
                         # 'nonlinear',
                         # closed_loop='full_state_fb')
                         closed_loop='red_output_fb')
                         # closed_loop=None)
                         # closed_loop=False)
