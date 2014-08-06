import lqgbt_lnse
import sys
import datetime

sys.stdout = open('logcompscript', 'a', 0)
print('{0}'*10 + '\n log started at {1} \n' + '{0}'*10).\
    format('X', str(datetime.datetime.now()))

# to compute stabilizing initial values for higher Re numbers
relist = [None, 5.0e1, 1.0e2, 1.5e2, 2.0e2, 2.5e2, 3.0e2, 3.5e2, 4.0e2]

# relist = [None, 2.0e2]  # , 2.5e2]
# mesh parameter for the cylinder meshes
cyldim = 3
# where to truncate the LQGBT characteristic values
trunclist = [1e-2]  # , 1e-3, 1e-2, 1e-1, 1e-0]
# dimension of in and output spaces
NU, NY = 3, 3

nwtn_adi_dict = dict(adi_max_steps=350,
                     adi_newZ_reltol=1e-7,
                     nwtn_max_steps=30,
                     nwtn_upd_reltol=4e-8,
                     nwtn_upd_abstol=1e-7,
                     verbose=True,
                     ms=[-5.0, -2.0, -1.5, -1.1, -1.0],
                     full_upd_norm_check=False,
                     check_lyap_res=False)

for ctrunc in trunclist:
    for cre in range(1, len(relist)):
        lqgbt_lnse.lqgbt(problemname='cylinderwake', N=cyldim,
                         use_ric_ini=relist[cre-1],
                         NU=NU, NY=NY,
                         Re=relist[cre], plain_bt=False,
                         trunc_lqgbtcv=ctrunc,
                         t0=0.0, tE=12.0, Nts=4.8e3+1,
                         nwtn_adi_dict=nwtn_adi_dict,
                         paraoutput=False,
                         comp_freqresp=False, comp_stepresp=False,
                         # 'nonlinear',
                         # closed_loop='full_state_fb')
                         # closed_loop='red_output_fb')
                         # closed_loop=None)
                         closed_loop=False)
