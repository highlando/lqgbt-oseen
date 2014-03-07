import lqgbt_lnse
# import os
# import glob
# import time

relist = [None, 2.0e2]  # , 2.5e2]
cyldim = 2
# os.chdir('data/')
# for fname in glob.glob('*__vel*'):
#     os.remove(fname)
# os.chdir('..')

for cre in range(1, len(relist)):
    lqgbt_lnse.lqgbt(problemname='cylinderwake', N=cyldim,
                     use_ric_ini=relist[cre-1],
                     Re=relist[cre], plain_bt=False,
                     trunc_lqgbtcv=1e-5,
                     t0=0.0, tE=15.0, Nts=3e3+1,
                     comp_freqresp=False, comp_stepresp=False,
                     # closed_loop='full_state_fb')
                     closed_loop='red_output_fb')
                     # closed_loop=None)

### Use for plots:
# from sadptprj_riclyap_adi.bal_trunc_utils import plot_step_resp
# plot_step_resp("data/
#     cylinderwake_Re250.0_NV19468NU3NY3_lqgbt_Nred85_t0tENts0.06.0201.0.json")
# NV = 19468, NP = 2591, k = 85

# from sadptprj_riclyap_adi.bal_trunc_utils import plot_step_resp
# plot_step_resp("data/
#     cylinderwake_Re250.0_NV9356NU3NY3_lqgbt_Nred87_t0tENts0.05.02001.0.json")
# NV = 9356, NP = 1288, k = 87
