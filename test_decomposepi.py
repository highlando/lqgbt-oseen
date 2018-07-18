import dolfin_navier_scipy.problem_setups as dnsps


Re = 5e1  # 1e2
ddir = 'data/'
bccontrol = True
problemname = 'cylinderwake'
N = 1

femp, stokesmatsc, rhsd_vfrc, rhsd_stbc \
    = dnsps.get_sysmats(problem=problemname, N=N, Re=Re,
                        bccontrol=bccontrol, scheme='TH')

mmat = stokesmatsc['M']
jmat = stokesmatsc['J']

import nse_extlin_utils as neu

thl, thr, minvthr = neu.decomp_leray(mmat, jmat)
# np.allclose(np.eye(jmat.shape[1]), thr.T.dot(thl))
