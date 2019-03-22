import numpy as np

import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.problem_setups as dnsps
import dolfin_navier_scipy.stokes_navier_utils as snu

import distr_control_fenics.cont_obs_utils as cou

import sadptprj_riclyap_adi.lin_alg_utils as lau

import nse_riccont_utils as nru

Re = 5.5e1  # 1e2
ddir = 'data/'
bccontrol = True
problemname = 'cylinderwake'
pymess = True
prbstr = '_pymessvsmymess'
NU, NY = 3, 3
N, NV = 2, 9384
gamma = 1e0
palpha = 1e-5
multiproc = True
Rmhalf = 1./np.sqrt(gamma)

nwtn_myadi_dict = dict(adi_max_steps=300,
                       adi_newZ_reltol=1e-8,
                       nwtn_max_steps=30,
                       nwtn_upd_reltol=1e-9,
                       nwtn_upd_abstol=1e-8,
                       ms=[-500., -100.0,
                           -30.0, -20.0, -10.0, -5.0, -3.0, -1.0],
                       verbose=True,
                       full_upd_norm_check=False,
                       check_lyap_res=False)

nwtn_pyadi_dict = dict(verbose=True, maxit=45, aditol=1e-12,
                       nwtn_res2_tol=1e-12, linesearch=True)

nwtn_pyadi_dict = dict(verbose=True, linesearch=True, maxit=50)

femp, stokesmatsc, rhsd_vfrc, rhsd_stbc \
    = dnsps.get_sysmats(problem=problemname, N=N, Re=Re,
                        bccontrol=bccontrol, scheme='TH')
invinds, NV = femp['invinds'], len(femp['invinds'])

stokesmatsc['A'] = stokesmatsc['A'] + 1./palpha*stokesmatsc['Arob']
b_mat = 1./palpha*stokesmatsc['Brob']

b_mat_reg = lau.app_prj_via_sadpnt(amat=stokesmatsc['M'],
                                   jmat=stokesmatsc['J'],
                                   rhsv=b_mat,
                                   transposedprj=True)

b_mat_rgscl = b_mat_reg*Rmhalf

contsetupstr = 'NV{0}_bcc_NY{1}_palpha{2}'.format(NV, NY, palpha)
shortcontsetupstr = '{0}{1}{2}'.format(NV, NY, np.int(np.log2(palpha)))
fdstr = ddir + problemname + '_Re{0}_gamma{1}_'.format(Re, gamma) + \
    contsetupstr + prbstr

try:
    mc_mat = dou.load_spa(ddir + contsetupstr + '__mc_mat')
    y_masmat = dou.load_spa(ddir + contsetupstr + '__y_masmat')
    print(('loaded `c_mat` from' + ddir + contsetupstr + '...'))
except IOError:
    print('computing `c_mat`...')
    mc_mat, y_masmat = cou.get_mout_opa(odcoo=femp['odcoo'],
                                        V=femp['V'], NY=NY)
    dou.save_spa(mc_mat, ddir + contsetupstr + '__mc_mat')
    dou.save_spa(y_masmat, ddir + contsetupstr + '__y_masmat')

c_mat = lau.apply_massinv(y_masmat, mc_mat, output='sparse')
c_mat = c_mat[:, invinds][:, :]
c_mat_reg = lau.app_prj_via_sadpnt(amat=stokesmatsc['M'],
                                   jmat=stokesmatsc['J'],
                                   rhsv=c_mat.T,
                                   transposedprj=True).T
soldict = {}
soldict.update(stokesmatsc)  # containing A, J, JT
soldict.update(femp)  # adding V, Q, invinds, diribcs
# soldict.update(rhsd_vfrc)  # adding fvc, fpr
veldatastr = ddir + problemname + '_Re{0}'.format(Re)
if bccontrol:
    veldatastr = veldatastr + '__bcc_palpha{0}'.format(palpha)

nu = femp['charlen']/Re
soldict.update(fv=rhsd_stbc['fv']+rhsd_vfrc['fvc'],
               fp=rhsd_stbc['fp']+rhsd_vfrc['fpr'],
               N=N, nu=nu)

vp_ss_nse, list_norm_nwtnupd = snu.\
    solve_steadystate_nse(vel_pcrd_stps=5, return_vp=True,
                          useolddata=False, **soldict)

v_ss_nse = vp_ss_nse[:NV]
p_ss_nse = vp_ss_nse[NV:]
(convc_mat, rhs_con,
 rhsv_conbc) = snu.get_v_conv_conts(prev_v=v_ss_nse, invinds=invinds,
                                    V=femp['V'], diribcs=femp['diribcs'])

f_mat = - stokesmatsc['A'] - convc_mat

# fmat = stokesmatsc['A']
J, M = stokesmatsc['J'], stokesmatsc['M']

myzwc, myzwo, _ = nru.get_ric_facs(fmat=f_mat, mmat=M, jmat=J,
                                   bmat=b_mat_rgscl, cmat=c_mat_reg,
                                   ric_ini_str=None,
                                   nwtn_adi_dict=nwtn_myadi_dict,
                                   fdstr=fdstr,
                                   multiproc=multiproc, pymess=False,
                                   checktheres=False)

pyzwc, pyzwo, _ = nru.get_ric_facs(fmat=f_mat, mmat=M, jmat=J,
                                   bmat=b_mat_rgscl, cmat=c_mat_reg,
                                   ric_ini_str=None,
                                   nwtn_adi_dict=nwtn_pyadi_dict,
                                   fdstr=fdstr+'__pymess',
                                   multiproc=multiproc, pymess=True,
                                   checktheres=True)


checkmat = np.random.randn(NV, 5)
pymXv = np.dot(pyzwc, np.dot(pyzwc.T, checkmat))
mymXv = np.dot(myzwc, np.dot(myzwc.T, checkmat))
print(np.linalg.norm(pymXv-mymXv)/np.linalg.norm(mymXv))
print(np.linalg.norm(mymXv))
print(np.linalg.norm(pymXv))
print(np.allclose(pymXv, mymXv))
print(np.linalg.norm(np.dot(pyzwo, np.dot(pyzwo.T, checkmat))))
print(np.linalg.norm(np.dot(myzwo, np.dot(myzwo.T, checkmat))))

# ## TODO: check the performance
# ## TODO: what about the observability -- does it stabilize well?
# 1. comp zwo/zwc with the Oseen linearization
# 2. integrate the cl-Oseen linearization with A - BBTXc and AT - CTCXo
# 3. check the costfun
