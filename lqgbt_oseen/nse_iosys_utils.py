import numpy as np

import dolfin_navier_scipy.problem_setups as dnsps

import distributed_control_fenics.cont_obs_utils as cou

import sadptprj_riclyap_adi.lin_alg_utils as lau


def compute_nse_steadystate():
    return


def assmbl_linrzd_nse():
    return


def assmbl_nse_sys(Re=None, scheme='TH', meshparams=None,
                   palpha=None, Cgrid=None, gamma=None):

    femp, stokesmatsc, rhsd = \
        dnsps.get_sysmats(problem='gen_bccont', Re=Re, bccontrol=True,
                          scheme='TH', mergerhs=True,
                          meshparams=meshparams)
    invinds = femp['invinds']

    stokesmatsc['A'] = stokesmatsc['A'] + 1./palpha*stokesmatsc['Arob']
    b_mat = 1./palpha*stokesmatsc['Brob']
    b_mat = lau.app_prj_via_sadpnt(amat=stokesmatsc['M'],
                                   jmat=stokesmatsc['J'],
                                   rhsv=b_mat, transposedprj=True)
    if gamma is not None:
        Rmhalf = 1./np.sqrt(gamma)
        b_mat = Rmhalf*b_mat

    mc_mat, y_masmat = cou.get_mout_opa(odcoo=femp['odcoo'],
                                        V=femp['V'], mfgrid=Cgrid)

    c_mat = lau.apply_massinv(y_masmat, mc_mat, output='sparse')
    # restrict the operators to the inner nodes

    mc_mat = mc_mat[:, invinds][:, :]
    c_mat = c_mat[:, invinds][:, :]
    c_mat_reg = lau.app_prj_via_sadpnt(amat=stokesmatsc['M'],
                                       jmat=stokesmatsc['J'],
                                       rhsv=c_mat.T,
                                       transposedprj=True).T

    return femp, stokesmatsc, rhsd, b_mat, c_mat_reg
