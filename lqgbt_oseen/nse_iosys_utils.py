import numpy as np

import dolfin_navier_scipy.problem_setups as dnsps
import dolfin_navier_scipy.stokes_navier_utils as snu

import distributed_control_fenics.cont_obs_utils as cou

import sadptprj_riclyap_adi.lin_alg_utils as lau


def compute_nse_steadystate(M=None, A=None, J=None,
                            stksbc_rhs=None, nom_rhs=None,
                            Re=None, relist=None, vpcachestr=None,
                            V=None, Q=None,
                            invinds=None, bcinds=None, bcvals=None):

    ''' compute the solution of the steadystate Navier-Stokes equations

    by sequentially increasing the `Re` number to have good initial guesses
    '''

    v_init = None
    for initre in relist:
        if initre >= Re:
            initre = Re
            rescl = 1.
        else:
            print('Initialising the steadystate solution with Re=', initre)
            rescl = 1/(initre/Re)
        try:
            if vpcachestr is None:
                raise IOError()
            else:
                cachssvs = vpcachestr + 'Re{0}_ssvsol.npy'.format(initre)
                cachssps = vpcachestr + 'Re{0}_sspsol.npy'.format(initre)
            vp_ss_nse = (np.load(cachssvs), np.load(cachssps))
            print('loaded sssol from: ', cachssvs)
        except IOError:
            # import ipdb
            # ipdb.set_trace()
            vp_ss_nse = snu.\
                solve_steadystate_nse(M=M, A=rescl*A, J=J, V=V, Q=Q,
                                      fv=rescl*stksbc_rhs['fv']+nom_rhs['fv'],
                                      fp=rescl*stksbc_rhs['fp']+nom_rhs['fp'],
                                      invinds=invinds,
                                      dbcinds=bcinds, dbcvals=bcvals,
                                      return_vp=True,
                                      vel_start_nwtn=v_init,
                                      vel_nwtn_tol=4e-13,
                                      clearprvdata=True)
            np.save(cachssvs, vp_ss_nse[0])
            np.save(cachssps, vp_ss_nse[1])
            print('saved sssol to: ', cachssvs)
        if initre == Re:
            break
        v_init = vp_ss_nse[0]

    return vp_ss_nse


def assmbl_linrzd_convtrm(vvec=None, invinds=None,
                          V=None, bcinds=None, bcvals=None):
    ''' assemble the linearized convection term
    '''

    if vvec is not None:
        (convc_mat, rhs_con,
         rhsv_conbc) = snu.get_v_conv_conts(vvec=vvec, invinds=invinds, V=V,
                                            dbcinds=bcinds, dbcvals=bcvals)
    return convc_mat


def assmbl_nse_sys(Re=None, scheme='TH', meshparams=None,
                   palpha=None, Cgrid=None, gamma=None):

    femp, stokesmatsc, nom_rhs, stksbc_rhs = \
        dnsps.get_sysmats(problem='gen_bccont', Re=Re, bccontrol=True,
                          scheme='TH', mergerhs=False,
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

    return femp, stokesmatsc, nom_rhs, stksbc_rhs, b_mat, c_mat_reg
