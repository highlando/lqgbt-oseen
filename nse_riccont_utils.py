import numpy as np

import dolfin_navier_scipy.data_output_utils as dou

import sadptprj_riclyap_adi.bal_trunc_utils as btu

__all__ = ['get_rl_projections',
           'get_prj_model']

# pymess = False
pymess_dict = {}
plain_bt = False
debug = False


def get_rl_projections(fdstr=None, truncstr=None,
                       zwc=None, zwo=None,
                       fmat=None, mmat=None, jmat=None,
                       bmat=None, cmat=None,
                       cmpricfacpars={}, hinf=False,
                       pymess=False,
                       trunc_lqgbtcv=None):

    print(('computing the left and right transformations'))
    tl, tr, svs = btu.\
        compute_lrbt_transfos(zfc=zwc, zfo=zwo,
                              mmat=mmat,
                              trunck={'threshh': trunc_lqgbtcv})
    print('... done! - computing the left and right transformations')

    return tl, tr


def get_prj_model(truncstr=None, fdstr=None,
                  hinf=False,
                  matsdict={},
                  abconly=False,
                  mmat=None, fmat=None, jmat=None, bmat=None, cmat=None,
                  zwo=None, zwc=None, pymess=False,
                  tl=None, tr=None,
                  return_tltr=True,
                  cmpricfacpars={}, cmprlprjpars={}):

    hinfgamma = None

    tl, tr = get_rl_projections(fdstr=fdstr, truncstr=truncstr,
                                zwc=zwc, zwo=zwo,
                                fmat=fmat, mmat=mmat, jmat=jmat,
                                cmat=cmat, bmat=bmat,
                                pymess=pymess, hinf=hinf,
                                cmpricfacpars=cmpricfacpars,
                                **cmprlprjpars)

    ak_mat = np.dot(tl.T, fmat.dot(tr))
    ck_mat = cmat.dot(tr)
    bk_mat = tl.T.dot(bmat)
    dou.save_npa(ak_mat, fdstr+truncstr+'__ak_mat')
    dou.save_npa(ck_mat, fdstr+truncstr+'__ck_mat')
    dou.save_npa(bk_mat, fdstr+truncstr+'__bk_mat')

    tltm, trtm = tl.T*mmat, tr.T*mmat
    xok = np.dot(np.dot(tltm, zwo), np.dot(zwo.T, tltm.T))
    xck = np.dot(np.dot(trtm, zwc), np.dot(zwc.T, trtm.T))

    return ak_mat, bk_mat, ck_mat, xok, xck, hinfgamma
