import numpy as np

import sadptprj_riclyap_adi.bal_trunc_utils as btu

__all__ = ['get_rl_projections',
           'get_prj_model']

# pymess = False
pymess_dict = {}
plain_bt = False
debug = False


def get_rl_projections(zwc=None, zwo=None, mmat=None,
                       trunc_lqgbtcv=None):

    print(('computing the left and right transformations'))
    tl, tr, svs = btu.\
        compute_lrbt_transfos(zfc=zwc, zfo=zwo,
                              mmat=mmat,
                              trunck={'threshh': trunc_lqgbtcv})
    print('... done! - computing the left and right transformations')

    return tl, tr


def get_prj_model(mmat=None, fmat=None, jmat=None, bmat=None, cmat=None,
                  zwo=None, zwc=None, cmprlprjpars={}):

    hinfgamma = None

    tl, tr = get_rl_projections(zwc=zwc, zwo=zwo,
                                fmat=fmat, mmat=mmat, jmat=jmat,
                                cmat=cmat, bmat=bmat,
                                **cmprlprjpars)

    ak_mat = np.dot(tl.T, fmat.dot(tr))
    ck_mat = cmat.dot(tr)
    bk_mat = tl.T.dot(bmat)

    tltm, trtm = tl.T*mmat, tr.T*mmat
    xok = np.dot(np.dot(tltm, zwo), np.dot(zwo.T, tltm.T))
    xck = np.dot(np.dot(trtm, zwc), np.dot(zwc.T, trtm.T))

    return ak_mat, bk_mat, ck_mat, xok, xck, hinfgamma
