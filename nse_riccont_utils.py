import numpy as np

import dolfin_navier_scipy.data_output_utils as dou

import sadptprj_riclyap_adi.proj_ric_utils as pru
import sadptprj_riclyap_adi.bal_trunc_utils as btu

__all__ = ['get_ric_facs',
           'get_rl_projections',
           'get_prj_model']

pymess = False
pymess_dict = {}
plain_bt = False
debug = True


def get_ric_facs(fmat=None, mmat=None, jmat=None,
                 bmat=None, cmat=None,
                 Rmhalf=None, Rmo=None,
                 ric_ini_str=None, fdstr=None,
                 nwtn_adi_dict=None,
                 multiproc=False, pymess=False, checktheres=False):

    if pymess:
        get_ricadifacs = pru.pymess_dae2_cnt_riccati
        adidict = pymess_dict
    else:
        get_ricadifacs = pru.proj_alg_ric_newtonadi
        adidict = nwtn_adi_dict

    zinic, zinio = None, None
    if ric_ini_str is not None:
        try:
            zinic = dou.load_npa(ric_ini_str + '__zwc')
            zinio = dou.load_npa(ric_ini_str + '__zwo')
            print('Initialize Newton ADI by zwc/zwo from ' + ric_ini_str)
        except IOError:
            raise UserWarning('No data at `{0}` (for init of Ric solves)'.
                              format(ric_ini_str))

    print('computing factors of Grams: \n\t' + fdstr + '__zwc/__zwo')

    def compobsg():
        try:
            zwo = dou.load_npa(fdstr + '__zwo')
            print('yeyeyeah, __zwo is there')
        except IOError:
            zwo = get_ricadifacs(mmat=mmat.T, amat=fmat.T, jmat=jmat,
                                 bmat=cmat.T, wmat=bmat,
                                 nwtn_adi_dict=adidict,
                                 z0=zinio)['zfac']
            dou.save_npa(zwo, fdstr + '__zwo')
        return

    def compcong():
        try:
            zwc = dou.load_npa(fdstr + '__zwc')
            print('yeyeyeah, __zwc is there')
        except IOError:
            # XXX: why here bmat*Rmhalf and in zwo not?
            zwc = get_ricadifacs(mmat=mmat, amat=fmat, jmat=jmat,
                                 bmat=bmat*Rmhalf, wmat=cmat.T,
                                 nwtn_adi_dict=adidict,
                                 z0=zinic)['zfac']
            dou.save_npa(zwc, fdstr + '__zwc')
        return

    if multiproc:
        import multiprocessing

        print('\n ### multithread start - output maybe intermangled')
        p1 = multiprocessing.Process(target=compobsg)
        p2 = multiprocessing.Process(target=compcong)
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        print('### multithread end')

    else:
        compobsg()
        compcong()

    zwc = dou.load_npa(fdstr + '__zwc')
    zwo = dou.load_npa(fdstr + '__zwo')

    if checktheres:
        print('checking the Riccati residuals....')
        # check the cont Ric residual
        umat = 0.5*bmat*Rmo
        vmat = np.dot(np.dot(bmat.T, zwc), zwc.T)*mmat
        res = pru.\
            comp_proj_lyap_res_norm(zwc, amat=fmat, mmat=mmat,
                                    jmat=jmat,
                                    wmat=cmat.T,
                                    umat=umat, vmat=vmat)
        print('sqrd Residual of cont-Riccati: ', res)
        ctc = np.dot(cmat, cmat.T)
        nrhs = (ctc * ctc).sum(-1).sum()
        print('sqrd f-norm of rhs=C.T*C: ', nrhs**2)

        # check the obsv Ric residual
        umat = 0.5*cmat.T
        vmat = np.dot(np.dot(cmat, zwo), zwo.T)*mmat
        res = pru.\
            comp_proj_lyap_res_norm(zwo, amat=fmat.T, mmat=mmat.T,
                                    jmat=jmat,
                                    wmat=bmat,
                                    umat=umat, vmat=vmat)
        print('sqrd Residual of obsv-Riccati: ', res)

        btb = np.dot(bmat.T, bmat)
        nrhs = (btb * btb).sum(-1).sum()
        print('sqrd f-norm of rhs=B*B.T: ', nrhs**2)
        print('... done with checking the Riccati residuals!')

    return zwc, zwo


def get_rl_projections(fdstr=None, truncstr=None,
                       zwc=None, zwo=None,
                       fmat=None, mmat=None, jmat=None,
                       bmat=None, cmat=None,
                       Rmhalf=None, Rmo=None,
                       cmpricfacpars={},
                       trunc_lqgbtcv=None):

    try:
        if debug:
            raise IOError
        tl = dou.load_npa(fdstr + truncstr + '__tl')
        tr = dou.load_npa(fdstr + truncstr + '__tr')
        print(('loaded the left and right transformations: \n' +
               fdstr + truncstr + '__tl/__tr'))
        # if robit:
        #     svs = dou.load_npa(fdstr + '__svs')

    except IOError:
        print(('computing the left and right transformations' +
               ' and saving to: \n' + fdstr + truncstr + '__tl/__tr'))
        if zwc is None or zwo is None:
            zwc, zwo = get_ric_facs(fdstr=fdstr,
                                    fmat=fmat, mmat=mmat, jmat=jmat,
                                    cmat=cmat, bmat=bmat,
                                    Rmhalf=Rmhalf, Rmo=Rmo,
                                    **cmpricfacpars)
        # print('norm zwc/zwo = {0}/{1}'.format(np.linalg.norm(zwc),
        #                                       np.linalg.norm(zwo)))
        # print('truncing at {0}:'.format(trunc_lqgbtcv))
        tl, tr, svs = btu.\
            compute_lrbt_transfos(zfc=zwc, zfo=zwo,
                                  mmat=mmat,
                                  trunck={'threshh': trunc_lqgbtcv})
        # tls = tl.shape
        # print('shape of tl: [{0}, {1}]'.format(tls[0], tls[1]))
        # print(svs)
        dou.save_npa(tl, fdstr + truncstr + '__tl')
        dou.save_npa(tr, fdstr + truncstr + '__tr')
        dou.save_npa(svs, fdstr + '__svs')
        print('... done! - computing the left and right transformations')

    return tl, tr


def get_prj_model(truncstr=None, fdstr=None,
                  matsdict={},
                  abconly=False,
                  mmat=None, fmat=None, jmat=None, bmat=None, cmat=None,
                  zwo=None, zwc=None,
                  Rmhalf=None, Rmo=None,
                  cmpricfacpars={}, cmprlprjpars={}):

    try:
        if debug:
            raise IOError
        ak_mat = dou.load_npa(fdstr+truncstr+'__ak_mat')
        ck_mat = dou.load_npa(fdstr+truncstr+'__ck_mat')
        bk_mat = dou.load_npa(fdstr+truncstr+'__bk_mat')

    except IOError:
        print('couldn"t load the red model - gonna compute it')

        tl, tr = get_rl_projections(fdstr=fdstr, truncstr=truncstr,
                                    zwc=zwc, zwo=zwo,
                                    fmat=fmat, mmat=mmat, jmat=jmat,
                                    cmat=cmat, bmat=bmat,
                                    Rmhalf=Rmhalf, Rmo=Rmo,
                                    cmpricfacpars=cmpricfacpars,
                                    **cmprlprjpars)

        ak_mat = np.dot(tl.T, fmat*tr)
        ck_mat = cmat.dot(tr)
        bk_mat = tl.T.dot(bmat)
        dou.save_npa(ak_mat, fdstr+truncstr+'__ak_mat')
        dou.save_npa(ck_mat, fdstr+truncstr+'__ck_mat')
        dou.save_npa(bk_mat, fdstr+truncstr+'__bk_mat')

    if abconly:
        return ak_mat, bk_mat, ck_mat

    else:
        try:
            if debug:
                raise IOError
            xok = dou.load_npa(fdstr+truncstr+'__xok')
            xck = dou.load_npa(fdstr+truncstr+'__xck')
        except IOError:
            if zwo is None and zwc is None:
                zwc, zwo = get_ric_facs(fdstr=fdstr,
                                        fmat=fmat, mmat=mmat, jmat=jmat,
                                        cmat=cmat, bmat=bmat,
                                        Rmhalf=Rmhalf, Rmo=Rmo,
                                        **cmpricfacpars)

            tl, tr = get_rl_projections(fdstr=fdstr, truncstr=truncstr,
                                        mmat=mmat,
                                        zwc=zwc, zwo=zwo,
                                        **cmprlprjpars)

            tltm, trtm = tl.T*mmat, tr.T*mmat
            xok = np.dot(np.dot(tltm, zwo), np.dot(zwo.T, tltm.T))
            xck = np.dot(np.dot(trtm, zwc), np.dot(zwc.T, trtm.T))
            dou.save_npa(xok, fdstr+truncstr+'__xok')
            dou.save_npa(xck, fdstr+truncstr+'__xck')

        return ak_mat, bk_mat, ck_mat, xok, xck
