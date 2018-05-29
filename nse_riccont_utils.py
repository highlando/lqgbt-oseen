import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla

import dolfin_navier_scipy.data_output_utils as dou

import sadptprj_riclyap_adi.proj_ric_utils as pru
import sadptprj_riclyap_adi.bal_trunc_utils as btu

__all__ = ['get_ric_facs',
           'get_rl_projections',
           'get_prj_model',
           'get_hinf_ric_facs']

# pymess = False
pymess_dict = {}
plain_bt = False
debug = False


def get_hinf_ric_facs(fmat=None, mmat=None, jmat=None,
                      bmat=None, cmat=None,
                      ric_ini_str=None, fdstr=None,
                      importexport='export',
                      checktheres=False):

    if importexport == 'export':
        from scipy.io import savemat
        zinic = dou.load_npa(ric_ini_str + '__zwc')
        zinio = dou.load_npa(ric_ini_str + '__zwo')
        savematdict = dict(mmat=mmat, amat=fmat, jmat=jmat,
                           bmat=bmat, cmat=cmat,
                           zinic=zinic, zinio=zinio)

        savemat(fdstr + '__mats', savematdict, do_compression=True)
        raise UserWarning('done with saving to ' + fdstr + '__mats')

    return


def get_ric_facs(fmat=None, mmat=None, jmat=None,
                 bmat=None, cmat=None,
                 ric_ini_str=None, fdstr=None,
                 nwtn_adi_dict=None,
                 zwconly=False,
                 multiproc=False, pymess=False, checktheres=False):

    if pymess:
        get_ricadifacs = pru.pymess_dae2_cnt_riccati
        adidict = nwtn_adi_dict
    else:
        get_ricadifacs = pru.proj_alg_ric_newtonadi
        adidict = dict(nwtn_adi_dict=nwtn_adi_dict)

    zinic, zinio = None, None
    if ric_ini_str is not None:
        try:
            zinic = dou.load_npa(ric_ini_str + '__zwc')
            if not zwconly:
                zinio = dou.load_npa(ric_ini_str + '__zwo')
            print('Initialize Newton ADI by zwc/zwo from ' + ric_ini_str)
        except IOError:
            raise UserWarning('No data at `{0}` (for init of Ric solves)'.
                              format(ric_ini_str))

    print('computing factors of Grams: \n\t' + fdstr + '__zwc/__zwo')

    def compobsg():
        if zwconly:
            print('we only compute zwc')
            return
        try:
            zwo = dou.load_npa(fdstr + '__zwo')
            print('yeyeyeah, __zwo is there')
        except IOError:
            zwo = get_ricadifacs(mmat=mmat.T, amat=fmat.T, jmat=jmat,
                                 bmat=cmat.T, wmat=bmat,
                                 z0=zinio, **adidict)['zfac']
            dou.save_npa(zwo, fdstr + '__zwo')
        return

    def compcong():
        try:
            zwc = dou.load_npa(fdstr + '__zwc')
            print('yeyeyeah, __zwc is there')
        except IOError:
            # XXX: why here bmat*Rmhalf and in zwo not?
            zwc = get_ricadifacs(mmat=mmat, amat=fmat, jmat=jmat,
                                 bmat=bmat, wmat=cmat.T,
                                 z0=zinic, **adidict)['zfac']
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
    if not zwconly:
        zwo = dou.load_npa(fdstr + '__zwo')

    if checktheres:
        print('checking the Riccati residuals....')
        # check the cont Ric residual
        umat = 0.5*bmat
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

    if zwconly:
        return zwc
    else:
        return zwc, zwo


def get_rl_projections(fdstr=None, truncstr=None,
                       zwc=None, zwo=None,
                       fmat=None, mmat=None, jmat=None,
                       bmat=None, cmat=None,
                       cmpricfacpars={},
                       pymess=False,
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
                                    pymess=pymess,
                                    fmat=fmat, mmat=mmat, jmat=jmat,
                                    cmat=cmat, bmat=bmat,
                                    **cmpricfacpars)
        tl, tr, svs = btu.\
            compute_lrbt_transfos(zfc=zwc, zfo=zwo,
                                  mmat=mmat,
                                  trunck={'threshh': trunc_lqgbtcv})
        dou.save_npa(tl, fdstr + truncstr + '__tl')
        dou.save_npa(tr, fdstr + truncstr + '__tr')
        dou.save_npa(svs, fdstr + '__svs')
        print('... done! - computing the left and right transformations')

    return tl, tr


def get_prj_model(truncstr=None, fdstr=None,
                  matsdict={},
                  abconly=False,
                  mmat=None, fmat=None, jmat=None, bmat=None, cmat=None,
                  zwo=None, zwc=None, pymess=False,
                  tl=None, tr=None,
                  return_tltr=True,
                  cmpricfacpars={}, cmprlprjpars={}):

    if tl is None or tr is None:
        tltristhere = False
    else:
        tltristhere = True

    if not tltristhere and return_tltr:
        tl, tr = get_rl_projections(fdstr=fdstr, truncstr=truncstr,
                                    zwc=zwc, zwo=zwo,
                                    fmat=fmat, mmat=mmat, jmat=jmat,
                                    cmat=cmat, bmat=bmat,
                                    pymess=pymess,
                                    cmpricfacpars=cmpricfacpars,
                                    **cmprlprjpars)
        tltristhere = True

    try:
        if debug:
            raise IOError
        ak_mat = dou.load_npa(fdstr+truncstr+'__ak_mat')
        ck_mat = dou.load_npa(fdstr+truncstr+'__ck_mat')
        bk_mat = dou.load_npa(fdstr+truncstr+'__bk_mat')

    except IOError:
        print('couldn"t load the red model - gonna compute it')

        if tltristhere:
            pass
        else:
            tl, tr = get_rl_projections(fdstr=fdstr, truncstr=truncstr,
                                        zwc=zwc, zwo=zwo,
                                        fmat=fmat, mmat=mmat, jmat=jmat,
                                        cmat=cmat, bmat=bmat,
                                        cmpricfacpars=cmpricfacpars,
                                        **cmprlprjpars)
            tltristhere = True

        ak_mat = np.dot(tl.T, fmat.dot(tr))
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
                                        **cmpricfacpars)

            if tltristhere:
                pass
            else:
                tl, tr = get_rl_projections(fdstr=fdstr, truncstr=truncstr,
                                            mmat=mmat,
                                            zwc=zwc, zwo=zwo,
                                            **cmprlprjpars)
                tltristhere = True

            tltm, trtm = tl.T*mmat, tr.T*mmat
            xok = np.dot(np.dot(tltm, zwo), np.dot(zwo.T, tltm.T))
            xck = np.dot(np.dot(trtm, zwc), np.dot(zwc.T, trtm.T))
            dou.save_npa(xok, fdstr+truncstr+'__xok')
            dou.save_npa(xck, fdstr+truncstr+'__xck')

        if return_tltr:
            return ak_mat, bk_mat, ck_mat, xok, xck, tl, tr

        else:
            return ak_mat, bk_mat, ck_mat, xok, xck


def get_sdrefb_upd(amat, t, fbtype=None, wnrm=2,
                   B=None, R=None, Q=None, maxeps=None,
                   baseA=None, baseZ=None, baseP=None, maxfac=None, **kwargs):
    if fbtype == 'sylvupdfb' or fbtype == 'singsylvupd':
        if baseP is not None:
            deltaA = amat - baseA
            epsP = spla.solve_sylvester(amat, -baseZ, -deltaA)
            eps = npla.norm(epsP, ord=wnrm)
            print('|amat - baseA|: {0} -- |E|: {1}'.
                  format(npla.norm(deltaA, ord=wnrm), eps))
            if maxeps is not None:
                if eps < maxeps:
                    opepsPinv = npla.inv(epsP+np.eye(epsP.shape[0]))
                    return baseP.dot(opepsPinv), True
            elif maxfac is not None:
                if (1+eps)/(1-eps) < maxfac and eps < 1:
                    opepsPinv = npla.inv(epsP+np.eye(epsP.shape[0]))
                    return baseP.dot(opepsPinv), True

    # otherwise: (SDRE feedback or `eps` too large already)
    # curX = spla.solve_continuous_are(amat, B, Q, R)
    # if fbtype == 'sylvupdfb' or fbtype == 'singsylvupd':
    #     logger.debug('in `get_fb_dict`: t={0}: eps={1} too large, switch!'.
    #                  format(t, eps))
    # else:
    #     logger.debug('t={0}: computed the SDRE feedback')
    return None, False
