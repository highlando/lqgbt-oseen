import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
from scipy.io import loadmat

import dolfin_navier_scipy.data_output_utils as dou

import sadptprj_riclyap_adi.proj_ric_utils as pru
import sadptprj_riclyap_adi.bal_trunc_utils as btu

__all__ = ['get_ric_facs',
           'get_rl_projections',
           'get_prj_model',
           'get_sdrefb_upd',
           'get_sdrgain_upd']

# pymess = False
pymess_dict = {}
plain_bt = False
debug = False


def get_ric_facs(fmat=None, mmat=None, jmat=None,
                 bmat=None, cmat=None,
                 ric_ini_str=None, fdstr=None,
                 nwtn_adi_dict=None,
                 zwconly=False, hinf=False,
                 multiproc=False, pymess=False, checktheres=False,
                 strtogramfacs=None):

    if strtogramfacs is not None:
        try:
            (fname, zwcaddrs, zwoaddrs, gammadrs) = strtogramfacs.\
                split(sep='%')
        except ValueError:
            (fname, zwcaddrs, zwoaddrs) = strtogramfacs.split(sep='%')
            gammadrs = None
        if pymess:
            try:
                lmd = {}
                loadmat(fname, mdict=lmd)
            except NotImplementedError:
                import mat73
                lmd = mat73.loadmat(fname)
            lmd = mat73.loadmat(fname)
            zwc, zwo, hinfgamma = lmd, lmd, lmd
            for isitthis in zwcaddrs.split(sep='.'):
                zwc = zwc[isitthis]
            for isitthis in zwoaddrs.split(sep='.'):
                zwo = zwo[isitthis]
            print('loaded the factors from: ', fname)
            if gammadrs is None:
                hinfgamma = None
                print('no gamma loaded')
            else:
                for isitthis in gammadrs.split(sep='.'):
                    hinfgamma = hinfgamma[isitthis]
                hinfgamma = np.atleast_1d(hinfgamma).flatten()[0]
                print('gamma_opt = {0}'.format(hinfgamma))
            return zwc, zwo, hinfgamma
        else:
            hinfgamma = None
            zwc = np.load(fname+zwcaddrs)
            zwo = np.load(fname+zwoaddrs)
            print('loaded the factors from: {0}+{1}/{2}'.
                  format(fname, zwcaddrs, zwoaddrs))
            if gammadrs is not None:
                hinfgamma = np.load(fname+gammadrs)
            return zwc, zwo, hinfgamma

    if hinf or pymess:
        # we can't compute we can only import export
        # other directory than `data`
        # hinfmatstr = 'oc-hinf-data/outputs/' + \
        from pathlib import Path
        hinfmatstr = 'external-data/oc-hinf-data/outputs/' + \
            Path(fdstr).name + '__mats'
        try:
            lmd = {}
            loadmat(hinfmatstr + '_output', mdict=lmd)
            if not hinf:
                zwc, zwo, hinfgamma = (lmd['outControl'][0, 0]['Z_LQG'],
                                       lmd['outFilter'][0, 0]['Z_LQG'],
                                       lmd['gam_opt'])
                print('loaded the lqg mats from the hinf matfile')
                return zwc, zwo, hinfgamma

            else:
                try:
                    zwc, zwo, hinfgamma = lmd['ZB'], lmd['ZC'], lmd['gam_opt']
                except KeyError:
                    zwc, zwo, hinfgamma = (lmd['outControl'][0, 0]['Z'],
                                           lmd['outFilter'][0, 0]['Z'],
                                           lmd['gam_opt'])
                print('loaded the hinf mats from the hinf matfile')
                print('gamma_opt = {0}'.format(hinfgamma))
                return zwc, zwo, hinfgamma

        except IOError:
            print('could not load: ' + hinfmatstr + '_output')
            hinfmatstr = 'external-data/oc-hinf-data/' + \
                Path(fdstr).name + '__mats'
            from scipy.io import savemat
            zinic = dou.load_npa(ric_ini_str + '__zwc')
            zinio = dou.load_npa(ric_ini_str + '__zwo')
            savematdict = dict(mmat=mmat, amat=fmat, jmat=jmat,
                               bmat=bmat, cmat=cmat,
                               zinic=zinic, zinio=zinio)
            savemat(hinfmatstr, savematdict, do_compression=True)
            raise UserWarning('done with saving to ' + hinfmatstr)

    if pymess:
        get_ricadifacs = pru.pymess_dae2_cnt_riccati
        adidict = nwtn_adi_dict
    else:
        get_ricadifacs = pru.proj_alg_ric_newtonadi
        adidict = dict(nwtn_adi_dict=nwtn_adi_dict)

    zinic, zinio = None, None
    if ric_ini_str is not None:
        try:
            zinic = np.load(ric_ini_str + '__zwc')
            if not zwconly:
                zinio = np.load(ric_ini_str + '__zwo')
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
            np.save(zwo, fdstr + '__zwo')
        return

    def compcong():
        try:
            zwc = np.load(fdstr + '__zwc')
            print('yeyeyeah, __zwc is there')
        except IOError:
            # XXX: why here bmat*Rmhalf and in zwo not?
            zwc = get_ricadifacs(mmat=mmat, amat=fmat, jmat=jmat,
                                 bmat=bmat, wmat=cmat.T,
                                 z0=zinic, **adidict)['zfac']
            np.save(zwc, fdstr + '__zwc')
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
        return zwc, None, None
    else:
        return zwc, zwo, None


def get_rl_projections(zwc=None, zwo=None,
                       mmat=None, trunc_lqgbtcv=None):

    tl, tr, svs = btu.\
        compute_lrbt_transfos(zfc=zwc, zfo=zwo, mmat=mmat,
                              trunck={'threshh': trunc_lqgbtcv})

    return tl, tr


def get_prj_model(abconly=False,
                  mmat=None, fmat=None, jmat=None, bmat=None, cmat=None,
                  zwo=None, zwc=None,
                  tl=None, tr=None):

    ak_mat = np.dot(tl.T, fmat.dot(tr))
    ck_mat = cmat.dot(tr)
    bk_mat = tl.T.dot(bmat)

    if abconly:
        return ak_mat, bk_mat, ck_mat

    tltm, trtm = tl.T*mmat, tr.T*mmat
    xok = np.dot(np.dot(tltm, zwo), np.dot(zwo.T, tltm.T))
    xck = np.dot(np.dot(trtm, zwc), np.dot(zwc.T, trtm.T))

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


def get_sdrgain_upd(amat, wnrm='fro', maxeps=None,
                    baseA=None, baseZ=None, baseGain=None,
                    maxfac=None):

    deltaA = amat - baseA
    # nda = npla.norm(deltaA, ord=wnrm)
    # nz = npla.norm(baseZ, ord=wnrm)
    # na = npla.norm(baseA, ord=wnrm)
    # import ipdb; ipdb.set_trace()

    epsP = spla.solve_sylvester(amat, -baseZ, -deltaA)
    # print('debugging!!!')
    # epsP = 0*amat
    eps = npla.norm(epsP, ord=wnrm)
    print('|amat - baseA|: {0} -- |E|: {1}'.
          format(npla.norm(deltaA, ord=wnrm), eps))
    if maxeps is not None:
        if eps < maxeps:
            updGaint = npla.solve(epsP+np.eye(epsP.shape[0]), baseGain.T)
            return updGaint.T, True
    elif maxfac is not None:
        if (1+eps)/(1-eps) < maxfac and eps < 1:
            updGaint = npla.solve(epsP+np.eye(epsP.shape[0]), baseGain.T)
            return updGaint.T, True

    return None, False
