import numpy as np

import dolfin_navier_scipy.data_output_utils as dou

import sadptprj_riclyap_adi.proj_ric_utils as pru
import sadptprj_riclyap_adi.bal_trunc_utils as btu

pymess = False
pymess_dict = {}
plain_bt = False


def get_ric_facs(fmat=None, mmat=None, jmat=None, bmat=None, cmat=None,
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


def get_rl_projections(mmat=None, fmat=None, jmat=None, bmat=None, cmat=None,
                       trunc_lqgbtcv=1e-6,
                       Rmhalf=None, Rmo=None,
                       Re=None, fdstr=None, truncstr=None, robit=False,
                       trytofail=False, get_fdstr=None, use_ric_ini=None,
                       nwtn_adi_dict=None, multiproc=False, checktheres=False):
    try:
        tl = dou.load_npa(fdstr + '__tl' + truncstr)
        tr = dou.load_npa(fdstr + '__tr' + truncstr)
        print(('loaded the left and right transformations: \n' +
               fdstr + '__tl/__tr' + truncstr))
        if robit:
            svs = dou.load_npa(fdstr + '__svs')

    except IOError:
        print(('computing the left and right transformations' +
               ' and saving to: \n' + fdstr + '__tl/__tr' + truncstr))

        try:
            zwc = dou.load_npa(fdstr + '__zwc')
            zwo = dou.load_npa(fdstr + '__zwo')
            print(('loaded factor of the Gramians: \n\t' +
                   fdstr + '__zwc/__zwo'))
        except IOError:

            print('computing the left and right transformations' +
                  ' and saving to:\n' + fdstr + '__tr/__tl' + truncstr)
            print('... done! - computing the left and right transformations')

            tl, tr, svs = btu.\
                compute_lrbt_transfos(zfc=zwc, zfo=zwo,
                                      mmat=mmat,
                                      trunck={'threshh': trunc_lqgbtcv})
            dou.save_npa(tl, fdstr + '__tl' + truncstr)
            dou.save_npa(tr, fdstr + '__tr' + truncstr)
            dou.save_npa(svs, fdstr + '__svs')

    print(('NV = {0}, NP = {2}, k = {1}'.format(tl.shape[0], tl.shape[1],
                                                jmat.shape[0])))
