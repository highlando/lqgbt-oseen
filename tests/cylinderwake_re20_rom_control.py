import numpy as np
import mat73
from scipy.io import loadmat
import sadptprj_riclyap_adi.bal_trunc_utils as btu
import sadptprj_riclyap_adi.lin_alg_utils as lau
import matplotlib.pyplot as plt
import lqgbt_oseen.nse_riccont_utils as nru

from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import scipy.sparse as sps
from scipy.sparse.linalg import factorized

import integrate_lticl_utils as ilu


def checkit(truncat=0.01, usercgrams=False, Re=60, tE=10.,
            Nts=5000, hinfcformula='ZDG', cpldscheme='trpz'):
    # %% Setup problem data.
    print('Setup problem data.')
    print('-------------------')

    fname = '/scratch/owncloud-gwdg/mpi-projects/18-hinf-lqgbt/data/' + \
            'cylinderwake_Re{0}.0'.format(Re) + \
            '_gamma1.0_NV41718_Bbcc_C31_palpha1e-05__mats'
    matdict = loadmat(fname)
    mmat = matdict['mmat']
    amat = matdict['amat']
    cmat = matdict['cmat']
    bmat = matdict['bmat']
    jmat = matdict['jmat']

    NP, NV = jmat.shape
    NU, NY = bmat.shape[1], cmat.shape[0]

    # %% Load Riccati results.
    print('Load Riccati results.')
    print('---------------------')

    fname = '/scratch/owncloud-gwdg/mpi-projects/18-hinf-lqgbt/results/' + \
            'cylinderwake_re{0}_hinf.mat'.format(Re)
    lmd = mat73.loadmat(fname)
    zwc = lmd['outRegulator']['Z']
    zwo = lmd['outFilter']['Z']
    gam = lmd['gam']

    scfc = np.sqrt(1-1/gam**2)  # [2]

    print('Check the Riccati Residuals')
    print('------------')

    cricres = lau.comp_sqfnrm_factrd_riccati_res(M=mmat, A=amat, B=scfc*bmat,
                                                 C=cmat, Z=zwc)
    normxc = ((zwc.T@zwc) ** 2).sum(-1).sum()
    print('c-ric-res^2: ', cricres)
    print('norm-xc^2: ', normxc)
    print('relres: ', np.sqrt(cricres/normxc))
    oricres = lau.comp_sqfnrm_factrd_riccati_res(M=mmat, A=amat.T,
                                                 B=scfc*cmat.T,
                                                 C=bmat.T, Z=zwo)
    normxo = ((zwo.T@zwo) ** 2).sum(-1).sum()
    print('oricres^2: ', oricres)
    print('normxo^2: ', normxo)
    print('relres: ', np.sqrt(oricres/normxo))

    print('Compute ROM.')
    print('------------')

    tl, tr, svs = btu.\
        compute_lrbt_transfos(zfc=zwc, zfo=zwo, mmat=mmat,
                              trunck={'threshh': truncat})
    cdim = tl.shape[1]
    print('Dimension of the controller: ', cdim)
    beta = np.sqrt(1-1/gam**2)
    trnctsvs = svs[cdim:].flatten()
    epsilon = 2 * (trnctsvs / np.sqrt(1+beta**2*trnctsvs**2)).sum()
    print('stable if `<0`: ', epsilon*beta-1/gam)

    # plt.figure(1)
    # plt.semilogy(svs[:cntrlsz], 'x')
    # plt.show(block=False)
    # import pdb
    # pdb.set_trace()

    def get_fom_trpz_step(M=None, A=None, dt=None, J=None):
        NP, NV = J.shape
        sysm1 = sps.hstack([M-.5*dt*A, J.T], format='csr')
        sysm2 = sps.hstack([J, sps.csr_matrix((NP, NP))], format='csr')
        sadptmat = sps.vstack([sysm1, sysm2], format='csr')
        alu = factorized(sadptmat)

        def fom_trpz_step(rhsv=None):
            vpvec = alu(np.r_[rhsv.flatten(), np.zeros((NP, ))])
            return vpvec[:NV].reshape((NV, 1))

        return fom_trpz_step

    ak_mat, bk_mat, ck_mat, xok, xck = nru.\
        get_prj_model(mmat=mmat, fmat=amat, jmat=None,
                      zwo=zwo, zwc=zwc,
                      tl=tl, tr=tr,
                      bmat=bmat, cmat=cmat)
    riccres = ak_mat.T.dot(xck) + xck.dot(ak_mat) - \
        (1-1/gam**2)*xck.dot(bk_mat).dot(bk_mat.T.dot(xck)) +\
        ck_mat.T.dot(ck_mat)
    print('red c-ric-res', np.linalg.norm(riccres))

    ricores = ak_mat.dot(xok) + xok.dot(ak_mat.T) - \
        (1-1/gam**2)*xok.dot(ck_mat.T).dot(ck_mat.dot(xok)) + \
        bk_mat.dot(bk_mat.T)

    print('red o-ric-res', np.linalg.norm(ricores))

    # Recompute the red Grams
    rcxok = solve_continuous_are(ak_mat.T, scfc*ck_mat.T, bk_mat.dot(bk_mat.T),
                                 np.eye(ck_mat.shape[0]))
    rcxokres = ak_mat.dot(rcxok) + rcxok.dot(ak_mat.T) - \
        (1-1/gam)*(1+1/gam)*rcxok.dot(ck_mat.T).dot(ck_mat.dot(rcxok)) + \
        bk_mat.dot(bk_mat.T)
    print('rc-o-ric-res|', np.linalg.norm(rcxokres))

    rcxck = solve_continuous_are(ak_mat, scfc*bk_mat, ck_mat.T.dot(ck_mat),
                                 np.eye(bk_mat.shape[1]))
    rcxckres = ak_mat.T.dot(rcxck) + rcxck.dot(ak_mat) - \
        (1-1/gam)*(1+1/gam)*rcxck.dot(bk_mat).dot(bk_mat.T@rcxok) + \
        ck_mat.T@ck_mat
    print('rc-c-ric-res|', np.linalg.norm(rcxckres))
    if usercgrams:
        xck, xok = rcxck, rcxok

    zk = np.linalg.inv(np.eye(xck.shape[0])
                       - 1./gam**2*xok.dot(xck))
    if hinfcformula == 'ZDG':
        # ## ZDG p. 412 formula
        obs_ak = (ak_mat - ((1. - 1./gam**2)*bk_mat) @ (bk_mat.T@xck)
                  - (zk @ (xok@ck_mat.T)) @ ck_mat)
        obs_bk = zk @ (xok@ck_mat.T)
        obs_ck = -bk_mat.T @ xck
    elif hinfcformula == 'MusG':
        # obsvakmat = (ak_mat
        #              - (1. - 1./gam**2)*np.dot(np.dot(xok, ck_mat.T), ck_mat)
        #              - np.dot(bk_mat, np.dot(bk_mat.T, xck).dot(zk)))
        raise NotImplementedError()

    fullrmmat = np.vstack([np.hstack([obs_ak, obs_bk@ck_mat]),
                           np.hstack([bk_mat@obs_ck, ak_mat])])
    fullevls = np.linalg.eigvals(fullrmmat)
    mxevidx = np.argmax(np.abs(np.real(fullevls)))
    mnevidx = np.argmin(np.abs(np.real(fullevls)))
    print('\nred-lin-CL-mat:\n max |EV|: {0},\n min |EV|: {1}'.
          format(fullevls[mxevidx], fullevls[mnevidx]))
    print(' * exp-Euler s-radius: {0}'.
          format(np.max(np.abs(1-tE/Nts*fullevls))))

    def fullsys(t, y):
        return fullrmmat@y

    hN = ak_mat.shape[0]
    dt = tE/(Nts+1)
    trange = np.linspace(0, tE, Nts+1)

    impitmat = np.linalg.inv(np.eye(2*hN)-dt*fullrmmat)
    impitmatevls = np.linalg.eigvals(impitmat)
    print(' * imp-Euler s-radius: {0}'.format(np.max(np.abs(impitmatevls))))

    # expitmatevls = np.linalg.eigvals(np.eye(2*hN)+dt*fullrmmat)
    # print('exp-itmats-evls:', expitmatevls)

    imptrprulevls = 1/(1-.5*dt*fullevls)*(1+.5*dt*fullevls)
    print(' * imp-trprul s-radius: {0}'.format(np.max(np.abs(imptrprulevls))))

    dcpld_xplct = True
    # dcpld_mxplct = True
    sim_xplct = True

    if dcpld_xplct:
        xk = np.ones((ak_mat.shape[0], 1))
        hxk = 0*xk
        for kkk in range(Nts):
            uk = obs_ck @ hxk
            yk = ck_mat @ xk
            xk = xk + dt*(ak_mat @ xk + bk_mat @ uk)
            hxk = hxk + dt*(obs_ak @ hxk + obs_bk @ yk)
        print('decoupled explicit:', np.linalg.norm(hxk), np.linalg.norm(xk))

    # if dcpld_mxplct:
    #     xk = np.ones((ak_mat.shape[0], 1))
    #     hxk = 0*xk
    #     obsakpmo = np.linalg.inv(np.eye(hN)-dt*obs_ak)

    #     for kkk in range(Nts):
    #         uk = obs_ck @ hxk
    #         xk = xk + dt*ak_mat @ xk + dt * bk_mat @ uk
    #         yk = ck_mat @ xk
    #         hxk = obsakpmo @ (hxk + dt*obs_bk @ yk)
    #     print('decoupled imex:', np.linalg.norm(hxk), np.linalg.norm(xk))

    dcpld_xplct_scnd = True
    if dcpld_xplct_scnd:
        xk = np.ones((ak_mat.shape[0], 1))
        sollist = [xk[0]]
        # sollist = [np.copy(xk.flatten())]
        hxk = 0*xk
        uk = obs_ck @ hxk
        yk = ck_mat @ xk

        # prediction
        pxkk = xk + dt*(ak_mat@xk + bk_mat@uk)
        pykk = ck_mat @ pxkk
        phxkk = hxk + dt*(obs_ak@hxk + obs_bk@yk)

        # correction
        hxkk = hxk + dt/2*(obs_ak@(hxk+phxkk)+obs_bk@(pykk+yk))
        ukk = obs_ck@hxkk
        xkk = xk + dt/2*(ak_mat@(xk+pxkk) + bk_mat@(ukk+uk))
        ykk = ck_mat@xkk
        # sollist.append(np.copy(xkk.flatten()[0]))
        sollist.append(xkk[0])

        obsitmat = np.linalg.inv(np.eye(hN)-dt/2*obs_ak)
        sysitmat = np.linalg.inv(np.eye(hN)-dt/2*ak_mat)
        for kkk in range(1, Nts):
            # xkk = sysitmat@(xk+dt/2*(ak_mat@xk + bk_mat@(3*ukk-uk)))
            xkkk = xkk+dt/2*(ak_mat@(3*xkk-xk) + bk_mat@(3*ukk-uk))
            # hxkk = obsitmat @ (hxk+dt/2*(obs_ak@hxk + obs_bk@(ykk+yk)))
            hxkkk = hxkk+dt/2*(obs_ak@(3*hxkk-hxk) + obs_bk@(3*ykk-yk))
            xk = xkk
            xkk = xkkk
            hxk = hxkk
            hxkk = hxkkk
            uk, ukk = obs_ck@hxk, obs_ck@hxkk
            yk, ykk = ck_mat@xk, ck_mat@xkk
            # sollist.append(np.copy(xkk.flatten()[0]))
            sollist.append(xkk[0])
        print('decoupled explicit 2nd:',
              np.linalg.norm(hxkk), np.linalg.norm(xkk))
        plt.figure(22)
        plt.plot(np.linspace(0, tE, Nts+1), sollist, label='dcupld xplct 2nd')

    dcpld_mplct_scnd = True
    if dcpld_mplct_scnd:
        xk = np.ones((ak_mat.shape[0], 1))
        sollist = [xk[0]]
        hxk = 0*xk
        uk = obs_ck @ hxk
        yk = ck_mat @ xk

        # prediction
        pxkk = xk + dt*(ak_mat@xk + bk_mat@uk)
        pykk = ck_mat @ pxkk
        phxkk = hxk + dt*(obs_ak@hxk + obs_bk@yk)

        # correction
        hxkk = hxk + dt/2*(obs_ak@(hxk+phxkk)+obs_bk@(pykk+yk))
        ukk = obs_ck@hxkk
        xkk = xk + dt/2*(ak_mat@(xk+pxkk) + bk_mat@(ukk+uk))
        ykk = ck_mat@xkk
        sollist.append(xkk[0])

        sysitmat = np.linalg.inv(np.eye(hN)-dt/2*ak_mat)
        obsitmat = np.linalg.inv(np.eye(hN)-dt/2*obs_ak)
        for kkk in range(1, Nts):
            xkkk = sysitmat@(xkk+dt/2*(ak_mat@xkk + bk_mat@(3*ukk-uk)))
            # xkkk = xkk+dt/2*(ak_mat@(3*xkk-xk) + bk_mat@(3*ukk-uk))
            # hxkk = obsitmat @ (hxk+dt/2*(obs_ak@hxk + obs_bk@(ykk+yk)))
            # hxkkk = obsitmat @ (hxkk+dt/2*(obs_ak@hxkk + obs_bk@(3*ykk-yk)))
            # hxkkk = hxkk+dt/2*(obs_ak@(3*hxkk-hxk) + obs_bk@(3*ykk-yk))
            xk = xkk
            xkk = xkkk
            yk, ykk = ck_mat@xk, ck_mat@xkk
            hxkkk = obsitmat @ (hxkk+dt/2*(obs_ak@hxkk + obs_bk@(ykk+yk)))
            hxk = hxkk
            hxkk = hxkkk
            uk, ukk = obs_ck@hxk, obs_ck@hxkk
            # sollist.append(np.copy(xkk.flatten()[0]))
            sollist.append(xkk[0])
        print('decoupled implicit 2nd:',
              np.linalg.norm(hxkk), np.linalg.norm(xkk))
        plt.plot(np.linspace(0, tE, Nts+1), sollist, label='dcpld mplct 2nd')

    if sim_xplct:
        xk = np.ones((ak_mat.shape[0], 1))
        hxk = 0*xk
        hxkxk = np.copy(np.vstack([hxk, xk]))
        for kkk in range(Nts):
            hxkxk = hxkxk + dt*fullrmmat@hxkxk
        print('coupled explicit:', np.linalg.norm(hxkxk[:hN]),
              np.linalg.norm(hxkxk[hN:]))

    # xk = np.ones((ak_mat.shape[0], 1))
    xk = np.random.randn(ak_mat.shape[0], 1)
    hxk = 0*xk
    hxkxk = np.copy(np.vstack([hxk, xk]))
    # sollist = [np.copy(hxkxk.flatten()[0])]
    sollist = [hxkxk[hN]]
    for kkk in range(Nts):
        hxkxk = impitmat @ hxkxk
        # sollist.append(hxkxk.flatten()[0].copy())
        sollist.append(hxkxk[hN])
    print('coupled implicit:', np.linalg.norm(hxkxk[:hN]),
          np.linalg.norm(hxkxk[hN:]))
    plt.plot(np.linspace(0, tE, Nts+1), sollist, label='coupled implicit')

    intgrtrdict = dict(xz=xk.copy(), hxz=hxk.copy(),
                       sys_a=ak_mat, sys_b=bk_mat, sys_c=ck_mat, sys_m=None,
                       obs_a=obs_ak, obs_b=obs_bk, obs_c=obs_ck, obs_m=None,
                       tE=tE, Nts=Nts, retylist=True, dense=True)

    plt.legend()

    ylist_ie = ilu.cpld_implicit_solver(**intgrtrdict, scheme='IE')
    plt.figure(33)
    plt.plot(trange, ylist_ie, label='ilu cpld rom IE')
    plt.legend()
    plt.figure(333)
    ylist_ie = ilu.cpld_implicit_solver(**intgrtrdict, scheme='trpz')
    plt.plot(trange, ylist_ie, label='ilu cpld rom trpz')
    plt.legend()
    return
    bigamat = sps.vstack([sps.hstack([amat, jmat.T]),
                          sps.hstack([jmat, sps.csr_matrix((NP, NP))])],
                         format='csr')
    bigmmat = sps.block_diag([mmat, sps.csr_matrix((NP, NP))])
    bigbmat = sps.vstack([bmat, sps.csr_matrix((NP, NU))])
    bigcmat = sps.hstack([cmat, sps.csr_matrix((NY, NP))])

    spsintgrtrdict = dict(xz=np.vstack([tr@xk.copy(), np.zeros((NP, 1))]),
                          hxz=hxk.copy(),
                          sys_a=bigamat, sys_b=bigbmat,
                          sys_c=bigcmat, sys_m=bigmmat,
                          obs_a=sps.csr_matrix(obs_ak),
                          obs_b=sps.csr_matrix(obs_bk),
                          obs_c=sps.csr_matrix(obs_ck),
                          obs_m=None,
                          tE=tE, Nts=Nts, retylist=True, dense=False)
    ylist_ie = ilu.cpld_implicit_solver(**spsintgrtrdict, scheme=cpldscheme)
    plt.figure(34)
    plt.plot(trange, ylist_ie, label='ilu cpld fom ie')
    plt.legend()
    plt.show()

    return

    xk = np.ones((ak_mat.shape[0], 1))
    hxk = 0*xk
    hxkxk = np.copy(np.vstack([hxk, xk]))
    bdfsol = solve_ivp(fullsys, y0=hxkxk.flatten(),
                       t_span=(0, tE), t_eval=trange)
    intodey = bdfsol['y']
    print('int_ode:', np.linalg.norm(intodey[:hN, -1]),
          np.linalg.norm(intodey[hN:, -1]))

    # plt.figure()
    plt.plot(trange, bdfsol['y'][hN, :], label='odeint')

    fom_dcpld_mplct_scnd = True
    if fom_dcpld_mplct_scnd:
        fomtrpzstp = get_fom_trpz_step(A=amat, M=mmat, J=jmat, dt=dt)
        xk = np.ones((amat.shape[0], 1))
        sollist = [xk[0]]
        hxk = np.zeros((hN, 1))
        uk = obs_ck @ hxk
        yk = cmat @ xk

        # prediction
        pxkk = fomtrpzstp(mmat*xk + .5*dt*amat@xk + dt*bmat@uk)
        pykk = cmat @ pxkk
        phxkk = hxk + dt*(obs_ak@hxk + obs_bk@yk)

        # correction
        hxkk = hxk + dt/2*(obs_ak@(hxk+phxkk)+obs_bk@(pykk+yk))
        ukk = obs_ck@hxkk
        xkk = xk + dt/2*(amat@(xk+pxkk) + bmat@(ukk+uk))
        xkk = fomtrpzstp(mmat*xk + .5*dt*amat@xk + .5*dt*bmat@(uk+ukk))
        ykk = cmat@xkk
        sollist.append(np.linalg.norm(xkk))

        obsitmat = np.linalg.inv(np.eye(hN)-dt/2*obs_ak)
        for kkk in range(1, Nts):
            # xkkk = fomtrpzstp(mmat@xkk+.5*dt*(amat@xkk + bmat@(3*ukk-uk)))
            xkkk = fomtrpzstp(mmat@xkk+.5*dt*(amat@xkk + bmat@(ukk+uk)))
            # xkkk = xkk+dt/2*(ak_mat@(3*xkk-xk) + bk_mat@(3*ukk-uk))
            # hxkk = obsitmat @ (hxk+dt/2*(obs_ak@hxk + obs_bk@(ykk+yk)))
            # hxkkk = obsitmat @ (hxkk+dt/2*(obs_ak@hxkk + obs_bk@(3*ykk-yk)))
            # hxkkk = hxkk+dt/2*(obs_ak@(3*hxkk-hxk) + obs_bk@(3*ykk-yk))
            xk = xkk
            xkk = xkkk
            yk, ykk = cmat@xk, cmat@xkk
            hxkkk = obsitmat @ (hxkk+dt/2*(obs_ak@hxkk + obs_bk@(ykk+yk)))
            hxk = hxkk
            hxkk = hxkkk
            uk, ukk = obs_ck@hxk, obs_ck@hxkk
            # sollist.append(np.copy(xkk.flatten()[0]))
            # sollist.append(xkk[0])
            sollist.append(np.linalg.norm(xkk))
        print('fom decoupled implicit 2nd:',
              np.linalg.norm(hxkk), np.linalg.norm(xkk))
        plt.plot(np.linspace(0, tE, Nts+1), sollist,
                 label='fom dcpld mplct 2nd')


if __name__ == '__main__':
    btE, bNts = 1., 400
    scaletest = .8
    # checkit(truncat=0.01, usercgrams=True, Re=60, tE=1., Nts=250,
    #         hinfcformula='ZDG')
    checkit(truncat=0.00001, usercgrams=True, Re=60, tE=scaletest*btE,
            Nts=np.int(np.ceil(scaletest*bNts)),
            hinfcformula='ZDG', cpldscheme='trpz')
    # checkit(truncat=0.01, usercgrams=True, Re=60, tE=10., Nts=4500,
    #         hinfcformula='ZDG')
    plt.legend()
    plt.show()
