import numpy as np
import mat73
from scipy.io import loadmat
from scipy.integrate import odeint
import sadptprj_riclyap_adi.bal_trunc_utils as btu
import sadptprj_riclyap_adi.lin_alg_utils as lau
import matplotlib.pyplot as plt
import lqgbt_oseen.nse_riccont_utils as nru

from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import scipy.sparse as sps

import integrate_lticl_utils as ilu
from dolfin_navier_scipy.stokes_navier_utils import solve_nse

verbose = False


def checkit(truncat=0.0001, usercgrams=False, Re=60, tE=10.,
            problem='cw', N=1,
            # problem='drc', N=2,
            Nts=5000, hinfcformula='ZDG', cpldscheme='trpz'):
    # %% Setup problem data.
    print('Setup problem data.')
    print('-------------------')

    fname = '/scratch/owncloud-gwdg/mpi-projects/18-hinf-lqgbt/data/' + \
            'cylinderwake_Re{0}.0'.format(Re) + \
            '_gamma1.0_NV41718_Bbcc_C31_palpha1e-05__mats'

    dimdict = {'cw': {1: 4171831},
               'drc': {2: 4552841}}
    cdim = dimdict[problem][N]
    fname = 'testdata/oseen_sys_{0}{1}.01.0_{2}-16'.\
        format(problem, Re, cdim)
    matdict = loadmat(fname)
    mmat = matdict['mmat']
    amat = matdict['amat']
    cmat = matdict['cmat']
    bmat = matdict['bmat']
    jmat = matdict['jmat']
    vinf = matdict['v_ss_nse']
    pinf = matdict['p_ss_nse']
    fv = matdict['fv']
    fp = matdict['fp']

    NV = amat.shape[0]
    dt = tE/(Nts+1)
    trange = np.linspace(0, tE, Nts+1)

    cntres = jmat @ vinf - fp
    print('cnt res: {0}', np.linalg.norm(cntres))
    momres = amat@vinf + jmat.T@pinf + fv
    print('mom res: {0}', np.linalg.norm(momres))

    def obsdrft(t):
        return 0.  # -hbystar

    NY, NU = cmat.shape[0], bmat.shape[1]
    hN = 5
    linobsrvdct = dict(ha=np.zeros((hN, hN)), hc=np.zeros((NU, hN)),
                       hb=np.zeros((hN, NY)), drift=obsdrft,
                       inihx=np.zeros((hN, 1)))

    sndict = dict(A=-amat, M=mmat, J=jmat,
                  b_mat=bmat, cv_mat=cmat,
                  iniv=vinf, invinds=np.arange(NV),
                  stokes_flow=True, time_int_scheme='cnab',
                  fv=fv, fp=fp,
                  t0=0., tE=tE, Nts=Nts,
                  # closed_loop=True, dynamic_feedback=True,
                  return_y_list=True,
                  treat_nonl_explicit=True)

    ylist = solve_nse(**sndict)
    ylist = [yl.flatten() for yl in ylist]
    plt.figure(55)
    plt.plot(trange, ylist, label='dns.solve_nse')

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

    ak_mat, bk_mat, ck_mat, xok, xck = nru.\
        get_prj_model(mmat=mmat, fmat=amat, jmat=None,
                      zwo=zwo, zwc=zwc,
                      tl=tl, tr=tr,
                      bmat=bmat, cmat=cmat)
    fvk = tl.T @ fv
    inivk = tl.T @ (mmat @ vinf)
    prinivk = tl.T @ (mmat @ (tr @ (tl.T @ (mmat @ vinf))))

    pev = vinf - tr @ inivk
    nrmpev = np.sqrt(pev.T @ mmat @ pev) / np.sqrt(vinf.T @ mmat @ vinf)
    pey = np.linalg.norm(cmat@vinf - ck_mat @ inivk)
    print('relnorm projection error in v: {0}'.format(nrmpev))
    print('projection error in y: {0}'.format(pey))
    print('projection?: {0}'.format(np.linalg.norm(prinivk-inivk)))

    def fwdsimrhs(vvec, t):
        return (ak_mat@vvec).flatten() + fvk.flatten()

    fwdsol = odeint(fwdsimrhs, inivk.flatten(), trange)
    yodeintt = fwdsol@ck_mat.T
    plt.figure(555)
    plt.plot(trange, yodeintt, label='rom odeint')
    return
    # import ipdb
    # ipdb.set_trace()

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
    if verbose:
        print('\nred-lin-CL-mat:\n max |EV|: {0},\n min |EV|: {1}'.
              format(fullevls[mxevidx], fullevls[mnevidx]))
    print(' * exp-Euler s-radius: {0}'.
          format(np.max(np.abs(1-tE/Nts*fullevls))))

    def fullsys(t, y):
        return fullrmmat@y

    hN = ak_mat.shape[0]

    impitmat = np.linalg.inv(np.eye(2*hN)-dt*fullrmmat)
    impitmatevls = np.linalg.eigvals(impitmat)
    print(' * imp-Euler s-radius: {0}'.format(np.max(np.abs(impitmatevls))))

    imptrprulevls = 1/(1-.5*dt*fullevls)*(1+.5*dt*fullevls)
    print(' * imp-trprul s-radius: {0}'.format(np.max(np.abs(imptrprulevls))))

    # xk = np.ones((ak_mat.shape[0], 1))
    xk = np.random.randn(ak_mat.shape[0], 1)
    hxk = 0*xk
    hxkxk = np.copy(np.vstack([hxk, xk]))
    # sollist = [np.copy(hxkxk.flatten()[0])]
    # sollist = [hxkxk[hN]]
    sollist = [hxkxk.flatten()]
    for kkk in range(Nts):
        hxkxk = impitmat @ hxkxk
        # sollist.append(hxkxk.flatten()[0].copy())
        # sollist.append(hxkxk[hN])
        sollist.append(hxkxk.flatten())
    print('coupled implicit:', np.linalg.norm(hxkxk[:hN]),
          np.linalg.norm(hxkxk[hN:]))
    plt.figure(22)
    plt.plot(np.linspace(0, tE, Nts+1), sollist)  # , label='coupled implicit')

    intgrtrdict = dict(xz=xk.copy(), hxz=hxk.copy(),
                       sys_a=ak_mat, sys_b=bk_mat, sys_c=ck_mat, sys_m=None,
                       obs_a=obs_ak, obs_b=obs_bk, obs_c=obs_ck, obs_m=None,
                       tE=tE, Nts=Nts, retylist=True, dense=True)

    plt.legend()

    ylist_ie = ilu.cpld_implicit_solver(**intgrtrdict, scheme='IE')
    plt.figure(33)
    plt.plot(trange, ylist_ie, label='ilu cpld rom IE')
    plt.legend()
    # plt.figure(333)
    # ylist_ie = ilu.cpld_implicit_solver(**intgrtrdict, scheme='trpz')
    # plt.plot(trange, ylist_ie, label='ilu cpld rom trpz')
    # plt.legend()
    plt.figure(3333)
    ylist_ie = ilu.cpld_implicit_solver(**intgrtrdict, scheme='BDF2')
    plt.plot(trange, ylist_ie, label='ilu cpld rom bdf2')
    plt.legend()

    fom_simu = False
    if fom_simu:
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
        ylist_ie = ilu.cpld_implicit_solver(**spsintgrtrdict,
                                            scheme=cpldscheme)
        plt.figure(34)
        plt.plot(trange, ylist_ie, label='ilu cpld fom ie')
        plt.legend()
        plt.show()

    # hbystar = obs_bk.dot(c_mat.dot(v_ss_nse[invinds]))
    def obsdrft(t):
        return 0.  # -hbystar

    linobsrvdct = dict(ha=obs_ak, hc=obs_ck, hb=obs_bk,
                       drift=obsdrft, inihx=np.zeros((obs_bk.shape[0], 1)))
    NV = amat.shape[0]
    sndict = dict(A=amat, M=mmat, J=jmat,
                  b_mat=bmat, cv_mat=cmat,
                  iniv=1e-5*np.ones((NV, 1)), invinds=np.arange(NV),
                  stokes_flow=True, time_int_scheme='cnab',
                  t0=0., tE=tE, Nts=Nts,
                  closed_loop=True, dynamic_feedback=True,
                  return_y_list=True,
                  treat_nonl_explicit=True,
                  dyn_fb_dict=linobsrvdct)

    ylist = solve_nse(**sndict)
    ylist = [yl.flatten() for yl in ylist]
    plt.figure(55)
    plt.plot(trange, ylist, label='dns.solve_nse')

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
        fomtrpzstp = ilu.get_fom_trpz_step(A=amat, M=mmat, J=jmat, dt=dt)
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
    btE, bNts = 1., 80000
    scaletest = .01
    # checkit(truncat=0.01, usercgrams=True, Re=60, tE=1., Nts=250,
    #         hinfcformula='ZDG')
    checkit(truncat=0.01, usercgrams=True, Re=60, tE=scaletest*btE,
            Nts=np.int(np.ceil(scaletest*bNts)),
            hinfcformula='ZDG', cpldscheme='trpz')
    # checkit(truncat=0.01, usercgrams=True, Re=60, tE=10., Nts=4500,
    #         hinfcformula='ZDG')
    plt.legend()
    plt.show()
