import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import factorized


def cpld_implicit_solver(xz=None, hxz=None,
                         sys_a=None, sys_b=None, sys_c=None, sys_m=None,
                         obs_a=None, obs_b=None, obs_c=None, obs_m=None,
                         tE=None, Nts=None,
                         retylist=True, dense=False, scheme='IE'):

    N, hN = sys_a.shape[0], obs_a.shape[0]
    dt = tE/(Nts+1)
    if dense:
        if sys_m is None:
            sys_m = np.eye(N)
        if obs_m is None:
            obs_m = np.eye(hN)
        cpldmmat = np.block([[sys_m, np.zeros((N, hN))],
                             [np.zeros((hN, N)), obs_m]])

        sboc = sys_b@obs_c
        obsc = obs_b@sys_c
        cpldamat = np.vstack([np.hstack([sys_a, sboc]),
                              np.hstack([obsc, obs_a])])
        if scheme == 'IE':
            sysmati = np.linalg.inv(cpldmmat - dt*cpldamat)

            def incrementplease(xk, xp=None, fp=None, fkk=None):
                rhs = cpldmmat@xk
                return sysmati@rhs

        elif scheme == 'trpz':
            sysmati = np.linalg.inv(cpldmmat - .5*dt*cpldamat)

            def incrementplease(xk, xp=None, fp=None, fkk=None):
                rhs = cpldmmat@xk + .5*dt*cpldamat@xk
                return sysmati@rhs

        elif scheme == 'BDF2':
            sysmati = np.linalg.inv(cpldmmat - 2/3*dt*cpldamat)

            def incrementplease(xc, xp=None, fc=None, fn=None, fp=None):
                rhs = 4/3*cpldmmat@xc - 1/3*cpldmmat@xp
                return sysmati@rhs

    else:
        if sys_m is None:
            sys_m = sps.eye(N, format='csr')
        if obs_m is None:
            obs_m = sps.eye(hN, format='csr')
        cpldmmat = sps.block_diag([sys_m, obs_m])

        sboc = sps.csr_matrix(sys_b@obs_c)
        sboc.eliminate_zeros()
        obsc = sps.csr_matrix(obs_b@sys_c)
        obsc.eliminate_zeros()
        cpldamat = sps.vstack([sps.hstack([sys_a, sboc]),
                               sps.hstack([obsc, obs_a])],
                              format='csr')
        if scheme == 'IE':
            sysmati = sps.linalg.factorized(cpldmmat - dt*cpldamat)

            def incrementplease(xk, xp=None, fp=None, fkk=None):
                rhs = cpldmmat@xk
                return sysmati(rhs.flatten()).reshape((N+hN, 1))

        elif scheme == 'trpz':
            sysmati = sps.linalg.factorized(cpldmmat - .5*dt*cpldamat)

            def incrementplease(xk, xp=None, fp=None, fkk=None):
                rhs = cpldmmat@xk + .5*dt*cpldamat@xk
                return sysmati(rhs.flatten()).reshape((N+hN, 1))

        elif scheme == 'BDF2':
            sysmati = sps.linalg.factorized(cpldmmat - 2/3*dt*cpldamat)

            def incrementplease(xc, xp=None, fc=None, fn=None, fp=None):
                rhs = 4/3*cpldmmat@xc - 1/3*cpldmmat@xp
                return sysmati(rhs.flatten()).reshape((N+hN, 1))

    xc = np.vstack([xz, hxz])
    ylist = [(sys_c@xz).flatten()]
    nsecs = np.arange(0, Nts, np.int(np.ceil(Nts/6))).tolist()
    nsecs.append(Nts)
    if scheme == 'BDF2':
        print('BDF2: init with Heun')
        xn = linsys_heun_upd(xc, mmat=cpldmmat, amat=cpldamat, dt=dt)
        ylist.append((sys_c@(xn[:N, :])).flatten())
        xp = xc
        xc = xn
        nsecs[0] = 1  # start actual integration at tk=1 (rather than tk=0)
    else:
        xp = None
    for k, kk in enumerate(nsecs[:-1]):
        print(scheme, ': time int {0:2.1f}% complete'.format(kk/Nts*100))
        for kkk in range(kk, nsecs[k+1]):
            xn = incrementplease(xc, xp=xp)
            # print(kkk, np.linalg.norm(xkhxk), xkhxk.shape)
            ylist.append((sys_c@(xn[:N, :])).flatten())
            xp = xc
            xc = xn

    return ylist


def linsys_heun_upd(xc, amat=None, mmat=None, ffunc=None, dense=True,
                    dt=None, IE=True):
    if IE:
        if dense:
            xpp = np.linalg.solve(mmat-dt*amat, xc)
        else:
            pass
    else:
        xpp = xc + dt*amat@xc
    xn = xc + .5*dt*amat@(xc+xpp)

    return xn


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


def dcpld_imex_solvers(xz=None, hxz=None,
                       sys_a=None, sys_b=None, sys_c=None, sys_m=None,
                       obs_a=None, obs_b=None, obs_c=None, obs_m=None,
                       tE=None, Nts=None,
                       fullrmmat=None,
                       retylist=True, dense=False, scheme='IE'):

    hN = obs_a.shape[0]
    dt = tE/(Nts+1)
    dcpld_xplct = False
    sim_xplct = False

    if dcpld_xplct:
        xk = np.ones((sys_a.shape[0], 1))
        hxk = 0*xk
        for kkk in range(Nts):
            uk = obs_c @ hxk
            yk = sys_c @ xk
            xk = xk + dt*(sys_a @ xk + sys_b @ uk)
            hxk = hxk + dt*(obs_a @ hxk + obs_b @ yk)
        print('decoupled explicit:', np.linalg.norm(hxk), np.linalg.norm(xk))

    dcpld_mxplct = True
    if dcpld_mxplct:
        xk = np.ones((sys_a.shape[0], 1))
        hxk = 0*xk
        obsakpmo = np.linalg.inv(np.eye(hN)-dt*obs_a)

        for kkk in range(Nts):
            uk = obs_c @ hxk
            xk = xk + dt*sys_a @ xk + dt * sys_b @ uk
            yk = sys_c @ xk
            hxk = obsakpmo @ (hxk + dt*obs_b @ yk)
        print('decoupled imex:', np.linalg.norm(hxk), np.linalg.norm(xk))

    dcpld_xplct_scnd = False
    if dcpld_xplct_scnd:
        xk = np.ones((sys_a.shape[0], 1))
        sollist = [xk[0]]
        # sollist = [np.copy(xk.flatten())]
        hxk = 0*xk
        uk = obs_c @ hxk
        yk = sys_c @ xk

        # prediction
        pxkk = xk + dt*(sys_a@xk + sys_b@uk)
        pykk = sys_c @ pxkk
        phxkk = hxk + dt*(obs_a@hxk + obs_b@yk)

        # correction
        hxkk = hxk + dt/2*(obs_a@(hxk+phxkk)+obs_b@(pykk+yk))
        ukk = obs_c@hxkk
        xkk = xk + dt/2*(sys_a@(xk+pxkk) + sys_b@(ukk+uk))
        ykk = sys_c@xkk
        # sollist.append(np.copy(xkk.flatten()[0]))
        sollist.append(xkk[0])

        obsitmat = np.linalg.inv(np.eye(hN)-dt/2*obs_a)
        sysitmat = np.linalg.inv(np.eye(hN)-dt/2*sys_a)
        for kkk in range(1, Nts):
            # xkk = sysitmat@(xk+dt/2*(sys_a@xk + sys_b@(3*ukk-uk)))
            xkkk = xkk+dt/2*(sys_a@(3*xkk-xk) + sys_b@(3*ukk-uk))
            # hxkk = obsitmat @ (hxk+dt/2*(obs_a@hxk + obs_b@(ykk+yk)))
            hxkkk = hxkk+dt/2*(obs_a@(3*hxkk-hxk) + obs_b@(3*ykk-yk))
            xk = xkk
            xkk = xkkk
            hxk = hxkk
            hxkk = hxkkk
            uk, ukk = obs_c@hxk, obs_c@hxkk
            yk, ykk = sys_c@xk, sys_c@xkk
            # sollist.append(np.copy(xkk.flatten()[0]))
            sollist.append(xkk[0])
        print('decoupled explicit 2nd:',
              np.linalg.norm(hxkk), np.linalg.norm(xkk))
        # plt.figure(22)
        # plt.plot(np.linspace(0, tE, Nts+1), sollist, label='dcpld xplct 2nd')

    dcpld_mplct_scnd = False
    if dcpld_mplct_scnd:
        xk = np.ones((sys_a.shape[0], 1))
        sollist = [xk[0]]
        hxk = 0*xk
        uk = obs_c @ hxk
        yk = sys_c @ xk

        # prediction
        pxkk = xk + dt*(sys_a@xk + sys_b@uk)
        pykk = sys_c @ pxkk
        phxkk = hxk + dt*(obs_a@hxk + obs_b@yk)

        # correction
        hxkk = hxk + dt/2*(obs_a@(hxk+phxkk)+obs_b@(pykk+yk))
        ukk = obs_c@hxkk
        xkk = xk + dt/2*(sys_a@(xk+pxkk) + sys_b@(ukk+uk))
        ykk = sys_c@xkk
        sollist.append(xkk[0])

        sysitmat = np.linalg.inv(np.eye(hN)-dt/2*sys_a)
        obsitmat = np.linalg.inv(np.eye(hN)-dt/2*obs_a)
        for kkk in range(1, Nts):
            xkkk = sysitmat@(xkk+dt/2*(sys_a@xkk + sys_b@(3*ukk-uk)))
            # xkkk = xkk+dt/2*(sys_a@(3*xkk-xk) + sys_b@(3*ukk-uk))
            # hxkk = obsitmat @ (hxk+dt/2*(obs_a@hxk + obs_b@(ykk+yk)))
            # hxkkk = obsitmat @ (hxkk+dt/2*(obs_a@hxkk + obs_b@(3*ykk-yk)))
            # hxkkk = hxkk+dt/2*(obs_a@(3*hxkk-hxk) + obs_b@(3*ykk-yk))
            xk = xkk
            xkk = xkkk
            yk, ykk = sys_c@xk, sys_c@xkk
            hxkkk = obsitmat @ (hxkk+dt/2*(obs_a@hxkk + obs_b@(ykk+yk)))
            hxk = hxkk
            hxkk = hxkkk
            uk, ukk = obs_c@hxk, obs_c@hxkk
            # sollist.append(np.copy(xkk.flatten()[0]))
            sollist.append(xkk[0])
        print('decoupled implicit 2nd:',
              np.linalg.norm(hxkk), np.linalg.norm(xkk))
        # plt.plot(np.linspace(0, tE, Nts+1), sollist, label='dcpld mplct 2nd')

    if sim_xplct:
        xk = np.ones((sys_a.shape[0], 1))
        hxk = 0*xk
        hxkxk = np.copy(np.vstack([hxk, xk]))
        for kkk in range(Nts):
            hxkxk = hxkxk + dt*fullrmmat@hxkxk
        print('coupled explicit:', np.linalg.norm(hxkxk[:hN]),
              np.linalg.norm(hxkxk[hN:]))
