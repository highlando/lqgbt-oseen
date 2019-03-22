import scipy.io
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import dolfin_navier_scipy.data_output_utils as dou
import matplotlib.pyplot as plt
import scipy.linalg as spla
import numpy as np

Re = 100
cyldim = 8
cyldiml = [3, 4, 5, 6, 7, 8]
palpha = 1e-5
ddir = 'data/'
problemname = 'cylinderwake'

debug = False
verbose = False
plotall = False
plotlast = True
compall = False
nevs = 50
sigma = .01  # for the shift and invert procedure
leftevecs = True
# compall = False
shiftall = True
bccontrol = True


def getthesysstr(cyldim=None, bccontrol=None, palpha=None):
    if bccontrol:
        Nlistbcc = [None, None, 9384, 19512, 28970, 38588, 48040, 58180, 68048]
        NN = Nlistbcc[cyldim]
        linsysstr = 'data/cylinderwake__mats_N{0}_Re{1}__penarob_palpha{2}'.\
            format(NN, Re, palpha)
    else:
        Nlist = [3022, 5812, None, 19468]
        NN = Nlist[cyldim]
        linsysstr = 'data/cylinderwake__mats_N{0}_Re{1}'.format(NN, Re)

    return linsysstr


def setup_oseencl_matslinops(M=None, A=None, B=None, C=None, J=None,
                             Mk=None, Ak=None, Bk=None, Ck=None,
                             Xck=None, Xok=None, XokF=None, XckF=None,
                             infevlshift=-10.):
    ''' setup of the coefficient matrices for the Oseen closed loop system

    [ M  0  0 ]                 [ A        -BBk.TXck              J.T ]
    [ 0  Mk 0 ] * (v, xk, p)' = [ XokCk.TC Ak-XokCk.TCk-BkBk.TXck 0   ] (vxkp)
    [ 0  0  0 ]                 [ J        0                      0   ]

    as linear operator and the shifted `bigM` matrix

    [ M    0  s*J.T ]
    [ 0    Mk 0     ]
    [ s*J  0  0     ]

    needed to compute the finite Eigenvalues of the closed loop system

    '''

    nv, k, np = M.shape[0], Mk.shape[0], B.shape[0]

    def bigAvxkp(vxkp):
        v = vxkp[:nv]
        xk = vxkp[nv:-np]
        p = vxkp[-np:]
        aonevxkp = A*v - B*np.dot(Bk.T, np.dot(Xck, xk)) + J.T*p
        atwovxkp = np.dot(Xok, np.dot(Ck.T, C*v)) + np.dot(Ak, xk) -\
            np.dot(Xok, np.dot(Ck.T, np.dot(Ck, xk))) -\
            np.dot(Bk, np.dot(Bk.T, np.dot(Xck, xk)))
        athrvxkp = J*v
        return np.vstack([aonevxkp, atwovxkp, athrvxkp])

    bigA = spsla.Linearoperator((nv+k+np, nv+k+np), matvec=bigAvxkp)

    # where to put the infdim Evals
    jshift = 1./infevlshift

    # asysmat = sps.vstack([sps.hstack([A, J.T]),
    #                       sps.hstack([J, sps.csc_matrix((NP, NP))])])
    # msysmat = sps.vstack([sps.hstack([M, sps.csc_matrix((NV, NP))]),
    #                       sps.csc_matrix((NP, NV+NP))])
    cscm = sps.csc_matrix
    shiftbigM = sps.vstack([sps.hstack([M, cscm((nv, k)), jshift*J.T]),
                            sps.hstack([cscm((k, nv)), Mk, cscm((k, np))]),
                            sps.hstack([jshift*J, cscm((np, k+np))])])

    return bigA, shiftbigM


def get_fwd_mats(linsysstr=None):
    try:
        linsysmats = scipy.io.loadmat(linsysstr)
    except IOError:
        raise IOError('Could not find the sysmatrices: ' + linsysstr)

    A = linsysmats['A']
    M = linsysmats['M']
    J = linsysmats['J']
    Brob = linsysmats['Brob']
    C = linsysmats['C']

    return M, A, J, Brob, C


def comp_evcs_evls(compall=False, nevs=None, linsysstr=None, bigamat=None,
                   bigmmatshift=None, bigmmat=None, retinstevcs=False):

    whichevecs = '_leftevs' if leftevecs else '_rightevs'
    bcstr = '_palpha{0}'.format(palpha) if bccontrol else ''

    levstr = linsysstr + bcstr + whichevecs

    try:
        if debug:
            raise IOError()
        evls = dou.load_npa(levstr)
        print 'loaded the eigenvalues of the linearized system: \n' + levstr
        if retinstevcs:
            evcs = dou.load_npa(levstr+'_instableevcs')
        return evls, evcs
    except IOError:
        print 'computing the eigenvalues of the linearized system: \n' + levstr
        if compall:
            A = bigamat.todense()
            M = bigmmat.todense() if not shiftall else bigmmatshift.todense()
            evls = spla.eigvals(A, M, overwrite_a=True, check_finite=False)
        else:
            if leftevecs:
                bigamat = bigamat.T
                # TODO: we assume that M is symmetric

            msplu = spsla.splu(bigmmatshift)

            def minvavec(vvec):
                return msplu.solve(bigamat*vvec)

            miaop = spsla.LinearOperator(bigamat.shape, matvec=minvavec,
                                         dtype=bigamat.dtype)

            shiftamm = bigamat - sigma*bigmmatshift
            sammsplu = spsla.splu(shiftamm)

            def samminvmvec(vvec):
                return sammsplu.solve(bigmmatshift*vvec)

            samminvop = spsla.LinearOperator(bigamat.shape, matvec=samminvmvec,
                                             dtype=bigamat.dtype)

            evls, evcs = spsla.eigs(miaop, k=nevs, sigma=sigma,
                                    OPinv=samminvop, maxiter=100,
                                    # sigma=None, which='SM',
                                    return_eigenvectors=retinstevcs)
        dou.save_npa(evls, levstr)
        if retinstevcs:
            instevlsindx = np.real(evls) > 0
            instevcs = evcs[:, instevlsindx]
            dou.save_npa(instevcs, levstr+'_instableevcs')
        else:
            instevcs = None
        return evls, instevcs

if __name__ == '__main__':
    thetakl = []
    NNl = []
    for cyldim in cyldiml:
        linsysstr = getthesysstr(cyldim=cyldim, bccontrol=bccontrol,
                                 palpha=palpha)
        matdict = getthemats(linsysstr=linsysstr)
        bigA = matdict['bigA']
        bigM = matdict['bigM']
        shiftbigM = matdict['shiftbigM']
        evls, evcs = comp_evcs_evls(compall=compall, nevs=nevs,
                                    linsysstr=linsysstr,
                                    bigamat=bigA, bigmmatshift=shiftbigM,
                                    bigmmat=bigM, retinstevcs=True)
        if plotall:
            plt.figure(1)
            plt.plot(np.real(evls), np.imag(evls), '+')
            plt.show(block=False)
        M = matdict['M']
        NN = M.shape[0]
        Brob = matdict['Brob']
        bobtlist = []
        for evc in evcs.T:
            evc = (evc[:NN].reshape((NN, 1)))
            thetak = 0
            for brobc in Brob.T:
                evbi = np.real(np.dot(evc.T, brobc))
                nev = np.real(np.sqrt(np.dot(evc.T.conj(), M*evc)))
                nbi = np.sqrt(np.dot(brobc.T, spsla.spsolve(M, brobc)))
                if verbose:
                    print 'Re((ev, Brob))/(|Re(ev)|*|bi|): ', evbi / (nev*nbi)
                thetak = thetak + np.abs(evbi) / (nev*nbi)
            bobtlist.append(thetak.flatten()[0])

        thetakl.append(min(bobtlist))
        NNl.append(NN)
        print 'N = {0}, thetak = {1}'.format(NN, thetak)
    from matlibplots import conv_plot_utils as cpu
    cpu.print_nparray_tex(np.array(thetakl), fstr='1.2e')
    cpu.print_nparray_tex(np.array(NNl), fstr='5d')
    if plotlast:
        plt.figure(2)
        plt.plot(np.real(evls), np.imag(evls), '*')
        plt.xlabel('$\\Re \\lambda$')
        plt.ylabel('$\\Im \\lambda$')
        plt.show(block=False)
        try:
            from matplotlib2tikz import save as tikz_save
            tikz_save(linsysstr + '.tikz',
                      figureheight='\\figureheight',
                      figurewidth='\\figurewidth')
        except ImportError:
            print 'to export to tikz consider installing `matplotlib2tikz`'
