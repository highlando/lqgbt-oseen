import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt

import sadptprj_riclyap_adi.lin_alg_utils as lau
import dolfin_navier_scipy.data_output_utils as dou


def compute_lrbt_transfos(zfc=None, zfo=None, mmat=None,
                          trunck=dict(threshh=1e-6)):
    """
    the transformation matrices for the BT MOR

    :param zfc:
        Factor of the controllability Gramian :math:`W_c = Z_cZ_c^H`
    :param zfo:
        Factor of the observability Gramian :math:`W_o = Z_oZ_o^H`
    :param mmat:
        mass matrix
    :param trunck:
        truncation parameters

    :return:
        the left and right transformation matrices `tl` and `tr` \
        for the balanced truncation

    """
    if mmat is None:
        mmat = sps.eye(zfo.shape[0])

    lsv_mat, sv, rsv_matt = np.linalg.svd(np.dot(zfc.T, mmat*zfo))

    k = np.where(sv > trunck['threshh'])[0].size
    lsvk, rsvk, svk = lsv_mat[:, :k], rsv_matt.T[:, :k], sv[:k]

    svsqri = 1./np.sqrt(svk)

    svsqri_mat = sps.dia_matrix((svsqri, np.array([0])), shape=(k, k))

    tl = np.dot(zfc, lsvk*svsqri_mat)
    tr = np.dot(zfo, rsvk*svsqri_mat)

    return tl, tr, sv


def compare_freqresp(mmat=None, amat=None, jmat=None, bmat=None,
                     cmat=None, tr=None, tl=None,
                     ahat=None, bhat=None, chat=None,
                     plot=False, datastr=None):
    """
    compare the frequency response of the original and the reduced model

    cf. [HeiSS08, p. 1059] but with B_2 = 0

    Returns
    -------
    freqrel : list of floats
        the frob norm of the transferfunction at a frequency range
    red_freqrel : list of floats
        from of the tf of the reduced model at the same frequencies
    absci : list of floats
        frequencies where the tfs are evaluated at
    """

    if ahat is None:
        ahat = np.dot(tl.T, amat*tr)
    if bhat is None:
        bhat = tl.T*bmat
    if chat is None:
        chat = cmat*tr

    NV, red_nv = amat.shape[0], ahat.shape[0]

    imunit = 1j

    absci = np.logspace(-4, 8, base=10)

    freqrel, red_freqrel, diff_freqrel = [], [], []

    for omega in absci:
        sadib = lau.solve_sadpnt_smw(amat=omega*imunit*mmat-amat,
                                     jmat=jmat, rhsv=bmat)
        freqrel.append(np.linalg.norm(cmat*sadib[:NV, :]))
        # print freqrel[-1]

        aib = np.linalg.solve(omega*imunit*np.eye(red_nv) - ahat, bhat)
        red_freqrel.append(np.linalg.norm(np.dot(chat, aib)))
        diff_freqrel.append(np.linalg.norm(np.dot(chat, aib)
                                           - cmat*sadib[:NV, :]))
        # print red_freqrel[-1]

    if datastr is not None:
        dou.save_output_json(dict(tmesh=absci.tolist(),
                                  fullsysfr=freqrel,
                                  redsysfr=red_freqrel,
                                  diffsysfr=diff_freqrel),
                             fstring=datastr + 'forfreqrespplot')

    if plot:
        legstr = ['NV was {0}'.format(mmat.shape[0]),
                  'nv is {0}'.format(tr.shape[1]),
                  'difference']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(absci, freqrel, color='b', linewidth=2.0)
        ax.plot(absci, red_freqrel, color='r', linewidth=2.0)
        ax.plot(absci, diff_freqrel, color='b', linewidth=1.0)
        plt.legend(legstr, loc=3)
        plt.semilogx()
        plt.semilogy()
        plt.show(block=False)

        from matplotlib2tikz import save as tikz_save
        tikz_save(datastr + 'freqresp.tikz',
                  figureheight='\\figureheight',
                  figurewidth='\\figurewidth'
                  )

    return freqrel, red_freqrel, absci


def compare_stepresp(tmesh=None, a_mat=None, c_mat=None, b_mat=None,
                     m_mat=None, tl=None, tr=None,
                     iniv=None,
                     fullresp=None, fsr_soldict=None,
                     plot=False, jsonstr=None):
    """ compare the system's step response to unit inputs in time domain

    with reduced system's response.

    We consider the system

    .. math::

        M\\dot v = Av + Bu, \\quad y = Cv

    on the discretized time interval :math:`(t_0, t_E)`

    and the reduced system with :math:`\\hat A = t_l^TAt_r`.

    Parameters
    ----------
    tmesh : iterable list or ndarray
        vector of the time instances
    a_mat : (N,N) sparse matrix
        system matrix
    c_mat : (ny,N) sparse matrix or ndarray
        output matrix
    b_mat : (N,nu) sparse matrix or ndarray
        input operator
    m_mat : (N,N) sparse matrix
        mass matrix
    tl, tr : (N,K) ndarrays
        left, right transformation for the reduced system
    iniv : (N,1) ndarray
        initial value and linearization point of the full system
    fullresp : callable f(v, **dict)
        returns the response of the full system
    fsr_soldict : dictionary
        parameters to be passed to `fullresp`
    plot : boolean, optional
        whether to plot, defaults to `False`
    jsonstr: string, optional
        if defined, the output is stored in this json file, defaults to `None`
    """

    from scipy.integrate import odeint

    ahat = np.dot(tl.T, a_mat*tr)
    chat = lau.mm_dnssps(c_mat, tr)

    inivhat = np.dot(tl.T, m_mat*iniv)

    inivout = lau.mm_dnssps(c_mat, iniv).tolist()

    red_stp_rsp, ful_stp_rsp = [], []
    for ccol in [0]:  # , b_mat.shape[1]-1]:  # range(2):  # b_mat.shape[1]):
        bmc = b_mat[:, ccol][:, :]
        red_bmc = tl.T * bmc

        def dtfunc(v, t):
            return (np.dot(ahat, v).flatten() + red_bmc.flatten())  # +\

        red_state = odeint(dtfunc, 0*inivhat.flatten(), tmesh)
        red_stp_rsp.append(np.dot(chat, red_state.T).T.tolist())
        ful_stp_rsp.append(fullresp(bcol=bmc, trange=tmesh, ini_vel=iniv,
                           cmat=c_mat, soldict=fsr_soldict))

    if jsonstr:
        try:
            tmesh = tmesh.tolist()
        except AttributeError:
            pass  # is a list already
        dou.save_output_json(datadict={"tmesh": tmesh,
                                       "ful_stp_rsp": ful_stp_rsp,
                                       "red_stp_rsp": red_stp_rsp,
                                       "inivout": inivout},
                             fstring=jsonstr,
                             module='sadptprj_riclyap_adi.bal_trunc_utils',
                             plotroutine='plot_step_resp')

    if plot:
        plot_step_resp(tmesh=tmesh, red_stp_rsp=red_stp_rsp,
                       ful_stp_rsp=ful_stp_rsp, inivout=inivout)


def plot_step_resp(str_to_json=None, tmesh=None,
                   red_stp_rsp=None, ful_stp_rsp=None, inivout=None,
                   compress=20):
    """
    compress : real, optional
        factor of compressing for plot, defaults to 20
    """
    from matplotlib2tikz import save as tikz_save

    if str_to_json is not None:
        jsdict = dou.load_json_dicts(str_to_json)
        tmesh = np.array(jsdict['tmesh'])
        red_stp_rsp = jsdict['red_stp_rsp']
        ful_stp_rsp = jsdict['ful_stp_rsp']
        inivout = jsdict['inivout']
    else:
        str_to_json = 'notspecified'

    redinds = list(range(0, len(tmesh), compress))
    redina = np.array(redinds)

    for ccol in range(len(red_stp_rsp)):
        # [0, b_mat.shape[1]-1]:  # range(2):  # b_mat.shape[1]):
        fuloutp = np.array(ful_stp_rsp[ccol])-np.array(inivout).T
        redoutp = np.array(red_stp_rsp[ccol])
        outdiff = fuloutp - redoutp
        NY = fuloutp.shape[1]/2
        fig = plt.figure(200 + ccol)

        ax1 = fig.add_subplot(131)
        ax1.plot(tmesh[redina], redoutp[redina, :NY], color='b', linewidth=2.0)
        ax1.plot(tmesh[redina], redoutp[redina, NY:], color='r', linewidth=2.0)

        ax2 = fig.add_subplot(132)
        ax2.plot(tmesh[redina], fuloutp[redina, :NY], color='b', linewidth=2.0)
        ax2.plot(tmesh[redina], fuloutp[redina, NY:], color='r', linewidth=2.0)

        ax3 = fig.add_subplot(133)
        ax3.plot(tmesh[redina], outdiff[redina, :NY], color='b', linewidth=2.0)
        ax3.plot(tmesh[redina], outdiff[redina, NY:], color='r', linewidth=2.0)

        tikz_save(str_to_json + '{0}'.format(200+ccol) + '.tikz',
                  figureheight='\\figureheight',
                  figurewidth='\\figurewidth'
                  )
        print('saved to ' + str_to_json + '{0}'.format(200+ccol) + '.tikz')
        fig.show()
