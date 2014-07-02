import dolfin
import os
import numpy as np

import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

import sadptprj_riclyap_adi.lin_alg_utils as lau
import sadptprj_riclyap_adi.proj_ric_utils as pru
import sadptprj_riclyap_adi.bal_trunc_utils as btu

import distr_control_fenics.cont_obs_utils as cou

import multiprocessing

dolfin.parameters.linear_algebra_backend = 'uBLAS'


def nwtn_adi_params():
    """
    Returns
    -------
    , : dictionary
        of the parameters for the Newton-ADI iteration
    """
    return dict(nwtn_adi_dict=dict(
                adi_max_steps=350,
                adi_newZ_reltol=1e-7,
                nwtn_max_steps=30,
                nwtn_upd_reltol=4e-8,
                nwtn_upd_abstol=1e-7,
                ms=[-30.0, -20.0, -10.0, -5.0, -3.0, -1.0],
                verbose=True,
                full_upd_norm_check=False,
                check_lyap_res=False))


def lqgbt(problemname='drivencavity',
          N=10, Re=1e2, plain_bt=False,
          use_ric_ini=None, t0=0.0, tE=1.0, Nts=11,
          NU=3, NY=3,
          paraoutput=True,
          trunc_lqgbtcv=1e-6,
          nwtn_adi_dict=None,
          comp_freqresp=False, comp_stepresp='nonlinear',
          closed_loop=False):
    """Main routine for LQGBT

    Parameters
    ----------
    problemname : string, optional
        what problem to be solved, 'cylinderwake' or 'drivencavity'
    N : int, optional
        parameter for the dimension of the space discretization
    Re : real, optional
        Reynolds number, defaults to `1e2`
    plain_bt : boolean, optional
        whether to try simple *balanced truncation*, defaults to False
    use_ric_ini : real, optional
        use the solution with this Re number as stabilizing initial guess,
        defaults to `None`
    t0, tE, Nts : real, real, int, optional
        starting and endpoint of the considered time interval, number of
        time instancses, default to `0.0, 1.0, 11`
    NU, NY : int, optional
        dimensions of components of in and output space (will double because
        there are two components), default to `3, 3`
    comp_freqresp : boolean, optional
        whether to compute and compare the frequency responses,
        defaults to `False`
    comp_stepresp : {'nonlinear', False, None}
        whether to compute and compare the step responses

        | if False -> no step response
        | if == 'nonlinear' -> compare linear reduced to nonlinear full model
        | else -> linear reduced versus linear full model

        defaults to `False`

    trunc_lqgbtcv : real, optional
        threshold at what the lqgbt characteristiv values are truncated,
        defaults to `1e-6`
    closed_loop : {'full_state_fb', 'red_output_fb', False, None}
        how to do the closed loop simulation:

        | if False -> no simulation
        | if == 'full_state_fb' -> full state feedback
        | if == 'red_output_fb' -> reduced output feedback
        | else -> no control is applied

        defaults to `False`

    """

    problemdict = dict(drivencavity=dnsps.drivcav_fems,
                       cylinderwake=dnsps.cyl_fems)

    typprb = 'BT' if plain_bt else 'LQG-BT'

    print '\n ### We solve the {0} problem for the {1} at Re={2} ###\n'.\
        format(typprb, problemname, Re)

    problemfem = problemdict[problemname]
    femp = problemfem(N)

    nu = femp['charlen']/Re
    # specify in what spatial direction Bu changes. The remaining is constant
    uspacedep = femp['uspacedep']

    if nwtn_adi_dict is not None:
        nap = nwtn_adi_dict
    else:
        nap = nwtn_adi_params()['nwtn_adi_dict']
    # output
    ddir = 'data/'
    try:
        os.chdir(ddir)
    except OSError:
        raise Warning('need "' + ddir + '" subdir for storing the data')
    os.chdir('..')

    stokesmats = dts.get_stokessysmats(femp['V'], femp['Q'], nu)

    rhsd_vf = dts.setget_rhs(femp['V'], femp['Q'],
                             femp['fv'], femp['fp'], t=0)

    # remove the freedom in the pressure
    stokesmats['J'] = stokesmats['J'][:-1, :][:, :]
    stokesmats['JT'] = stokesmats['JT'][:, :-1][:, :]
    rhsd_vf['fp'] = rhsd_vf['fp'][:-1, :]

    # reduce the matrices by resolving the BCs
    (stokesmatsc,
     rhsd_stbc,
     invinds,
     bcinds,
     bcvals) = dts.condense_sysmatsbybcs(stokesmats,
                                         femp['diribcs'])

    # pressure freedom and dirichlet reduced rhs
    rhsd_vfrc = dict(fpr=rhsd_vf['fp'], fvc=rhsd_vf['fv'][invinds, ])

    # add the info on boundary and inner nodes
    bcdata = {'bcinds': bcinds,
              'bcvals': bcvals,
              'invinds': invinds}
    femp.update(bcdata)

    # casting some parameters
    NV = len(femp['invinds'])

    prbstr = '_bt' if plain_bt else '_lqgbt'
    # contsetupstr = 'NV{0}NU{1}NY{2}alphau{3}'.format(NV, NU, NY, alphau)
    contsetupstr = 'NV{0}NU{1}NY{2}'.format(NV, NU, NY)

    def get_fdstr(Re):
        return ddir + problemname + '_Re{0}_'.format(Re) + \
            contsetupstr + prbstr

    fdstr = get_fdstr(Re)

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(rhsd_vfrc)  # adding fvc, fpr
    soldict.update(fv_stbc=rhsd_stbc['fv'], fp_stbc=rhsd_stbc['fp'],
                   N=N, nu=nu, data_prfx=fdstr)

#
# Prepare for control
#

    # get the control and observation operators
    try:
        b_mat = dou.load_spa(ddir + contsetupstr + '__b_mat')
        u_masmat = dou.load_spa(ddir + contsetupstr + '__u_masmat')
        print 'loaded `b_mat`'
    except IOError:
        print 'computing `b_mat`...'
        b_mat, u_masmat = cou.get_inp_opa(cdcoo=femp['cdcoo'], V=femp['V'],
                                          NU=NU, xcomp=uspacedep)
        dou.save_spa(b_mat, ddir + contsetupstr + '__b_mat')
        dou.save_spa(u_masmat, ddir + contsetupstr + '__u_masmat')
    try:
        mc_mat = dou.load_spa(ddir + contsetupstr + '__mc_mat')
        y_masmat = dou.load_spa(ddir + contsetupstr + '__y_masmat')
        print 'loaded `c_mat`'
    except IOError:
        print 'computing `c_mat`...'
        mc_mat, y_masmat = cou.get_mout_opa(odcoo=femp['odcoo'],
                                            V=femp['V'], NY=NY)
        dou.save_spa(mc_mat, ddir + contsetupstr + '__mc_mat')
        dou.save_spa(y_masmat, ddir + contsetupstr + '__y_masmat')

    # restrict the operators to the inner nodes
    mc_mat = mc_mat[:, invinds][:, :]
    b_mat = b_mat[invinds, :][:, :]

    # tb_mat = 1./np.sqrt(alphau)

    c_mat = lau.apply_massinv(y_masmat, mc_mat, output='sparse')
    c_mat_reg = lau.app_prj_via_sadpnt(amat=stokesmatsc['M'],
                                       jmat=stokesmatsc['J'],
                                       rhsv=c_mat.T,
                                       transposedprj=True).T
    # c_mat_reg = np.array(c_mat.todense())

    # TODO: right choice of norms for y
    #       and necessity of regularization here
    #       by now, we go on number save
#
# setup the system for the correction
#
#
# compute the uncontrolled steady state Stokes solution
#
    v_ss_nse, list_norm_nwtnupd = snu.solve_steadystate_nse(**soldict)
    (convc_mat, rhs_con,
     rhsv_conbc) = snu.get_v_conv_conts(prev_v=v_ss_nse, invinds=invinds,
                                        V=femp['V'], diribcs=femp['diribcs'])

    f_mat = - stokesmatsc['A'] - convc_mat
    mmat = stokesmatsc['M']

    # ssv_rhs = rhsv_conbc + rhsv_conbc + rhsd_vfrc['fvc'] + rhsd_stbc['fv']

    if plain_bt:
        get_gramians = pru.solve_proj_lyap_stein
    else:
        get_gramians = pru.proj_alg_ric_newtonadi

    truncstr = '__lqgbtcv{0}'.format(trunc_lqgbtcv)
    try:
        tl = dou.load_npa(fdstr + '__tl' + truncstr)
        tr = dou.load_npa(fdstr + '__tr' + truncstr)
        print 'loaded the left and right transformations: \n' + \
            fdstr + '__tl/__tr' + truncstr
    except IOError:
        print 'computing the left and right transformations' + \
            ' and saving to: \n' + fdstr + '__tl/__tr' + truncstr

        try:
            zwc = dou.load_npa(fdstr + '__zwc')
            zwo = dou.load_npa(fdstr + '__zwo')
            print 'loaded factor of the Gramians: \n\t' + \
                fdstr + '__zwc/__zwo'
        except IOError:
            zinic, zinio = None, None
            if use_ric_ini is not None:
                fdstr = get_fdstr(use_ric_ini)
                try:
                    zinic = dou.load_npa(fdstr + '__zwc')
                    zinio = dou.load_npa(fdstr + '__zwo')
                    print 'Initialize Newton ADI by zwc/zwo from ' + fdstr
                except IOError:
                    raise UserWarning('No initial guess with Re={0}'.
                                      format(use_ric_ini))

            fdstr = get_fdstr(Re)
            print 'computing factors of Gramians: \n\t' + \
                fdstr + '__zwc/__zwo'

            def compobsg():
                zwo = get_gramians(mmat=mmat.T, amat=f_mat.T,
                                   jmat=stokesmatsc['J'],
                                   bmat=c_mat_reg.T, wmat=b_mat,
                                   nwtn_adi_dict=nap,
                                   z0=zinio)['zfac']
                dou.save_npa(zwo, fdstr + '__zwo')

            def compcong():
                zwc = get_gramians(mmat=mmat, amat=f_mat,
                                   jmat=stokesmatsc['J'],
                                   bmat=b_mat, wmat=c_mat_reg.T,
                                   nwtn_adi_dict=nap,
                                   z0=zinic)['zfac']
                dou.save_npa(zwc, fdstr + '__zwc')

            print '\n ### multithread start - output might be intermangled'

            p1 = multiprocessing.Process(target=compobsg)
            p2 = multiprocessing.Process(target=compcong)
            p1.start()
            p2.start()
            p1.join()
            p2.join()

            print '### multithread end'

        print 'computing the left and right transformations' + \
            ' and saving to:\n' + fdstr + '__tr/__tl' + truncstr
        tl, tr = btu.compute_lrbt_transfos(zfc=zwc, zfo=zwo,
                                           mmat=stokesmatsc['M'],
                                           trunck={'threshh': trunc_lqgbtcv}
                                           )
        dou.save_npa(tl, fdstr + '__tl' + truncstr)
        dou.save_npa(tr, fdstr + '__tr' + truncstr)

    if comp_freqresp:
        btu.compare_freqresp(mmat=stokesmatsc['M'], amat=f_mat,
                             jmat=stokesmatsc['J'], bmat=b_mat,
                             cmat=c_mat, tr=tr, tl=tl,
                             plot=True, datastr=fdstr + '__tl' + truncstr)

    if comp_stepresp is not False:
        if comp_stepresp == 'nonlinear':
            stp_rsp_nwtn = 3
            stp_rsp_dtpr = 'nonl_stepresp_'
        else:
            stp_rsp_nwtn = 1
            stp_rsp_dtpr = 'stepresp_'

        def fullstepresp_lnse(bcol=None, trange=None, ini_vel=None,
                              cmat=None, soldict=None):
            soldict.update(fv_stbc=rhsd_stbc['fv']+bcol,
                           vel_nwtn_stps=stp_rsp_nwtn, trange=trange,
                           iniv=ini_vel, lin_vel_point=ini_vel,
                           clearprvdata=True, data_prfx=stp_rsp_dtpr,
                           return_dictofvelstrs=True)

            dictofvelstrs = snu.solve_nse(**soldict)

            return cou.extract_output(strdict=dictofvelstrs, tmesh=trange,
                                      c_mat=cmat, load_data=dou.load_npa)

    # differences in the initial vector
    # print np.dot(c_mat_reg, v_ss_nse)
    # print np.dot(np.dot(c_mat_reg, tr),
    #              np.dot(tl.T, stokesmatsc['M']*v_ss_nse))

        jsonstr = fdstr + stp_rsp_dtpr + '_Nred{0}_t0tENts{1}{2}{3}.json'.\
            format(tl.shape[1], t0, tE, Nts)
        btu.compare_stepresp(tmesh=np.linspace(t0, tE, Nts),
                             a_mat=f_mat, c_mat=c_mat_reg, b_mat=b_mat,
                             m_mat=stokesmatsc['M'],
                             tr=tr, tl=tl, iniv=v_ss_nse,
                             # ss_rhs=ssv_rhs,
                             fullresp=fullstepresp_lnse, fsr_soldict=soldict,
                             plot=True, jsonstr=jsonstr)

# compute the regulated system
    zwc = dou.load_npa(fdstr + '__zwc')
    zwo = dou.load_npa(fdstr + '__zwo')

    trange = np.linspace(t0, tE, Nts)

    print 'NV = {0}, NP = {2}, k = {1}'.\
        format(tl.shape[0], tl.shape[1], stokesmatsc['J'].shape[0])

    if closed_loop is False:
        return

    elif closed_loop == 'full_state_fb':

        mtxtb = pru.get_mTzzTtb(stokesmatsc['M'].T, zwc, b_mat)

        def fv_tmdp_fullstatefb(time=None, curvel=None,
                                linv=None, tb_mat=None, tbxm_mat=None, **kw):
            """realizes a full state static feedback as a function

            that can be passed to a solution routine for the
            unsteady Navier-Stokes equations

            Parameters
            ----------
            time : real
                current time
            curvel : (N,1) nparray
                current velocity
            linv : (N,1) nparray
                linearization point for the linear model
            tb_mat : (N,K) nparray
                input matrix containing the input weighting
            tbxm_mat : (N,K) nparray
                `tb_mat * gain * mass`

            Returns
            -------
            actua : (N,1) nparray
                current contribution to the right-hand side
            , : dictionary
                dummy `{}` for consistency
            """

            actua = -lau.comp_uvz_spdns(tb_mat, tbxm_mat, curvel-linv)
            # actua = 0*curvel
            print '\nnorm of deviation', np.linalg.norm(curvel-linv)
            # print 'norm of actuation {0}'.format(np.linalg.norm(actua))
            return actua, {}

        tmdp_fsfb_dict = dict(linv=v_ss_nse, tb_mat=b_mat, tbxm_mat=mtxtb.T)

        fv_tmdp = fv_tmdp_fullstatefb
        fv_tmdp_params = tmdp_fsfb_dict
        fv_tmdp_memory = None

    elif closed_loop == 'red_output_fb':
        ak_mat = np.dot(tl.T, f_mat*tr)
        ck_mat = lau.matvec_densesparse(c_mat_reg, tr)
        bk_mat = tl.T*b_mat

        tltm, trtm = tl.T*stokesmatsc['M'], tr.T*stokesmatsc['M']
        xok = np.dot(np.dot(tltm, zwo), np.dot(zwo.T, tltm.T))
        xck = np.dot(np.dot(trtm, zwc), np.dot(zwc.T, trtm.T))
        # sysmatk = (ak_mat - np.dot(np.dot(xok, ck_mat.T), ck_mat) -
        #                             np.dot(bk_mat, np.dot(bk_mat.T, xck)))

        obs_bk = np.dot(xok, ck_mat.T)

        DT = (tE - t0)/(Nts-1)

        sysmatk_inv = np.linalg.inv(np.eye(tl.shape[1]) - DT*(ak_mat -
                                    np.dot(np.dot(xok, ck_mat.T), ck_mat) -
                                    np.dot(bk_mat, np.dot(bk_mat.T, xck))))

        # raise Warning('TODO: debug')

        def fv_tmdp_redoutpfb(time=None, curvel=None, memory=None,
                              linvel=None,
                              ipsysk_mat_inv=None,
                              obs_bk=None, cts=None,
                              b_mat=None, c_mat=None,
                              xck=None, bk_mat=None,
                              **kw):
            """realizes a reduced static output feedback as a function

            that can be passed to a solution routine for the
            unsteady Navier-Stokes equations

            For convinience the
            Parameters
            ----------
            time : real
                current time
            curvel : (N,1) nparray
                current velocity. For consistency, the full state is taken
                as input. However, internally, we only use the observation
                `y = c_mat*curvel`
            memory : dictionary
                contains values from previous call, in particular the
                previous state estimate
            linvel : (N,1) nparray
                linearization point for the linear model
            ipsysk_mat_inv : (K,K) nparray
                inverse of the system matrix that defines the update
                of the state estimate
            obs_bk : (K,NU) nparray
                input matrix in the observer
            cts : real
                time step length
            b_mat : (N,NU) sparse matrix
                input matrix of the full system
                c_mat=None,
            c_mat : (NY,N) sparse matrix
                output matrix of the full system
            xck : (K,K) nparray
                reduced solution of the CARE
            bk_mat : (K,NU) nparray
                reduced input matrix

            Returns
            -------
            actua : (N,1) nparray
                the current actuation
            memory : dictionary
                to be passed back in the next timestep

            """
            xk_old = memory['xk_old']
            buk = cts*np.dot(obs_bk,
                             lau.matvec_densesparse(c_mat, (curvel-linvel)))
            xk_old = np.dot(ipsysk_mat_inv, xk_old + buk)
            #         cts*np.dot(obs_bk,
            #                 lau.matvec_densesparse(c_mat, (curvel-linvel))))
            memory['xk_old'] = xk_old
            actua = -b_mat*np.dot(bk_mat.T, np.dot(xck, xk_old))
            print '\nnorm of deviation', np.linalg.norm(curvel-linvel)
            # print 'norm of actuation {0}'.format(np.linalg.norm(actua))
            return actua, memory

        fv_rofb_dict = dict(cts=DT, linvel=v_ss_nse, b_mat=b_mat,
                            c_mat=c_mat_reg, obs_bk=obs_bk, bk_mat=bk_mat,
                            ipsysk_mat_inv=sysmatk_inv, xck=xck)

        fv_tmdp = fv_tmdp_redoutpfb
        fv_tmdp_params = fv_rofb_dict
        fv_tmdp_memory = dict(xk_old=np.zeros((tl.shape[1], 1)))

    else:
        fv_tmdp = None
        fv_tmdp_params = {}
        fv_tmdp_memory = {}

    perturbini = 1e-3*np.ones((NV, 1))
    reg_pertubini = lau.app_prj_via_sadpnt(amat=stokesmatsc['M'],
                                           jmat=stokesmatsc['J'],
                                           rhsv=perturbini)

    soldict.update(fv_stbc=rhsd_stbc['fv'],
                   trange=trange,
                   iniv=v_ss_nse + reg_pertubini,
                   lin_vel_point=v_ss_nse,
                   clearprvdata=True, data_prfx=fdstr + truncstr,
                   fv_tmdp=fv_tmdp,
                   vel_nwtn_stps=1,
                   comp_nonl_semexp=True,
                   fv_tmdp_params=fv_tmdp_params,
                   fv_tmdp_memory=fv_tmdp_memory,
                   return_dictofvelstrs=True)

    outstr = truncstr + '{0}'.format(closed_loop) \
        + 't0{0}tE{1}Nts{2}'.format(t0, tE, Nts)
    if paraoutput:
        soldict.update(paraviewoutput=True,
                       vfileprfx='results/vel_'+outstr,
                       pfileprfx='results/p_'+outstr)

    dictofvelstrs = snu.solve_nse(**soldict)

    yscomplist = cou.extract_output(strdict=dictofvelstrs, tmesh=trange,
                                    c_mat=c_mat, load_data=dou.load_npa)

    dou.save_output_json(dict(tmesh=trange.tolist(), outsig=yscomplist),
                         fstring=fdstr + truncstr + '{0}'.format(closed_loop) +
                         't0{0}tE{1}Nts{2}'.format(t0, tE, Nts))

    dou.plot_outp_sig(tmesh=trange, outsig=yscomplist)
    # import matplotlib.pyplot as plt
    # plt.plot(trange, yscomplist)
    # plt.show(block=False)

if __name__ == '__main__':
    # lqgbt(N=10, Re=500, use_ric_ini=None, plain_bt=False)
    lqgbt(problemname='cylinderwake', N=3,  # use_ric_ini=2e2,
          Re=1.0e2, plain_bt=False,
          t0=0.0, tE=2.0, Nts=1e3+1,
          comp_freqresp=False, comp_stepresp=False)
