import os
import numpy as np

# import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

import sadptprj_riclyap_adi.lin_alg_utils as lau
import sadptprj_riclyap_adi.proj_ric_utils as pru
import sadptprj_riclyap_adi.bal_trunc_utils as btu

import distr_control_fenics.cont_obs_utils as cou

import nse_riccont_utils as nru
import nse_extlin_utils as neu

debug = False

checktheres = True  # whether to check the Riccati Residuals
checktheres = False

switchonsfb = 0  # 1.5

# TODO: clear distinction of target state, linearization point, initial value
# TODO: maybe redefine: by now we need to use -fmat all the time (but +ak_mat)


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
          gamma=1.,
          use_ric_ini=None, t0=0.0, tE=1.0, Nts=11,
          NU=3, NY=3,
          bccontrol=True, palpha=1e-5,
          npcrdstps=8,
          pymess=False,
          paraoutput=True,
          plotit=True,
          trunc_lqgbtcv=1e-6,
          nwtn_adi_dict=None,
          pymess_dict=None,
          whichinival='sstate',
          tpp=5.,  # time to add on Stokes inival for `sstokes++`
          comp_freqresp=False, comp_stepresp='nonlinear',
          closed_loop=False, multiproc=False,
          perturbpara=1e-3,
          trytofail=False, ttf_npcrdstps=3,
          robit=False, robmrgnfac=0.5):
    """Main routine for LQGBT

    Parameters
    ----------
    problemname : string, optional
        what problem to be solved, 'cylinderwake' or 'drivencavity'
    N : int, optional
        parameter for the dimension of the space discretization
    Re : real, optional
        Reynolds number, defaults to `1e2`
    gamma : real, optional
        regularization parameter, puts weight on `|u|` in the underlying
        LQR cost functional that, defaults to `1.`
    plain_bt : boolean, optional
        whether to try simple *balanced truncation*, defaults to False
    use_ric_ini : real, optional
        use the solution with this Re number as stabilizing initial guess,
        defaults to `None`
    t0, tE, Nts : real, real, int, optional
        starting and endpoint of the considered time interval, number of
        time instancses, default to `0.0, 1.0, 11`
    bccontrol : boolean, optional
        whether to apply boundary control via penalized robin conditions,
        defaults to `False`
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

    typprb = 'BT' if plain_bt else 'LQG-BT'

    print('\n ### We solve the {0} problem for the {1} at Re={2} ###\n'.
          format(typprb, problemname, Re))
    print(' ### The control is weighted with Gamma={0}'.format(gamma))

    # output
    ddir = 'data/'
    try:
        os.chdir(ddir)
    except OSError:
        raise Warning('need "' + ddir + '" subdir for storing the data')
    os.chdir('..')

    femp, stokesmatsc, rhsd_vfrc, rhsd_stbc \
        = dnsps.get_sysmats(problem=problemname, N=N, Re=Re,
                            bccontrol=bccontrol, scheme='TH')

    # casting some parameters
    invinds, NV = femp['invinds'], len(femp['invinds'])

#
# Prepare for control
#
    prbstr = '_bt' if plain_bt else '_lqgbt'
    if pymess:
        prbstr = prbstr + '__pymess'
    # contsetupstr = 'NV{0}NU{1}NY{2}alphau{3}'.format(NV, NU, NY, alphau)
    if bccontrol:
        import scipy.sparse as sps
        contsetupstr = 'NV{0}_bcc_NY{1}_palpha{2}'.format(NV, NY, palpha)
        shortcontsetupstr = '{0}{1}{2}'.format(NV, NY, np.int(np.log2(palpha)))
        stokesmatsc['A'] = stokesmatsc['A'] + 1./palpha*stokesmatsc['Arob']
        b_mat = 1./palpha*stokesmatsc['Brob']
        u_masmat = sps.eye(b_mat.shape[1], format='csr')
        print(' ### Robin-type boundary control palpha={0}'.format(palpha))
    else:
        contsetupstr = 'NV{0}NU{1}NY{2}'.format(NV, NU, NY)
        shortcontsetupstr = '{0}{1}{2}'.format(NV, NU, NY)

    inivstr = '_' + whichinival if not whichinival == 'sstokes++' \
        else '_sstokes++{0}'.format(tpp)

    def get_fdstr(Re, short=False):
        if short:
            return ddir + 'cw' + '{0}{1}_'.format(Re, gamma) + \
                shortcontsetupstr
        return ddir + problemname + '_Re{0}_gamma{1}_'.format(Re, gamma) + \
            contsetupstr + prbstr

    fdstr = get_fdstr(Re)
    fdstrini = get_fdstr(use_ric_ini) if use_ric_ini is not None else None

#
# Prepare for control
#

    # get the control and observation operators
    if not bccontrol:
        try:
            b_mat = dou.load_spa(ddir + contsetupstr + '__b_mat')
            u_masmat = dou.load_spa(ddir + contsetupstr + '__u_masmat')
            print('loaded `b_mat`')
        except IOError:
            print('computing `b_mat`...')
            b_mat, u_masmat = cou.get_inp_opa(cdcoo=femp['cdcoo'], V=femp['V'],
                                              NU=NU, xcomp=femp['uspacedep'])
            dou.save_spa(b_mat, ddir + contsetupstr + '__b_mat')
            dou.save_spa(u_masmat, ddir + contsetupstr + '__u_masmat')

        b_mat = b_mat[invinds, :][:, :]
        # tb_mat = 1./np.sqrt(alphau)

    b_mat_reg = lau.app_prj_via_sadpnt(amat=stokesmatsc['M'],
                                       jmat=stokesmatsc['J'],
                                       rhsv=b_mat,
                                       transposedprj=True)

    try:
        mc_mat = dou.load_spa(ddir + contsetupstr + '__mc_mat')
        y_masmat = dou.load_spa(ddir + contsetupstr + '__y_masmat')
        print(('loaded `c_mat` from' + ddir + contsetupstr + '...'))
    except IOError:
        print('computing `c_mat`...')
        mc_mat, y_masmat = cou.get_mout_opa(odcoo=femp['odcoo'],
                                            V=femp['V'], NY=NY)
        dou.save_spa(mc_mat, ddir + contsetupstr + '__mc_mat')
        dou.save_spa(y_masmat, ddir + contsetupstr + '__y_masmat')

    c_mat = lau.apply_massinv(y_masmat, mc_mat, output='sparse')
    # restrict the operators to the inner nodes

    mc_mat = mc_mat[:, invinds][:, :]
    c_mat = c_mat[:, invinds][:, :]
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

    # compute the uncontrolled steady state NSE solution for the linearization
    soldict = {}
    soldict.update(stokesmatsc)  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    # soldict.update(rhsd_vfrc)  # adding fvc, fpr
    veldatastr = ddir + problemname + '_Re{0}'.format(Re)
    if bccontrol:
        veldatastr = veldatastr + '__bcc_palpha{0}'.format(palpha)

    nu = femp['charlen']/Re
    soldict.update(fv=rhsd_stbc['fv']+rhsd_vfrc['fvc'],
                   fp=rhsd_stbc['fp']+rhsd_vfrc['fpr'],
                   N=N, nu=nu, data_prfx=veldatastr)

    v_ss_nse, list_norm_nwtnupd = snu.\
        solve_steadystate_nse(vel_pcrd_stps=npcrdstps,
                              clearprvdata=debug, **soldict)

    (convc_mat, rhs_con,
     rhsv_conbc) = snu.get_v_conv_conts(prev_v=v_ss_nse, invinds=invinds,
                                        V=femp['V'], diribcs=femp['diribcs'])

    f_mat = - stokesmatsc['A'] - convc_mat
    # the robin term `arob` has been added before
    mmat = stokesmatsc['M']
    amat = stokesmatsc['A']
    jmat = stokesmatsc['J']

    # MAF -- need to change the convc_mat, i.e. we need another v_ss_nse
    # MAF -- need to change the f_mat, i.e. we need another convc_mat
    if trytofail:
        v_ss_nse_MAF, _ = snu.\
            solve_steadystate_nse(vel_pcrd_stps=ttf_npcrdstps, vel_nwtn_stps=0,
                                  vel_pcrd_tol=1e-15,
                                  clearprvdata=True, **soldict)
        diffv = v_ss_nse - v_ss_nse_MAF
        convc_mat_MAF, _, _ = \
            snu.get_v_conv_conts(prev_v=v_ss_nse_MAF, invinds=invinds,
                                 V=femp['V'], diribcs=femp['diribcs'])
        relnormdiffv = np.sqrt(np.dot(diffv.T, mmat*diffv) /
                               np.dot(v_ss_nse.T, mmat*v_ss_nse))
        print('relative difference to linearization: {0}'.
              format(relnormdiffv))
        f_mat_gramians = - stokesmatsc['A'] - convc_mat_MAF
        fdstr = fdstr + '_MAF_ttfnpcrds{0}'.format(ttf_npcrdstps)
    else:
        f_mat_gramians = f_mat

#
# ### Compute or get the Gramians
#
    Rmo, Rmhalf = 1./gamma, 1./np.sqrt(gamma)

    if closed_loop == 'red_output_fb' or closed_loop == 'red_sdre_fb':
        truncstr = '__lqgbtcv{0}'.format(trunc_lqgbtcv)
        shorttruncstr = '{0}'.format(trunc_lqgbtcv)
    else:
        truncstr = '_'
        shorttruncstr = '_'

    cmpricfacpars = dict(multiproc=multiproc, nwtn_adi_dict=nwtn_adi_dict,
                         ric_ini_str=fdstrini)
    cmprlprjpars = dict(trunc_lqgbtcv=trunc_lqgbtcv)

    if plain_bt:
        print('`plain_bt` -- this is not maintained anymore -- good luck')
        # get_gramians = pru.solve_proj_lyap_stein
    # else:
    #     get_gramians = pru.proj_alg_ric_newtonadi

    if comp_freqresp:
        tl, tr = nru.get_rl_projections(fdstr=fdstr, truncstr=truncstr,
                                        fmat=f_mat_gramians, mmat=mmat,
                                        jmat=jmat,
                                        bmat=b_mat_reg, cmat=c_mat_reg,
                                        Rmhalf=Rmhalf,
                                        cmpricfacpars=cmpricfacpars,
                                        trunc_lqgbtcv=trunc_lqgbtcv)
        btu.compare_freqresp(mmat=stokesmatsc['M'], amat=f_mat,
                             jmat=stokesmatsc['J'], bmat=b_mat,
                             cmat=c_mat, tr=tr, tl=tl,
                             plot=True, datastr=fdstr + truncstr + '__tl')

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
    trange = np.linspace(t0, tE, Nts)
    DT = (tE - t0)/(Nts-1)

    if closed_loop is False:
        return

    elif closed_loop == 'full_state_fb':
        shortclstr = 'fsfb'
        zwc = nru.get_ric_facs(fdstr=fdstr,
                               fmat=f_mat_gramians, mmat=mmat,
                               jmat=jmat, bmat=b_mat_reg, cmat=c_mat_reg,
                               ric_ini_str=fdstrini, Rmhalf=Rmhalf,
                               nwtn_adi_dict=nwtn_adi_dict, zwconly=True,
                               multiproc=multiproc, pymess=False,
                               checktheres=False)

        mtxb = pru.get_mTzzTtb(stokesmatsc['M'].T, zwc, b_mat_reg)

        dimu = b_mat.shape[1]
        zerou = np.zeros((dimu, 1))
        if switchonsfb > 0:
            print('the feedback will switched on at ' +
                  't={0:.4f}'.format(switchonsfb))

        def fv_tmdp_fullstatefb(time=None, curvel=None,
                                linv=None, tb_mat=None, btxm_mat=None, **kw):
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
            btxm_mat : (N,K) nparray
                `b_mat.T * gain * mass`

            Returns
            -------
            actua : (N,1) nparray
                current contribution to the right-hand side
            , : dictionary
                dummy `{}` for consistency
            """

            if time < switchonsfb:
                return tb_mat.dot(zerou), {}
            else:
                actua = -lau.comp_uvz_spdns(tb_mat, btxm_mat, curvel-linv)
                return actua, {}

        tmdp_fsfb_dict = dict(linv=v_ss_nse, tb_mat=b_mat*Rmo,
                              btxm_mat=mtxb.T)

        fv_tmdp = fv_tmdp_fullstatefb
        fv_tmdp_params = tmdp_fsfb_dict
        fv_tmdp_memory = None

    elif closed_loop == 'red_output_fb':
        shortclstr = 'rofb'
        DT = (tE - t0)/(Nts-1)

        ak_mat, bk_mat, ck_mat, xok, xck, tl, tr = \
            nru.get_prj_model(truncstr=truncstr, fdstr=fdstr,
                              abconly=False,
                              mmat=mmat, fmat=f_mat_gramians, jmat=jmat,
                              bmat=b_mat_reg, cmat=c_mat_reg,
                              cmpricfacpars=cmpricfacpars,
                              Rmhalf=Rmhalf,
                              cmprlprjpars=cmprlprjpars)

        obs_bk = np.dot(xok, ck_mat.T)

        sysmatk_inv = np.linalg.inv(np.eye(ak_mat.shape[1]) - DT*(ak_mat -
                                    np.dot(np.dot(xok, ck_mat.T), ck_mat) -
                                    np.dot(bk_mat, np.dot(bk_mat.T, xck))))

        def fv_tmdp_redoutpfb(time=None, curvel=None, memory=None,
                              linvel=None,
                              ipsysk_mat_inv=None,
                              obs_bk=None, cts=None,
                              b_mat=None, c_mat=None, Rmo=None,
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
            obs_ck : (NY,K) nparray
                output matrix in the observer
            cts : real
                time step length
            b_mat : (N,NU) sparse matrix
                input matrix of the full system
                c_mat=None,
            c_mat : (NY,N) sparse matrix
                output matrix of the full system
            Rmo : float
                inverse of the input weighting scalar
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
                             lau.mm_dnssps(c_mat, (curvel-linvel)))
            xk_old = np.dot(ipsysk_mat_inv, xk_old + buk)
            memory['xk_old'] = xk_old
            actua = -lau.mm_dnssps(b_mat*Rmo,
                                   np.dot(bk_mat.T, np.dot(xck, xk_old)))
            if np.mod(np.int(time/DT), np.int(tE/DT)/100) == 0:
                print(('time now: {0}, end time: {1}'.format(time, tE)))
                print('\nnorm of deviation', np.linalg.norm(curvel-linvel))
                print('norm of actuation {0}'.format(np.linalg.norm(actua)))
            return actua, memory

        fv_rofb_dict = dict(cts=DT, linvel=v_ss_nse, b_mat=b_mat, Rmo=Rmo,
                            c_mat=c_mat_reg, obs_bk=obs_bk, bk_mat=bk_mat,
                            ipsysk_mat_inv=sysmatk_inv, xck=xck)

        fv_tmdp = fv_tmdp_redoutpfb
        fv_tmdp_params = fv_rofb_dict
        fv_tmdp_memory = dict(xk_old=np.zeros((tl.shape[1], 1)))

        soldict.update(dict(verbose=False))

    elif closed_loop == 'red_sdre_fb':
        shortclstr = 'rdsdrfb'

        sdcpicard = 1.
        vinf = v_ss_nse
        get_cur_sdccoeff = neu.get_get_cur_extlin(vinf=vinf, amat=amat,
                                                  picrdvsnwtn=sdcpicard,
                                                  **femp)
        sdre_ric_ini = fdstr
        cmpricfacpars.update(ric_ini_str=sdre_ric_ini)

        def solve_sdre(curfmat, memory=None, eps=None, time=None):
            if memory['basePk'] is None:  # initialization
                sdrefdstr = fdstr + inivstr + '_SDREini'
            else:
                sdrefdstr = (fdstr + inivstr +
                             '_SDREeps{0}t{1:.2e}'.format(eps, time))
            ak_mat, cur_bk, _, _, cxck, cur_tl, cur_tr = \
                nru.get_prj_model(truncstr=truncstr, fdstr=sdrefdstr,
                                  mmat=mmat, fmat=-curfmat, jmat=jmat,
                                  bmat=b_mat_reg, cmat=c_mat_reg,
                                  Rmhalf=Rmhalf,
                                  cmpricfacpars=cmpricfacpars,
                                  cmprlprjpars=cmprlprjpars)
            cmpricfacpars.update(ric_ini_str=sdrefdstr)
            baseGain = cur_bk.T.dot(cxck)
            memory.update(baseAk=ak_mat)
            memory.update(basePk=cxck, baseGain=baseGain)
            memory.update(baseZk=ak_mat - cur_bk.dot(baseGain))
            memory.update(cur_bk=cur_bk)
            memory.update(cur_tl=cur_tl, cur_tr=cur_tr)
            return

        def sdre_feedback(curvel=None, memory=None,
                          updtthrsh=None, time=None, **kw):
            ''' function for the SDRE feedback

            Parameters
            ---
            use_ric_ini : string, optional
                path to a stabilizing initial guess
            '''

            norm = np.linalg.norm
            print('time: {0}, |v|: {1}'.format(time, norm(curvel)))
            print('time: {0}, |vinf|: {1}'.format(time, norm(vinf)))
            curfmat = get_cur_sdccoeff(vcur=curvel)

            if memory['basePk'] is None:  # initialization
                savethev = np.copy(curvel)
                memory.update(basev=savethev)
                solve_sdre(curfmat, memory=memory)

                redvdiff = memory['cur_tl'].T.dot(mmat*(curvel-vinf))
                actua = -b_mat_reg.dot(memory['baseGain'].dot(redvdiff))
                return actua, memory

                # sdrefdstr = sdre_ric_ini  # TODO: debugging here
                # czwc, czwo = nru.\
                #     get_ric_facs(fdstr=sdrefdstr, fmat=-curfmat, mmat=mmat,
                #                  jmat=jmat,
                #                  bmat=b_mat_reg, cmat=c_mat_reg,
                #                  ric_ini_str=sdre_ric_ini,
                #                  Rmhalf=Rmhalf, nwtn_adi_dict=nwtn_adi_dict,
                #                  zwconly=False, multiproc=multiproc)

            # ## updated sdre feedback
            cur_tr = memory['cur_tr']
            cur_tl = memory['cur_tl']
            curak = -cur_tl.T.dot(curfmat.dot(cur_tr))
            # print('norm Ak: {0}'.format(np.linalg.norm(curak)))
            # print('diff: `akbase-aknow`: {0}'.
            #       format(np.linalg.norm(curak-memory['baseAk'])))
            redvdiff = memory['cur_tl'].T.dot(mmat*(curvel-vinf))
            print('diff: `curv-linv`: {0}'.format(np.linalg.norm(redvdiff)))
            print('|basegain|: {0}'.format(np.linalg.norm(memory['baseGain'])))
            pupd, elteps = nru.\
                get_sdrefb_upd(curak, time, fbtype='sylvupdfb', wnrm=2,
                               baseA=memory['baseAk'], baseZ=memory['baseZk'],
                               baseP=memory['basePk'],
                               maxfac=None, maxeps=updtthrsh)

            if elteps:  # E less than eps
                updGain = memory['cur_bk'].T.dot(pupd)
                redvdiff = memory['cur_tl'].T.dot(mmat*(curvel-vinf))
                actua = -b_mat_reg.dot(updGain.dot(redvdiff))
                return actua, memory

            else:
                tl = memory['cur_tl']
                tr = memory['cur_tr']
                prvvel = memory['basev']
                prvfmat = get_cur_sdccoeff(vcur=prvvel)
                prvak = -tl.T.dot(prvfmat.dot(tr))
                basak = memory['baseAk']
                print('|prv Ak|: {0}'.format(norm(prvak)))
                print('|cur Ak|: {0}'.format(norm(curak)))
                print('|bas Ak|: {0}'.format(norm(basak)))

                savethev = np.copy(curvel)
                memory.update(basev=savethev)
                # memory.update(basev=curvel)
                solve_sdre(curfmat, memory=memory, eps=updtthrsh, time=time)
                redvdiff = memory['cur_tl'].T.dot(mmat*(curvel-vinf))
                actua = -b_mat_reg.dot(memory['baseGain'].dot(redvdiff))
                return actua, memory

            # zwc = memory['czwc']

            # btxm_mat = pru.get_mTzzTtb(stokesmatsc['M'].T, zwc, b_mat_reg).T
            # actua = -lau.comp_uvz_spdns(b_mat_reg, btxm_mat, curvel-vinf)
            # actua = -lau.comp_uvz_spdns(b_mat_reg, cur_f,
            #                             cur_tl.T.dot(mmat*(curvel-vinf)))
            # diffv = curvel-vinf
            # rldiffv = cur_tr.dot(cur_tr.T.dot(mmat*(diffv)))
            # nrmrldiffv = np.linalg.norm(rldiffv)
            # print('norm of prj/lfd veldiff: {0}'.format(nrmrldiffv))
            # import ipdb; ipdb.set_trace()
            return actua, memory

        fv_sdre_dict = dict(updtthrsh=.9)

        fv_tmdp = sdre_feedback
        fv_tmdp_params = fv_sdre_dict
        fv_tmdp_memory = dict(basePk=None)

        # 1. prelims
        #    * func:get Riccati Grams at current state
        #    * get reduced model (func: Ak(vdelta(t)))
        # 2. func: sdre_fblaw
        #    compute E -- solve sylvester
        #    reset Grams and Ak, Bk, etc

    elif closed_loop == 'redmod_sdre_fb':
        shortclstr = 'rdmdsdrfb'

        ak_mat, bk_mat, ck_mat, _, _, basetl, basetr = \
            nru.get_prj_model(truncstr=truncstr, fdstr=fdstr,
                              mmat=mmat, fmat=-f_mat_gramians, jmat=jmat,
                              bmat=b_mat_reg, cmat=c_mat_reg,
                              cmpricfacpars=cmpricfacpars,
                              Rmhalf=Rmhalf,
                              cmprlprjpars=cmprlprjpars)
        cktck = ck_mat.T.dot(ck_mat)

        sdcpicard = 1.
        vinf = v_ss_nse
        get_cur_sdccoeff = neu.get_get_cur_extlin(vinf=vinf, amat=amat,
                                                  picrdvsnwtn=sdcpicard,
                                                  **femp)

        sdre_ric_ini = fdstr
        cmpricfacpars.update(ric_ini_str=sdre_ric_ini)

        def redsdre_feedback(curvel=None, memory=None,
                             updtthrsh=None, time=None, **kw):
            ''' function for the SDRE feedback

            '''

            norm = np.linalg.norm
            # print('time: {0}, |v|: {1}'.format(time, norm(curvel)))
            # print('time: {0}, |vinf|: {1}'.format(time, norm(vinf)))
            curfmat = get_cur_sdccoeff(vcur=curvel)
            curak = -basetl.T.dot(curfmat.dot(basetr))
            redvdiff = basetl.T.dot(mmat*(curvel-vinf))

            print('time: {0:.5f} -- reddiff: `curv-linv`: {1}'.
                  format(time, norm(redvdiff)))

            pupd, elteps = nru.\
                get_sdrefb_upd(curak, time, fbtype='sylvupdfb', wnrm=2,
                               baseA=memory['baseAk'], baseZ=memory['baseZk'],
                               baseP=memory['basePk'],
                               B=bk_mat, Q=cktck,
                               R=Rmo*np.eye(bk_mat.shape[1]),
                               maxfac=None, maxeps=updtthrsh)

            if time > 0.006:
                prvvel = memory['prevvel']
                difftopv = prvvel - curvel
                difff = get_cur_sdccoeff(vcur=curvel) - \
                    get_cur_sdccoeff(vcur=prvvel)
                diffak = -basetl.T.dot(difff.dot(basetr))
                ndiffak = norm(diffak)
                import ipdb; ipdb.set_trace()

            actua = -b_mat_reg.dot(bk_mat.T.dot(pupd.dot(redvdiff)))

            if not elteps:
                baseZk = curak - bk_mat.dot(bk_mat.T.dot(pupd))
                memory.update(dict(basePk=pupd, baseAk=curak, baseZk=baseZk))

            memory.update(prevvel=curvel)

            return actua, memory

        fv_sdre_dict = dict(updtthrsh=.9)

        fv_tmdp = redsdre_feedback
        fv_tmdp_params = fv_sdre_dict
        fv_tmdp_memory = dict(basePk=None, baseAk=None, baseZk=None)

    else:
        fv_tmdp = None
        fv_tmdp_params = {}
        fv_tmdp_memory = {}
        shortclstr = '_'

    soldict.update(fv_stbc=rhsd_stbc['fv'],
                   trange=trange,
                   lin_vel_point=None,
                   clearprvdata=True,
                   fv_tmdp=fv_tmdp,
                   comp_nonl_semexp=True,
                   fv_tmdp_params=fv_tmdp_params,
                   fv_tmdp_memory=fv_tmdp_memory,
                   return_dictofvelstrs=True)

    if whichinival == 'sstokes':
        print('we start with Stokes -- `perturbpara` is not considered')
        soldict.update(dict(iniv=None, start_ssstokes=True))
        shortinivstr = 'sks'
    elif whichinival == 'sstate+d':
        perturbini = perturbpara*np.ones((NV, 1))
        reg_pertubini = lau.app_prj_via_sadpnt(amat=stokesmatsc['M'],
                                               jmat=stokesmatsc['J'],
                                               rhsv=perturbini)
        soldict.update(dict(iniv=v_ss_nse + reg_pertubini))
        shortinivstr = 'ssd'
    elif whichinival == 'sstokes++':
        lctrng = (trange[trange < tpp]).tolist()
        lctrng.append(tpp)

        stksppdtstr = fdstr + 't0{0:.1f}tE{1:.4f}Nts{2}'.\
            format(t0, tpp, len(lctrng)) + '__stokesppvel'
        try:
            sstokspp = dou.load_npa(stksppdtstr)
            print('loaded `stokespp({0})` for inival'.format(tpp))
        except IOError:
            print('solving for `stokespp({0})` as inival'.format(tpp))
            inivsoldict = {}
            inivsoldict.update(stokesmatsc)  # containing A, J, JT
            inivsoldict.update(femp)  # adding V, Q, invinds, diribcs
            inivsoldict.update(fv=rhsd_stbc['fv']+rhsd_vfrc['fvc'],
                               fp=rhsd_stbc['fp']+rhsd_vfrc['fpr'],
                               N=N, nu=nu, data_prfx=veldatastr)
            inivsoldict.update(trange=np.array(lctrng),
                               iniv=None, start_ssstokes=True,
                               comp_nonl_semexp=True,
                               return_dictofvelstrs=True)
            dcvlstrs = snu.solve_nse(**inivsoldict)
            sstokspp = dou.load_npa(dcvlstrs[tpp])
            dou.save_npa(sstokspp, stksppdtstr)
        soldict.update(dict(iniv=sstokspp))
        shortinivstr = 'sk{0}'.format(tpp)

    checkdaredmod = True
    checkdaredmod = False
    if checkdaredmod:
        import spacetime_galerkin_pod.gen_pod_utils as gpu
        # akm = basetl.T.dot(amat*basetr)
        # nk = basetl.shape[1]

        redmod = False
        redmod = True
        if redmod:
            curiniv = basetl.T.dot(mmat*(soldict['iniv']-vinf))
            nk = basetr.shape[1]

            def rednonl(vvec, t):
                inflv = basetr.dot(vvec.reshape((nk, 1)))
                curcoeff = get_cur_sdccoeff(vdelta=inflv)
                returval = basetl.T.dot(curcoeff.dot(inflv))
                return returval.flatten()
            curnonl = rednonl
            tstrunstr = 'testdaredmod'
            mmatforlsoda = None
            tstc = ck_mat

        else:
            curiniv = soldict['iniv'] - vinf
            NV = mmat.shape[0]

            def fulnonl(vvec, t):
                curcoeff = get_cur_sdccoeff(vvec.reshape((NV, 1)))
                apconv = curcoeff.dot(vvec.reshape((NV, 1)))
                prjapc = lau.app_prj_via_sadpnt(amat=mmat, jmat=jmat,
                                                rhsv=apconv)
                return prjapc.flatten()
            mmatforlsoda = mmat
            curnonl = fulnonl
            tstrunstr = 'testdafulmod'
            tstc = c_mat

        print('doing the `lsoda` integration...')
        tstsol = gpu.time_int_semil(tmesh=trange, A=None, M=mmatforlsoda,
                                    nfunc=curnonl, iniv=curiniv)

        print('done with the `lsoda` integration!')
        outptlst = []
        for kline in range(tstsol.shape[0]):
            # outptlst.append((ck_mat.dot(redsol[k, :])).tolist())
            outptlst.append((tstc.dot(tstsol[kline, :])).tolist())
        dou.save_output_json(dict(tmesh=trange.tolist(), outsig=outptlst),
                             fstring=tstrunstr)
        import ipdb; ipdb.set_trace()

    outstr = truncstr + '{0}'.format(closed_loop) \
        + 't0{0}tE{1}Nts{2}N{3}Re{4}'.format(t0, tE, Nts, N, Re)
    if paraoutput:
        soldict.update(paraviewoutput=True,
                       vfileprfx='results/vel_'+outstr,
                       pfileprfx='results/p_'+outstr)

    shortstring = (get_fdstr(Re, short=True) + shortcontsetupstr +
                   shortclstr + shorttruncstr + shortinivstr)
    soldict.update(data_prfx=shortstring)
    dictofvelstrs = snu.solve_nse(**soldict)

    yscomplist = cou.extract_output(strdict=dictofvelstrs, tmesh=trange,
                                    c_mat=c_mat, load_data=dou.load_npa)

    if robit:
        robitstr = '_robmgnfac{0}'.format(robmrgnfac)
    else:
        robitstr = ''

    dou.save_output_json(dict(tmesh=trange.tolist(), outsig=yscomplist),
                         fstring=fdstr + truncstr + '{0}'.format(closed_loop) +
                         't0{0:.4f}tE{1:.4f}Nts{2}'.format(t0, tE,
                                                           np.int(Nts)) +
                         'inipert{0}'.format(perturbpara) + robitstr)

    if plotit:
        dou.plot_outp_sig(tmesh=trange, outsig=yscomplist)
    # import matplotlib.pyplot as plt
    # plt.plot(trange, yscomplist)
    # plt.show(block=False)

if __name__ == '__main__':
    # lqgbt(N=10, Re=500, use_ric_ini=None, plain_bt=False)
    lqgbt(problemname='cylinderwake', N=3,  # use_ric_ini=2e2,
          Re=7.5e1, plain_bt=False,
          t0=0.0, tE=2.0, Nts=1e3+1, palpha=1e-6,
          comp_freqresp=False, comp_stepresp=False)
