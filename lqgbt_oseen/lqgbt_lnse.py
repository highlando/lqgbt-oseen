import os
import numpy as np
import scipy.sparse as sps

import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

import sadptprj_riclyap_adi.lin_alg_utils as lau
import sadptprj_riclyap_adi.proj_ric_utils as pru
import sadptprj_riclyap_adi.bal_trunc_utils as btu

import distr_control_fenics.cont_obs_utils as cou

import lqgbt_oseen.nse_riccont_utils as nru
import lqgbt_oseen.nse_extlin_utils as neu
import lqgbt_oseen.cntrl_simu_helpers as csh

debug = False

checktheres = True  # whether to check the Riccati Residuals
checktheres = False

switchonsfb = 0  # 1.5

# TODO: clear distinction of target state, linearization point, initial value
# TODO: maybe redefine: by now we need to use -fmat all the time (but +ak_mat)
# TODO: outsource the contrlr definitions


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
          N=10, Re=1e2, plain_bt=False, cl_linsys=False,
          simuN=None,
          gamma=1.,
          use_ric_ini=None, t0=0.0, tE=1.0, Nts=11,
          NU=3, Cgrid=(3, 1),
          bccontrol=True, palpha=1e-5,
          npcrdstps=8,
          pymess=False,
          paraoutput=True,
          plotit=True,
          trunc_lqgbtcv=1e-6,
          hinf=False,
          hinfgammainfty=False,
          nwtn_adi_dict=None,
          pymess_dict=None,
          whichinival='sstate',
          tpp=5.,  # time to add on Stokes inival for `sstokes++`
          comp_freqresp=False, comp_stepresp='nonlinear',
          closed_loop=False, multiproc=False,
          perturbpara=1e-3,
          trytofail=False, ttf_npcrdstps=3):
    """Main routine for LQGBT

    Parameters
    ----------
    closed_loop : {'full_state_fb', 'red_output_fb', False, None}
        how to do the closed loop simulation:

        | if False -> no simulation
        | if == 'full_state_fb' -> full state feedback
        | if == 'red_output_fb' -> reduced output feedback
        | else -> no control is applied

        defaults to `False`
    comp_freqresp : boolean, optional
        whether to compute and compare the frequency responses,
        defaults to `False`
    comp_stepresp : {'nonlinear', False, None}
        whether to compute and compare the step responses

        | if False -> no step response
        | if == 'nonlinear' -> compare linear reduced to nonlinear full model
        | else -> linear reduced versus linear full model

        defaults to `False`
    bccontrol : boolean, optional
        whether to apply boundary control via penalized robin conditions,
        defaults to `False`
    gamma : real, optional
        regularization parameter, puts weight on `|u|` in the underlying
        LQR cost functional that, defaults to `1.`
    hinf : boolean, optional
        whether to compute the normalized hinf aware central controller
    N : int, optional
        parameter for the dimension of the space discretization
    NU : int, optional
        defines the `B` and, thus, the dimension the input space
        (will double because there are two components), defaults to `3`
    Cgrid : tuple, optional
        defines the `C` and, thus, the dimension of the output space
        (dim Y = 2*Cgrid[0]*Cgrid[1]), defaults to `(3, 1)`
    problemname : string, optional
        what problem to be solved, 'cylinderwake' or 'drivencavity'
    plain_bt : boolean, optional
        whether to try simple *balanced truncation*, defaults to False
    Re : real, optional
        Reynolds number, defaults to `1e2`
    t0, tE, Nts : real, real, int, optional
        starting and endpoint of the considered time interval, number of
        time instancses, default to `0.0, 1.0, 11`
    trunc_lqgbtcv : real, optional
        threshold at what the lqgbt characteristiv values are truncated,
        defaults to `1e-6`
    use_ric_ini : real, optional
        use the solution with this Re number as stabilizing initial guess,
        defaults to `None`
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

    contsetupstr = 'NV{0}_B{3}_C{1[0]}{1[1]}_palpha{2}'.\
        format(NV, Cgrid, palpha, NU)
    shortcontsetupstr = '{0}{1[0]}{1[1]}{2}'.\
        format(NV, Cgrid, np.int(np.log2(palpha)))

    if bccontrol:
        stokesmatsc['A'] = stokesmatsc['A'] + 1./palpha*stokesmatsc['Arob']
        b_mat = 1./palpha*stokesmatsc['Brob']
        u_masmat = sps.eye(b_mat.shape[1], format='csr')
        print(' ### Robin-type boundary control palpha={0}'.format(palpha))

    if whichinival == 'sstokes++' or whichinival == 'snse+d++':
        inivstr = '_' + whichinival + '{0}'.format(tpp)
    else:
        inivstr = '_' + whichinival

    def get_fdstr(Re, short=False):
        if short:
            return ddir + 'cw' + '{0}{1}_'.format(Re, gamma) + \
                shortcontsetupstr
        return ddir + problemname + '_Re{0}_gamma{1}_'.format(Re, gamma) + \
            contsetupstr + prbstr

    fdstr = get_fdstr(Re)
    fdstr = fdstr + '_hinf' if hinf else fdstr
    # if hinf and hinfgammainfty:
    #     fdstr = fdstr + 'hinf'
    fdstrini = get_fdstr(use_ric_ini) if use_ric_ini is not None else None

#
# ### CHAP: Prepare for control
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

    # Rmo = 1./gamma
    Rmhalf = 1./np.sqrt(gamma)
    b_mat_rgscld = b_mat_reg*Rmhalf
    # We scale the input matrix to acommodate for input weighting

    try:
        mc_mat = dou.load_spa(ddir + contsetupstr + '__mc_mat')
        y_masmat = dou.load_spa(ddir + contsetupstr + '__y_masmat')
        print(('loaded `c_mat` from' + ddir + contsetupstr + '...'))
    except IOError:
        print('computing `c_mat`...')
        mc_mat, y_masmat = cou.get_mout_opa(odcoo=femp['odcoo'],
                                            V=femp['V'], mfgrid=Cgrid)
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

    vp_ss_nse = snu.\
        solve_steadystate_nse(vel_pcrd_stps=npcrdstps, return_vp=True,
                              clearprvdata=debug, **soldict)

    v_ss_nse = vp_ss_nse[:NV]
    p_ss_nse = vp_ss_nse[NV:]
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
        v_ss_nse_MAF = snu.\
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
        shortfailstr = 'maf{0}'.format(ttf_npcrdstps)
    else:
        f_mat_gramians = f_mat
        shortfailstr = ''

#
# ### Compute or get the Gramians
#
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
                                        bmat=b_mat_rgscld, cmat=c_mat_reg,
                                        cmpricfacpars=cmpricfacpars,
                                        pymess=pymess,
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
                               jmat=jmat, bmat=b_mat_rgscld, cmat=c_mat_reg,
                               ric_ini_str=fdstrini,
                               nwtn_adi_dict=nwtn_adi_dict, zwconly=True,
                               multiproc=multiproc, pymess=pymess,
                               checktheres=False)

        mtxb = pru.get_mTzzTtb(stokesmatsc['M'].T, zwc, b_mat_rgscld)

        dimu = b_mat.shape[1]
        zerou = np.zeros((dimu, 1))
        if switchonsfb > 0:
            print('the feedback will be switched on at ' +
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

        tmdp_fsfb_dict = dict(linv=v_ss_nse, tb_mat=b_mat_rgscld,
                              btxm_mat=mtxb.T)

        fv_tmdp = fv_tmdp_fullstatefb
        fv_tmdp_params = tmdp_fsfb_dict
        fv_tmdp_memory = None

    # ### CHAP: define the reduced output feedback
    elif closed_loop == 'red_output_fb':
        shortclstr = 'hinfrofb' if hinf else 'rofb'
        if hinfgammainfty:
            shortclstr = 'hinfrofbginf'
        DT = (tE - t0)/(Nts-1)

        ak_mat, bk_mat, ck_mat, xok, xck, hinfgamma, tl, tr = \
            nru.get_prj_model(truncstr=truncstr, fdstr=fdstr,
                              abconly=False,
                              mmat=mmat, fmat=f_mat_gramians, jmat=jmat,
                              bmat=b_mat_rgscld, cmat=c_mat_reg,
                              cmpricfacpars=cmpricfacpars,
                              pymess=pymess,
                              hinf=hinf, hinfgammainfty=hinfgammainfty,
                              cmprlprjpars=cmprlprjpars)
        print('Controller has dimension: {0}'.format(ak_mat.shape[0]))

        if hinf and not hinfgammainfty:
            zk = np.linalg.inv(np.eye(xck.shape[0])
                               - 1./hinfgamma**2*xok.dot(xck))
            amatk = (ak_mat
                     - (1. - 1./hinfgamma**2)*np.dot(np.dot(xok, ck_mat.T),
                                                     ck_mat)
                     - np.dot(bk_mat, np.dot(bk_mat.T, xck).dot(zk)))
            obs_ck = -np.dot(bk_mat.T.dot(xck), zk)

        else:
            amatk = (ak_mat - np.dot(np.dot(xok, ck_mat.T), ck_mat) -
                     np.dot(bk_mat, np.dot(bk_mat.T, xck)))
            obs_ck = -bk_mat.T.dot(xck)

        obs_bk = np.dot(xok, ck_mat.T)
        sysmatk_inv = np.linalg.inv(np.eye(ak_mat.shape[1]) - DT*amatk)

        def fv_tmdp_redoutpfb(time=None, memory=None,
                              cury=None, ystar=None,
                              curvel=None, velstar=None, c_mat=None,
                              ipsysk_mat_inv=None, cts=None,
                              obs_bk=None, obs_ck=None,
                              b_mat=None,
                              **kw):
            """realizes a reduced static output feedback as a function

            that can be passed to a solution routine for the
            unsteady Navier-Stokes equations

            Parameters
            ----------
            time : real
                current time
            memory : dictionary
                contains values from previous call, in particular the
                previous state estimate
            curvel : (N,1) nparray
                current velocity. For consistency, the full state can be taken
                as input. Internally, the observation `y=c_mat*curvel` is used
            velstar : (N,1) nparray
                target velocity
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
            c_mat : (NY,N) sparse matrix
                output matrix of the full system

            Returns
            -------
            actua : (N,1) nparray
                the current actuation
            memory : dictionary
                to be passed back in the next timestep

            """
            xk_old = memory['xk_old']
            if cury is not None and ystar is not None:
                ydiff = cury - ystar
            else:
                ydiff = c_mat.dot(curvel-velstar)
                print('sorry, I used the full state for y=Cx ...')
            # print('ydiff', ydiff)
            buk = cts*np.dot(obs_bk, ydiff)
            xk_old = np.dot(ipsysk_mat_inv, xk_old + buk)
            # print(obs_ck.dot(xk_old))
            memory['xk_old'] = xk_old
            actua = b_mat.dot(obs_ck.dot(xk_old))
            memory['actualist'].append(actua)

            return actua, memory

        fv_rofb_dict = dict(cts=DT,
                            ystar=c_mat.dot(v_ss_nse),
                            # velstar=v_ss_nse, c_mat=c_mat_reg,
                            b_mat=b_mat_rgscld,
                            obs_bk=obs_bk, obs_ck=obs_ck,
                            ipsysk_mat_inv=sysmatk_inv)

        fv_tmdp = fv_tmdp_redoutpfb
        fv_tmdp_params = fv_rofb_dict
        fv_tmdp_memory = dict(xk_old=np.zeros((tl.shape[1], 1)),
                              actualist=[])

    elif closed_loop == 'full_sdre_fb':
        shortclstr = 'fsdrfb'

        sdcpicard = 1.
        vinf = v_ss_nse
        get_cur_sdccoeff = neu.get_get_cur_extlin(vinf=vinf, amat=amat,
                                                  picrdvsnwtn=sdcpicard,
                                                  **femp)
        sdre_ric_ini = fdstr

        def _lrymts(fmat, leftonly=False, rightonly=False, thrtonly=False):
            if leftonly:
                return minvthr.T.dot(fmat)
            elif rightonly:
                return fmat.dot(thl)
            elif thrtonly:
                return thr.T.dot(fmat)
            else:
                return minvthr.T.dot(fmat.dot(thl))

        thrminvbmat = _lrymts(b_mat_rgscld, leftonly=True)

        def solve_full_sdre(curfmat, memory=None, eps=None, time=None):
            if memory['baseGain'] is None:  # initialization
                sdrefdstr = fdstr + inivstr + '_SDREini'
            else:
                sdrefdstr = (fdstr + inivstr +
                             '_SDREeps{0}t{1:.2e}'.format(eps, time))

            cmpricfacpars.update(ric_ini_str=sdrefdstr)
            zwc = nru.get_ric_facs(fdstr=sdrefdstr,
                                   fmat=curfmat, mmat=mmat,
                                   jmat=jmat, bmat=b_mat_rgscld,
                                   cmat=c_mat_reg,
                                   ric_ini_str=sdre_ric_ini,
                                   nwtn_adi_dict=nwtn_adi_dict,
                                   zwconly=True,
                                   multiproc=False, pymess=pymess,
                                   checktheres=False)

            mtxb = pru.get_mTzzTtb(stokesmatsc['M'].T, zwc, b_mat_rgscld)
            baseA = _lrymts(curfmat)
            baseGain = _lrymts(mtxb.T, rightonly=True)
            baseZ = baseA - thrminvbmat.dot(baseGain)
            memory.update(baseA=baseA)
            memory.update(baseGain=baseGain)
            memory.update(baseZ=baseZ)
            # XXX: memory!
            return

        def updsdre_feedback(curvel=None, memory=None,
                             updtthrsh=None, time=None, **kw):
            ''' function for the SDRE feedback

            Parameters
            ---
            use_ric_ini : string, optional
                path to a stabilizing initial guess
            '''

            curfmat = (-1)*get_cur_sdccoeff(vcur=curvel)
            thrtvdiff = _lrymts(curvel-vinf, thrtonly=True)

            if memory['baseGain'] is None:  # initialization
                savethev = np.copy(curvel)
                memory.update(basev=savethev)
                solve_full_sdre(curfmat, memory=memory)

                thrtvdiff = _lrymts(curvel-vinf, thrtonly=True)
                actua = -b_mat_rgscld.dot(memory['baseGain'].dot(thrtvdiff))
                return actua, memory

            # ## updated sdre feedback
            # nomvdiff = curvel-vinf
            # print('diff: `curv-linv`: {0}'.format(np.linalg.norm(nomvdiff)))

            curak = _lrymts(curfmat)
            print('solving for the update...')
            updGain, elteps = nru.\
                get_sdrgain_upd(curak, wnrm='fro',
                                baseA=memory['baseA'],
                                baseZ=memory['baseZ'],
                                baseGain=memory['baseGain'],
                                maxfac=None, maxeps=updtthrsh)

            if elteps:  # E less than eps
                print('update is small, yeyeye...')
                actua = -b_mat_rgscld.dot(updGain.dot(thrtvdiff))
                return actua, memory

            else:
                # prvvel = memory['basev']
                # prvfmat = get_cur_sdccoeff(vcur=prvvel)
                # prvak = -tl.T.dot(prvfmat.dot(tr))
                # basak = memory['baseAk']
                # print('|prv Ak|: {0}'.format(norm(prvak)))
                # print('|cur Ak|: {0}'.format(norm(curak)))
                # print('|bas Ak|: {0}'.format(norm(basak)))

                savethev = np.copy(curvel)
                memory.update(basev=savethev)
                # memory.update(basev=curvel)
                solve_full_sdre(curfmat, memory=memory, eps=updtthrsh,
                                time=time)
                thrtvdiff = _lrymts(curvel-vinf, thrtonly=True)
                actua = -b_mat_rgscld.dot(memory['baseGain'].dot(thrtvdiff))
                return actua, memory

            return actua, memory

        fv_sdre_dict = dict(updtthrsh=.9)

        fv_tmdp = updsdre_feedback
        fv_tmdp_params = fv_sdre_dict
        fv_tmdp_memory = dict(baseGain=None)
    elif closed_loop == 'full_updsdre_fb':
        shortclstr = 'fupsdrfb'

        sdcpicard = 1.
        vinf = v_ss_nse
        get_cur_sdccoeff = neu.get_get_cur_extlin(vinf=vinf, amat=amat,
                                                  picrdvsnwtn=sdcpicard,
                                                  **femp)
        sdre_ric_ini = fdstr
        cmpricfacpars.update(ric_ini_str=sdre_ric_ini)
        # we decompose the Leray projector `Pi=thl*thr.T`
        leraystr = (ddir + 'discleray_' + problemname +
                    'Nv{0}Np{1}'.format(jmat.shape[0], jmat.shape[1]))
        thl, thr, minvthr = neu.decomp_leray(mmat, jmat, dataprfx=leraystr)

        def _lrymts(fmat, leftonly=False, rightonly=False, thrtonly=False):
            if leftonly:
                return minvthr.T.dot(fmat)
            elif rightonly:
                return fmat.dot(thl)
            elif thrtonly:
                return thr.T.dot(fmat)
            else:
                return minvthr.T.dot(fmat.dot(thl))

        thrminvbmat = _lrymts(b_mat_rgscld, leftonly=True)

        def solve_full_sdre(curfmat, memory=None, eps=None, time=None):
            if memory['baseGain'] is None:  # initialization
                sdrefdstr = fdstr + inivstr + '_SDREini'
            else:
                sdrefdstr = (fdstr + inivstr +
                             '_SDREeps{0}t{1:.2e}'.format(eps, time))

            cmpricfacpars.update(ric_ini_str=sdrefdstr)
            zwc = nru.get_ric_facs(fdstr=sdrefdstr,
                                   fmat=curfmat, mmat=mmat,
                                   jmat=jmat, bmat=b_mat_rgscld,
                                   cmat=c_mat_reg,
                                   ric_ini_str=sdre_ric_ini,
                                   nwtn_adi_dict=nwtn_adi_dict,
                                   zwconly=True,
                                   multiproc=False, pymess=pymess,
                                   checktheres=False)

            mtxb = pru.get_mTzzTtb(stokesmatsc['M'].T, zwc, b_mat_rgscld)
            baseA = _lrymts(curfmat)
            baseGain = _lrymts(mtxb.T, rightonly=True)
            baseZ = baseA - thrminvbmat.dot(baseGain)
            memory.update(baseA=baseA)
            memory.update(baseGain=baseGain)
            memory.update(baseZ=baseZ)
            # XXX: memory!
            return

        def updsdre_feedback(curvel=None, memory=None,
                             updtthrsh=None, time=None, **kw):
            ''' function for the SDRE feedback

            Parameters
            ---
            use_ric_ini : string, optional
                path to a stabilizing initial guess
            '''

            curfmat = (-1)*get_cur_sdccoeff(vcur=curvel)
            thrtvdiff = _lrymts(curvel-vinf, thrtonly=True)

            if memory['baseGain'] is None:  # initialization
                savethev = np.copy(curvel)
                memory.update(basev=savethev)
                solve_full_sdre(curfmat, memory=memory)

                thrtvdiff = _lrymts(curvel-vinf, thrtonly=True)
                actua = -b_mat_rgscld.dot(memory['baseGain'].dot(thrtvdiff))
                return actua, memory

            # ## updated sdre feedback
            # nomvdiff = curvel-vinf
            # print('diff: `curv-linv`: {0}'.format(np.linalg.norm(nomvdiff)))

            curak = _lrymts(curfmat)
            print('solving for the update...')
            updGain, elteps = nru.\
                get_sdrgain_upd(curak, wnrm='fro',
                                baseA=memory['baseA'],
                                baseZ=memory['baseZ'],
                                baseGain=memory['baseGain'],
                                maxfac=None, maxeps=updtthrsh)

            if elteps:  # E less than eps
                print('update is small, yeyeye...')
                actua = -b_mat_rgscld.dot(updGain.dot(thrtvdiff))
                return actua, memory

            else:
                # prvvel = memory['basev']
                # prvfmat = get_cur_sdccoeff(vcur=prvvel)
                # prvak = -tl.T.dot(prvfmat.dot(tr))
                # basak = memory['baseAk']
                # print('|prv Ak|: {0}'.format(norm(prvak)))
                # print('|cur Ak|: {0}'.format(norm(curak)))
                # print('|bas Ak|: {0}'.format(norm(basak)))

                savethev = np.copy(curvel)
                memory.update(basev=savethev)
                # memory.update(basev=curvel)
                solve_full_sdre(curfmat, memory=memory, eps=updtthrsh,
                                time=time)
                thrtvdiff = _lrymts(curvel-vinf, thrtonly=True)
                actua = -b_mat_rgscld.dot(memory['baseGain'].dot(thrtvdiff))
                return actua, memory

            return actua, memory

        fv_sdre_dict = dict(updtthrsh=.9)

        fv_tmdp = updsdre_feedback
        fv_tmdp_params = fv_sdre_dict
        fv_tmdp_memory = dict(baseGain=None)

        # 1. prelims
        #    * func:get Riccati Grams at current state
        #    * get reduced model (func: Ak(vdelta(t)))
        # 2. func: sdre_fblaw
        #    compute E -- solve sylvester
        #    reset Grams and Ak, Bk, etc

    elif closed_loop == 'red_updsdre_fb':
        shortclstr = 'rdsdrfb'

        sdcpicard = 1.
        vinf = v_ss_nse
        get_cur_sdccoeff = neu.get_get_cur_extlin(vinf=vinf, amat=amat,
                                                  picrdvsnwtn=sdcpicard,
                                                  **femp)
        sdre_ric_ini = fdstr
        cmpricfacpars.update(ric_ini_str=sdre_ric_ini)

        def solve_red_sdre(curfmat, memory=None, eps=None, time=None):
            if memory['basePk'] is None:  # initialization
                sdrefdstr = fdstr + inivstr + '_SDREini'
            else:
                sdrefdstr = (fdstr + inivstr +
                             '_SDREeps{0}t{1:.2e}'.format(eps, time))
            ak_mat, cur_bk, _, _, cxck, cur_tl, cur_tr = \
                nru.get_prj_model(truncstr=truncstr, fdstr=sdrefdstr,
                                  mmat=mmat, fmat=-curfmat, jmat=jmat,
                                  bmat=b_mat_rgscld, cmat=c_mat_reg,
                                  pymess=pymess,
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
                solve_red_sdre(curfmat, memory=memory)

                redvdiff = memory['cur_tl'].T.dot(mmat*(curvel-vinf))
                actua = -b_mat_rgscld.dot(memory['baseGain'].dot(redvdiff))
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
                actua = -b_mat_rgscld.dot(updGain.dot(redvdiff))
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
                solve_red_sdre(curfmat, memory=memory, eps=updtthrsh,
                               time=time)
                redvdiff = memory['cur_tl'].T.dot(mmat*(curvel-vinf))
                actua = -b_mat_rgscld.dot(memory['baseGain'].dot(redvdiff))
                return actua, memory

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
                              bmat=b_mat_rgscld, cmat=c_mat_reg,
                              cmpricfacpars=cmpricfacpars,
                              pymess=pymess,
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
                               maxfac=None, maxeps=updtthrsh)

            actua = -b_mat_rgscld.dot(bk_mat.T.dot(pupd.dot(redvdiff)))

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

    shortclstr = shortclstr + 'pm' if pymess else shortclstr

    soldict.update(trange=trange,
                   lin_vel_point=None,
                   clearprvdata=True,
                   fv_tmdp=fv_tmdp,
                   cv_mat=c_mat,  # needed for the output feedback
                   comp_nonl_semexp=True,
                   fv_tmdp_params=fv_tmdp_params,
                   fv_tmdp_memory=fv_tmdp_memory,
                   return_dictofvelstrs=True)

    # ### CHAP: define the initial values
    if simuN == N:
        shortinivstr, _ = csh.\
            set_inival(soldict=soldict, whichinival=whichinival, trange=trange,
                       tpp=tpp, v_ss_nse=v_ss_nse, perturbpara=perturbpara,
                       fdstr=fdstr)
        shortstring = (get_fdstr(Re, short=True) + shortclstr +
                       shorttruncstr + shortinivstr + shortfailstr)

    outstr = truncstr + '{0}'.format(closed_loop) \
        + 't0{0}tE{1}Nts{2}N{3}Re{4}'.format(t0, tE, Nts, N, Re)
    if paraoutput:
        soldict.update(paraviewoutput=True,
                       vfileprfx='results/vel_'+outstr,
                       pfileprfx='results/p_'+outstr)

    if cl_linsys:
        adjlinsys = True
        if adjlinsys:

            # XXX: only works if `closed_loop == 'full_state_fb'`
            zwc, zwo = nru.get_ric_facs(fdstr=fdstr,
                                        fmat=f_mat_gramians, mmat=mmat,
                                        jmat=jmat, bmat=b_mat_rgscld,
                                        cmat=c_mat_reg, ric_ini_str=fdstrini,
                                        nwtn_adi_dict=nwtn_adi_dict,
                                        multiproc=multiproc, pymess=pymess,
                                        checktheres=False)
            mtxct = pru.get_mTzzTtb(stokesmatsc['M'].T, zwo, c_mat_reg.T)
            adj_fsfb_dict = dict(linv=v_ss_nse, tb_mat=c_mat_reg.T,
                                 btxm_mat=mtxct.T)

            adj_fv_tmdp = fv_tmdp_fullstatefb
            adj_fv_tmdp_params = adj_fsfb_dict
            adj_fv_tmdp_memory = None
            # TODO: we only solve the difference system
            # the full system will require the proper right hand sides
            adjlinsysrhs = 0*soldict['fv']
            adjlinsysrhsfp = 0*soldict['fp']
            # adjiniv = soldict['iniv']
            # adjiniv_reg = lau.app_prj_via_sadpnt(amat=stokesmatsc['M'],
            #                                      jmat=stokesmatsc['J'],
            #                                      rhsv=adjiniv,
            #                                      transposedprj=False)
            # soldict.update(iniv=adjiniv_reg)

            soldict.update(fv_tmdp=adj_fv_tmdp,
                           comp_nonl_semexp=True,
                           fv_tmdp_params=adj_fv_tmdp_params,
                           fv_tmdp_memory=adj_fv_tmdp_memory)

            soldict.update(stokes_flow=True, A=-f_mat.T,
                           fv=adjlinsysrhs,
                           fp=adjlinsysrhsfp)
            soldict.update(data_prfx=shortstring + '_adj')
            print('norm fp: {0}'.format(np.linalg.norm(soldict['fp'])))
            print('norm fv: {0}'.format(np.linalg.norm(soldict['fv'])))
            dictofvelstrs = snu.solve_nse(**soldict)

            yscomplist = cou.\
                extract_output(strdict=dictofvelstrs, tmesh=trange,
                               c_mat=b_mat_rgscld.T, load_data=dou.load_npa)
            dou.save_output_json(dict(tmesh=trange.tolist(),
                                      outsig=yscomplist),
                                 fstring=shortstring + '_adj')
            return

        else:
            linsysrhs = soldict['fv'] + rhs_con + rhsv_conbc
            shortstring = shortstring + '_linclsys'
            nseres = -f_mat*v_ss_nse - jmat.T*p_ss_nse - linsysrhs
            print('oseen res: {0}'.format(np.linalg.norm(nseres)))
            soldict.update(stokes_flow=True, A=-f_mat, fv=linsysrhs)

    timediscstr = 't{0}{1}Nts{2}'.format(t0, tE, Nts)

    # ### CHAP: the simulation
    if closed_loop == 'red_output_fb' and not simuN == N:
        simuxtrstr = 'SN{0}'.format(simuN)
        print('Controller with N={0}, Simulation with N={1}'.format(N, simuN))
        sfemp, sstokesmatsc, srhsd \
            = dnsps.get_sysmats(problem=problemname, N=simuN, Re=Re,
                                bccontrol=bccontrol, scheme='TH',
                                mergerhs=True)
        sinvinds, sNV = sfemp['invinds'], sstokesmatsc['A'].shape[0]
        if bccontrol:
            sstokesmatsc['A'] = sstokesmatsc['A'] +\
                1./palpha*sstokesmatsc['Arob']
            sb_mat = 1./palpha*sstokesmatsc['Brob']
            u_masmat = sps.eye(b_mat.shape[1], format='csr')

        else:
            sb_mat, u_masmat = cou.get_inp_opa(cdcoo=sfemp['cdcoo'],
                                               V=sfemp['V'], NU=NU,
                                               xcomp=sfemp['uspacedep'])
            sb_mat = sb_mat[sinvinds, :][:, :]
        sb_mat_scld = sb_mat*Rmhalf

        smc_mat, sy_masmat = cou.get_mout_opa(odcoo=sfemp['odcoo'],
                                              V=sfemp['V'], mfgrid=Cgrid)
        sc_mat = lau.apply_massinv(sy_masmat, smc_mat, output='sparse')
        sc_mat = sc_mat[:, sinvinds][:, :]

        soldict.update(sstokesmatsc)  # containing A, J, JT
        soldict.update(sfemp)  # adding V, Q, invinds, diribcs
        soldict.update(srhsd)  # right hand sides
        soldict.update(dict(cv_mat=sc_mat))  # needed for the output feedback
        fv_rofb_dict.update(dict(b_mat=sb_mat_scld))

        saveloadsimuinivstr = ddir + 'cw{0}Re{1}g{2}d{3}_iniv'.\
            format(sNV, Re, gamma, perturbpara)
        shortinivstr, retnssnse = csh.\
            set_inival(soldict=soldict, whichinival=whichinival, trange=trange,
                       tpp=tpp, perturbpara=perturbpara,
                       fdstr=saveloadsimuinivstr, retvssnse=True)

        fv_rofb_dict.update(dict(ystar=sc_mat.dot(retnssnse)))
        shortstring = (get_fdstr(Re, short=True) + shortclstr +
                       shorttruncstr + shortinivstr + shortfailstr)

    else:
        simuxtrstr = ''
        sc_mat = c_mat
        shortstring = (get_fdstr(Re, short=True) + shortclstr +
                       shorttruncstr + shortinivstr + shortfailstr)

    try:
        yscomplist = dou.load_json_dicts(shortstring + simuxtrstr +
                                         timediscstr)['outsig']
        print('loaded the outputs from: ' + shortstring)

    except IOError:
        soldict.update(data_prfx=shortstring + simuxtrstr)
        dictofvelstrs = snu.solve_nse(**soldict)

        yscomplist = cou.extract_output(strdict=dictofvelstrs, tmesh=trange,
                                        c_mat=sc_mat, load_data=dou.load_npa)

    dou.save_output_json(dict(tmesh=trange.tolist(), outsig=yscomplist),
                         fstring=(shortstring + simuxtrstr + timediscstr))

    if plotit:
        dou.plot_outp_sig(tmesh=trange, outsig=yscomplist)

    ymys = dou.meas_output_diff(tmesh=trange, ylist=yscomplist,
                                ystar=c_mat.dot(v_ss_nse))
    print('|y-y*|: {0}'.format(ymys))

if __name__ == '__main__':
    # lqgbt(N=10, Re=500, use_ric_ini=None, plain_bt=False)
    lqgbt(problemname='cylinderwake', N=3,  # use_ric_ini=2e2,
          Re=7.5e1, plain_bt=False,
          t0=0.0, tE=2.0, Nts=1e3+1, palpha=1e-6,
          comp_freqresp=False, comp_stepresp=False)
