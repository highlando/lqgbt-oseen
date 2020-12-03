# import scipy.sparse as sps
import numpy as np
# import scipy.linalg as spla

from pathlib import Path

import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.stokes_navier_utils as snu
# import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.problem_setups as dnsps

import sadptprj_riclyap_adi.lin_alg_utils as lau
# import sadptprj_riclyap_adi.proj_ric_utils as pru

import distributed_control_fenics.cont_obs_utils as cou

import lqgbt_oseen.nse_riccont_utils as nru
import lqgbt_oseen.cntrl_simu_helpers as csh

debug = False

checktheres = True  # whether to check the Riccati Residuals
checktheres = False

switchonsfb = 0  # 1.5


def _get_inputd(ta=None, tb=None, uvec=None, ampltd=1., **kwargs):

    intvl = tb - ta

    def _inputd(t):
        if t < ta or t > tb:
            return 0*uvec
        else:
            s = (t - ta)/intvl
            du = np.sin(s*2*np.pi)
            return ampltd*du*uvec
    return _inputd


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


def lqgbt(Re=1e2,
          problemname='cylinderwake', shortname='cw',
          meshparams=None,
          gamma=1.,
          use_ric_ini=None, t0=0.0, tE=1.0, Nts=11,
          NU=3, Cgrid=(3, 1),
          bccontrol=True, palpha=1e-5,
          npcrdstps=15,
          pymess=False,
          paraoutput=True,
          plotit=True,
          ddir='data/',
          trunc_lqgbtcv=1e-6,
          hinf=False,
          nwtn_adi_dict=None,
          pymess_dict=None,
          whichinival='sstate',
          dudict=dict(addinputd=False),
          tpp=5.,  # time to add on Stokes inival for `sstokes++`
          comp_freqresp=False, comp_stepresp='nonlinear',
          closed_loop=False, multiproc=False,
          perturbpara=1e-3,
          strtogramfacs=None,
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

    print('\n ### We gonna regulate the {0} at Re={1} ###\n'.
          format(problemname, Re))
    print(' ### The control is weighted with Gamma={0}'.format(gamma))

    femp, stokesmatsc, rhsd = \
        dnsps.get_sysmats(problem='gen_bccont', Re=Re, bccontrol=True,
                          scheme='TH', mergerhs=True,
                          meshparams=meshparams)

    # casting some parameters
    invinds, NV = femp['invinds'], len(femp['invinds'])

#
# Prepare for control
#
    prbstr = '_pymess' if pymess else ''

    contsetupstr = 'NV{0}_B{3}_C{1[0]}{1[1]}_palpha{2}'.\
        format(NV, Cgrid, palpha, NU)
    shortcontsetupstr = '{0}{1[0]}{1[1]}{2}'.\
        format(NV, Cgrid, np.int(np.log2(palpha)))

    if bccontrol:
        stokesmatsc['A'] = stokesmatsc['A'] + 1./palpha*stokesmatsc['Arob']
        b_mat = 1./palpha*stokesmatsc['Brob']
        # u_masmat = sps.eye(b_mat.shape[1], format='csr')
        print(' ### Robin-type boundary control palpha={0}'.format(palpha))

    if whichinival == 'sstokes++' or whichinival == 'snse+d++':
        inivstr = '_' + whichinival + '{0}'.format(tpp)
    else:
        inivstr = '_' + whichinival

    print(inivstr)

    def get_fdstr(Re, short=False):
        if short:
            return ddir + shortname + '{0}{1}_'.format(Re, gamma) + \
                shortcontsetupstr
        return ddir + problemname + '_Re{0}_gamma{1}_'.format(Re, gamma) + \
            contsetupstr + prbstr

    fdstr = get_fdstr(Re)
    # fdstr = fdstr + '_hinf' if hinf else fdstr
    fdstrini = get_fdstr(use_ric_ini) if use_ric_ini is not None else None

#
# ### CHAP: Prepare for control
#

    b_mat_reg = lau.app_prj_via_sadpnt(amat=stokesmatsc['M'],
                                       jmat=stokesmatsc['J'],
                                       rhsv=b_mat,
                                       transposedprj=True)
    Rmhalf = 1./np.sqrt(gamma)
    b_mat = Rmhalf*b_mat_reg
    # We scale the input matrix to acommodate for input weighting
    # TODO: we should consider the u mass matrix here
    # TODO: this regularization shouldn't be necessary

    print('computing `c_mat`...')
    mc_mat, y_masmat = cou.get_mout_opa(odcoo=femp['odcoo'],
                                        V=femp['V'], mfgrid=Cgrid)
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
    soldict.update(femp)  # adding V, Q, invinds, dbcinds, dbcvals
    # soldict.update(rhsd_vfrc)  # adding fvc, fpr
    veldatastr = ddir + problemname + '_Re{0}'.format(Re)
    if bccontrol:
        veldatastr = veldatastr + '__bcc_palpha{0}'.format(palpha)

    nu = femp['charlen']/Re
    soldict.update(fv=rhsd['fv'], fp=rhsd['fp'],
                   nu=nu, data_prfx=veldatastr)

    v_init = None
    initssres = [40, 60, 80]
    for initre in initssres:
        if initre >= Re:
            initre = Re
        else:
            print('Initialising the steadystate solution with Re=', initre)
        cachedsss = 'cachedata/' + Path(get_fdstr(initre)).name + '_sssol.npy'
        try:
            vp_ss_nse = (np.load(cachedsss),
                         None)
            print('loaded sssol from: ', cachedsss)
        except IOError:
            print("couldn't load sssol from: ", cachedsss)
            initssfemp, initssstokesmatsc, initssrhsd = \
                dnsps.get_sysmats(problem='gen_bccont', Re=initre,
                                  bccontrol=True, scheme='TH', mergerhs=True,
                                  meshparams=meshparams)
            initssstokesmatsc['A'] = initssstokesmatsc['A'] \
                + 1./palpha*initssstokesmatsc['Arob']
            initsssoldict = {}
            initsssoldict.update(initssstokesmatsc)
            initsssoldict.update(initssfemp)
            initssveldatastr = ddir + problemname + '_Re{0}'.format(initre)
            initssnu = femp['charlen']/initre
            initsssoldict.update(fv=initssrhsd['fv'], fp=initssrhsd['fp'],
                                 nu=initssnu, data_prfx=initssveldatastr)
            vp_ss_nse = snu.\
                solve_steadystate_nse(vel_pcrd_stps=npcrdstps, return_vp=True,
                                      vel_start_nwtn=v_init,
                                      vel_nwtn_tol=4e-13,
                                      clearprvdata=debug, **initsssoldict)
            np.save(cachedsss, vp_ss_nse[0])
            print('saved sssol to: ', cachedsss)
        if initre == Re:
            break
        v_init = vp_ss_nse[0]

    v_ss_nse = vp_ss_nse[0]
    dbcinds, dbcvals = femp['dbcinds'], femp['dbcvals']
    (convc_mat, rhs_con,
     rhsv_conbc) = snu.get_v_conv_conts(vvec=v_ss_nse, invinds=invinds,
                                        dbcinds=dbcinds, dbcvals=dbcvals,
                                        V=femp['V'])

    f_mat = - stokesmatsc['A'] - convc_mat
    # the robin term `arob` has been added before
    mmat = stokesmatsc['M']
    # amat = stokesmatsc['A']
    jmat = stokesmatsc['J']

    # MAF -- need to change the convc_mat, i.e. we need another v_ss_nse
    # MAF -- need to change the f_mat, i.e. we need another convc_mat
    if trytofail:
        v_ss_nse_MAF = snu.\
            solve_steadystate_nse(vel_pcrd_stps=ttf_npcrdstps, vel_nwtn_stps=0,
                                  vel_pcrd_tol=1e-15,
                                  vel_start_nwtn=v_init,
                                  clearprvdata=True, **soldict)
        diffv = (v_ss_nse - v_ss_nse_MAF)[invinds]
        convc_mat_MAF, _, _ = \
            snu.get_v_conv_conts(vvec=v_ss_nse_MAF, invinds=invinds,
                                 V=femp['V'], dbcinds=dbcinds, dbcvals=dbcvals)
        nrmvsqrd = np.dot(v_ss_nse[invinds].T, mmat*v_ss_nse[invinds])
        relnormdiffv = np.sqrt(np.dot(diffv.T, mmat*diffv)/nrmvsqrd)
        print('relative difference to linearization: {0}'.
              format(relnormdiffv))
        f_mat_gramians = -stokesmatsc['A'] - convc_mat_MAF
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

# compute the regulated system
    trange = np.linspace(t0, tE, Nts)

    if dudict['addinputd']:
        print('u disturbed in [{0}, {1}]'.format(dudict['ta'], dudict['tb']) +
              ' to trigger instabilities')
        print('ampltd used: {0}'.format(dudict['ampltd']))
        inputd = _get_inputd(**dudict)

    if closed_loop is not None:
        zwconly = (closed_loop == 'full_state_fb')
        comploadricfacsdct = dict(fdstr=fdstr, fmat=f_mat_gramians,
                                  mmat=mmat, jmat=jmat, bmat=b_mat,
                                  cmat=c_mat_reg,
                                  ric_ini_str=fdstrini,
                                  nwtn_adi_dict=nwtn_adi_dict,
                                  zwconly=zwconly, hinf=hinf,
                                  multiproc=multiproc, pymess=pymess,
                                  checktheres=False,
                                  strtogramfacs=strtogramfacs)
        zwc, zwo, hinfgamma = nru.get_ric_facs(**comploadricfacsdct)

        if closed_loop == 'red_output_fb':
            import sadptprj_riclyap_adi.bal_trunc_utils as btu
            tl, tr, _ = btu.\
                compute_lrbt_transfos(zfc=zwc, zfo=zwo, mmat=mmat,
                                      trunck={'threshh': trunc_lqgbtcv})

    if closed_loop is False:
        return  # we only want the Gramians

    elif closed_loop == 'full_state_fb':
        shortclstr = 'fsfb'

        # mtxb = pru.get_mTzzTtb(stokesmatsc['M'].T, zwc, b_mat)

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

        # tmdp_fsfb_dict = dict(linv=v_ss_nse, tb_mat=b_mat,
        #                       btxm_mat=mtxb.T)

        # fv_tmdp = fv_tmdp_fullstatefb
        # fv_tmdp_params = tmdp_fsfb_dict
        # fv_tmdp_memory = None

    # ### CHAP: define the reduced output feedback
    elif closed_loop == 'red_output_fb':
        shortclstr = 'hinfrofb' if hinf else 'rofb'

        ak_mat, bk_mat, ck_mat, xok, xck = nru.\
            get_prj_model(mmat=mmat, fmat=f_mat_gramians, jmat=jmat,
                          zwo=zwo, zwc=zwc,
                          tl=tl, tr=tr,
                          bmat=b_mat, cmat=c_mat_reg)
        print('Controller has dimension: {0}'.format(ak_mat.shape[0]))

        if hinf:
            print('hinf red fb: gamma={0}'.format(hinfgamma))
            # scfc = np.sqrt(1-1/hinfgamma**2)  # [2]
            # print('recomputing the reduced obs riccati solution...')
            # rsxok = spla.solve_continuous_are(ak_mat.T, scfc*ck_mat.T,
            #                                   bk_mat.dot(bk_mat.T),
            #                                   np.eye(ck_mat.shape[0]))
            # xok = rsxok
            # print('xok: ', np.diag(xok))
            # print('xck: ', np.diag(xck))
            zk = np.linalg.inv(np.eye(xck.shape[0])
                               - 1./hinfgamma**2*xok.dot(xck))
            zkdi = np.diag(1./(1 - 1./hinfgamma**2*np.diag(xok)*np.diag(xck)))
            # print('zk: ', np.diag(zk))
            print(np.linalg.norm(zk-zkdi))
            zk = zkdi
            amatk = (ak_mat
                     - (1. - 1./hinfgamma**2)*np.dot(np.dot(xok, ck_mat.T),
                                                     ck_mat)
                     - np.dot(bk_mat, np.dot(bk_mat.T, xck).dot(zk)))
            obs_ck = -np.dot(bk_mat.T.dot(xck), zk)
            obs_bk = xok @ ck_mat.T
            fullrmmat = np.vstack([np.hstack([amatk, obs_bk@ck_mat]),
                                   np.hstack([bk_mat@obs_ck, ak_mat])])
            evls = np.linalg.eigvals(fullrmmat)
            print(np.linalg.norm(obs_ck), np.linalg.norm(obs_bk))
            print(evls)

        else:
            print('lqg-feedback!!')
            # if pymess:
            #     scfc = 1.
            #     print('recomputing the reduced gramians!!')
            #     rsxok = spla.solve_continuous_are(ak_mat.T, scfc*ck_mat.T,
            #                                       bk_mat.dot(bk_mat.T),
            #                                       np.eye(ck_mat.shape[0]))
            #     rsxck = spla.solve_continuous_are(ak_mat, scfc*bk_mat,
            #                                       ck_mat.T.dot(ck_mat),
            #                                       np.eye(bk_mat.T.shape[0]))
            #     xok, xck = rsxok, rsxck
            amatk = (ak_mat - np.dot(np.dot(xok, ck_mat.T), ck_mat) -
                     np.dot(bk_mat, np.dot(bk_mat.T, xck)))
            obs_ck = -bk_mat.T.dot(xck)
            obs_bk = np.dot(xok, ck_mat.T)

        hbystar = obs_bk.dot(c_mat.dot(v_ss_nse[invinds]))

        def obsdrft(t):
            return -hbystar

        linobsrvdct = dict(ha=amatk, hc=obs_ck, hb=obs_bk,
                           drift=obsdrft, inihx=np.zeros((obs_bk.shape[0], 1)))
        soldict.update(dynamic_feedback=True, dyn_fb_dict=linobsrvdct)
        soldict.update(dict(closed_loop=True))
    else:
        shortclstr = '_'

    if dudict['addinputd']:
        def fvtd(t):
            return b_mat.dot(inputd(t))
    else:
        fvtd = None

    shortclstr = shortclstr + 'pm' if pymess else shortclstr

    soldict.update(trange=trange,
                   lin_vel_point=None,
                   clearprvdata=True,
                   fvtd=fvtd,
                   cv_mat=c_mat,  # needed for the output feedback
                   treat_nonl_explct=True,
                   b_mat=b_mat,
                   return_y_list=True,
                   return_dictofvelstrs=False)

    # ### CHAP: define the initial values
    # if simuN == N:
    shortinivstr, _ = csh.\
        set_inival(soldict=soldict, whichinival=whichinival, trange=trange,
                   tpp=tpp, v_ss_nse=v_ss_nse, perturbpara=perturbpara,
                   fdstr=fdstr)
    shortstring = (get_fdstr(Re, short=True) + shortclstr +
                   shorttruncstr + shortinivstr + shortfailstr)

    outstr = truncstr + '{0}'.format(closed_loop) \
        + 't0{0}tE{1}Nts{2}N{3}Re{4}'.format(t0, tE, Nts, NV, Re)
    if paraoutput:
        soldict.update(paraviewoutput=True,
                       vfileprfx='results/vel_'+outstr,
                       pfileprfx='results/p_'+outstr)

    timediscstr = 't{0}{1:.4f}Nts{2}'.format(t0, tE, Nts)
    if dudict['addinputd']:
        inputdstr = 'ab{0}{1}A{2}'.format(dudict['ta'], dudict['tb'],
                                          dudict['ampltd'])
    else:
        inputdstr = ''

    # ### CHAP: the simulation
    # if closed_loop == 'red_output_fb' and not simuN == N:
    #     simuxtrstr = 'SN{0}'.format(simuN)
    #     print('Controller with N={0}, Simulation w\ N={1}'.format(N, simuN))
    #     sfemp, sstokesmatsc, srhsd \
    #         = dnsps.get_sysmats(problem=problemname, N=simuN, Re=Re,
    #                             bccontrol=bccontrol, scheme='TH',
    #                             mergerhs=True)
    #     sinvinds, sNV = sfemp['invinds'], sstokesmatsc['A'].shape[0]
    #     if bccontrol:
    #         sstokesmatsc['A'] = sstokesmatsc['A'] +\
    #             1./palpha*sstokesmatsc['Arob']
    #         sb_mat = 1./palpha*sstokesmatsc['Brob']
    #         u_masmat = sps.eye(b_mat.shape[1], format='csr')

    #     else:
    #         sb_mat, u_masmat = cou.get_inp_opa(cdcoo=sfemp['cdcoo'],
    #                                            V=sfemp['V'], NU=NU,
    #                                            xcomp=sfemp['uspacedep'])
    #         sb_mat = sb_mat[sinvinds, :][:, :]

    #     smc_mat, sy_masmat = cou.get_mout_opa(odcoo=sfemp['odcoo'],
    #                                           V=sfemp['V'], mfgrid=Cgrid)
    #     sc_mat = lau.apply_massinv(sy_masmat, smc_mat, output='sparse')
    #     sc_mat = sc_mat[:, sinvinds][:, :]

    #     soldict.update(sstokesmatsc)  # containing A, J, JT
    #     soldict.update(sfemp)  # adding V, Q, invinds, diribcs
    #     soldict.update(srhsd)  # right hand sides
    #     soldict.update(dict(cv_mat=sc_mat))  # needed for the output feedback

    #     saveloadsimuinivstr = ddir + 'cw{0}Re{1}g{2}d{3}_iniv'.\
    #         format(sNV, Re, gamma, perturbpara)
    #     shortinivstr, retnssnse = csh.\
    #         set_inival(soldict=soldict, whichinival=whichinival,
    #                    trange=trange, tpp=tpp, perturbpara=perturbpara,
    #                    fdstr=saveloadsimuinivstr, retvssnse=True)

    #     shortstring = (get_fdstr(Re, short=True) + shortclstr +
    #                    shorttruncstr + shortinivstr + shortfailstr +
    #                    inputdstr)

    # else:
    simuxtrstr = ''
    # sc_mat = c_mat
    shortstring = (get_fdstr(Re, short=True) + shortclstr +
                   shorttruncstr + shortinivstr + shortfailstr)

    ystr = shortstring + simuxtrstr + timediscstr + inputdstr

    try:
        raise IOError()
        yscomplist = dou.load_json_dicts(ystr)['outsig']
        print('loaded the outputs from: ' + shortstring)

    except IOError:
        soldict.update(data_prfx=shortstring + simuxtrstr)
        # dictofvelstrs = snu.solve_nse(**soldict)
        yscomplist = snu.solve_nse(**soldict)
        yscomplist = [ykk.flatten().tolist() for ykk in yscomplist]

        # yscomplist = cou.extract_output(strdict=dictofvelstrs, tmesh=trange,
        #                                 invinds=invinds,
        #                                 c_mat=sc_mat, load_data=dou.load_npa)

    dou.save_output_json(dict(tmesh=trange.tolist(), outsig=yscomplist),
                         fstring=(ystr))

    if plotit:
        dou.plot_outp_sig(tmesh=trange, outsig=yscomplist)

    ymys = dou.meas_output_diff(tmesh=trange, ylist=yscomplist,
                                ystar=c_mat.dot(v_ss_nse[femp['invinds']]))
    print('|y-y*|: {0}'.format(ymys))


if __name__ == '__main__':
    # lqgbt(N=10, Re=500, use_ric_ini=None, plain_bt=False)
    lqgbt(problemname='cylinderwake', N=2,  # use_ric_ini=2e2,
          Re=7.5e1, plain_bt=False,
          t0=0.0, tE=2.0, Nts=1e3+1, palpha=1e-6,
          comp_freqresp=False, comp_stepresp=False)
