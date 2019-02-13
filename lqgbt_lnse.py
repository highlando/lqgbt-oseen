import os
import numpy as np

# import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

import sadptprj_riclyap_adi.lin_alg_utils as lau
import sadptprj_riclyap_adi.proj_ric_utils as pru

import distr_control_fenics.cont_obs_utils as cou

import nse_riccont_utils as nru

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
          N=10, Re=1e2, plain_bt=False, cl_linsys=False,
          gamma=1.,
          use_ric_ini=None, t0=0.0, tE=1.0, Nts=11,
          NU=3, NY=3,
          bccontrol=True, palpha=1e-5,
          npcrdstps=8,
          pymess=False,
          paraoutput=True,
          plotit=True,
          trunc_lqgbtcv=1e-6,
          hinf=False,
          whichinival='sstate',
          tpp=5.,  # time to add on Stokes inival for `sstokes++`
          closed_loop=False, multiproc=False,
          perturbpara=1e-3,
          trytofail=False, ttf_npcrdstps=3,
          robit=False, robmrgnfac=0.5):
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
    NU, NY : int, optional
        dimensions of components of in and output space (will double because
        there are two components), default to `3, 3`
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
    # contsetupstr = 'NV{0}NU{1}NY{2}alphau{3}'.format(NV, NU, NY, alphau)
    if bccontrol:
        contsetupstr = 'NV{0}_bcc_NY{1}_palpha{2}'.format(NV, NY, palpha)
        shortcontsetupstr = '{0}{1}{2}'.format(NV, NY, np.int(np.log2(palpha)))
        stokesmatsc['A'] = stokesmatsc['A'] + 1./palpha*stokesmatsc['Arob']
        b_mat = 1./palpha*stokesmatsc['Brob']
        print(' ### Robin-type boundary control palpha={0}'.format(palpha))
    else:
        contsetupstr = 'NV{0}NU{1}NY{2}'.format(NV, NU, NY)
        shortcontsetupstr = '{0}{1}{2}'.format(NV, NU, NY)

    # inivstr = '_' + whichinival if not whichinival == 'sstokes++' \
    #     else '_sstokes++{0}'.format(tpp)

    def get_fdstr(Re, short=False):
        if short:
            return ddir + 'cw' + '{0}{1}_'.format(Re, gamma) + \
                shortcontsetupstr
        return ddir + problemname + '_Re{0}_gamma{1}_'.format(Re, gamma) + \
            contsetupstr + prbstr

    fdstr = get_fdstr(Re)
    fdstr = fdstr + '_hinf' if hinf else fdstr

#
# Prepare for control
#

    b_mat_reg = lau.app_prj_via_sadpnt(amat=stokesmatsc['M'],
                                       jmat=stokesmatsc['J'],
                                       rhsv=b_mat,
                                       transposedprj=True)

    # Rmo = 1./gamma
    Rmhalf = 1./np.sqrt(gamma)
    b_mat_rgscld = b_mat_reg*Rmhalf
    # We scale the input matrix to accomodate for input weighting

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

    v_ss_nse, _ = snu.\
        solve_steadystate_nse(vel_pcrd_stps=npcrdstps, return_vp=True,
                              clearprvdata=debug, **soldict)

    (convc_mat, rhs_con,
     rhsv_conbc) = snu.get_v_conv_conts(prev_v=v_ss_nse, invinds=invinds,
                                        V=femp['V'], diribcs=femp['diribcs'])

    f_mat = - stokesmatsc['A'] - convc_mat
    # the robin term `arob` has been added before
    mmat = stokesmatsc['M']
    jmat = stokesmatsc['J']

    # MAF -- need to change the convc_mat, i.e. we need another v_ss_nse
    # MAF -- need to change the f_mat, i.e. we need another convc_mat
    if trytofail:
        v_ss_nse_MAF = snu.solve_steadystate_nse(vel_pcrd_stps=ttf_npcrdstps,
                                                 vel_nwtn_stps=0,
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

    cmprlprjpars = dict(trunc_lqgbtcv=trunc_lqgbtcv)

# compute the regulated system
    trange = np.linspace(t0, tE, Nts)
    DT = (tE - t0)/(Nts-1)
    loadhinfmatstr = 'oc-hinf-recover/output/' + \
        fdstr.partition('/')[2] + '__mats'
    loadmatmatstr = 'oc-hinf-recover/cylinderwake_Re{0}_gamma1.0_'.format(Re) +\
        'NV{0}_bcc_NY3_palpha1e-05_lqgbt_hinf_MAF_'.format(NV) +\
        'ttfnpcrds{0}__mats'.format(ttf_npcrdstps)
    loadhinfmatstr = 'oc-hinf-recover/output/' +\
        'cylinderwake_Re{0}_gamma1.0_'.format(Re) +\
        'NV{0}_bcc_NY3_palpha1e-05_lqgbt_hinf_MAF_'.format(NV) +\
        'ttfnpcrds{0}__mats_output'.format(ttf_npcrdstps)
    from scipy.io import loadmat
    mmd = {}
    loadmat(loadmatmatstr, mdict=mmd)
    c_mat_reg = mmd['cmat']
    lmd = {}
    loadmat(loadhinfmatstr, mdict=lmd)
    # try:
    #     zwchinf, zwohinf, hinfgamma = lmd['ZB'], lmd['ZC'], lmd['gam_opt']
    # except KeyError:
    zwchinf, zwohinf, hinfgamma = (lmd['outControl'][0, 0]['Z'],
                                   lmd['outFilter'][0, 0]['Z'],
                                   lmd['gam_opt'])
    zwclqg, zwolqg = (lmd['outControl'][0, 0]['Z_LQG'],
                      lmd['outFilter'][0, 0]['Z_LQG'])
    if hinf:
        print('we use the hinf-Riccatis, gamma={0}'.format(hinfgamma))
        zwc, zwo = zwchinf, zwohinf
    else:
        zwc, zwo = zwclqg, zwolqg
        print('we use the lqg-Riccatis')

    if closed_loop is False:
        return

    elif closed_loop == 'red_output_fb':
        shortclstr = 'hinfrofb' if hinf else 'rofb'
        DT = (tE - t0)/(Nts-1)

        ak_mat, bk_mat, ck_mat, xok, xck = \
            nru.get_prj_model(mmat=mmat, fmat=f_mat_gramians, jmat=jmat,
                              zwo=zwo, zwc=zwc,
                              bmat=b_mat_rgscld, cmat=c_mat_reg,
                              cmprlprjpars=cmprlprjpars)
        print('Controller has dimension: {0}'.format(ak_mat.shape[0]))

        if hinf:
            print('hinf-feedback!!')
            zk = np.linalg.inv(np.eye(xck.shape[0])
                               - 1./hinfgamma**2*xok.dot(xck))
            amatk = (ak_mat
                     - (1. - 1./hinfgamma**2)*np.dot(np.dot(xok, ck_mat.T),
                                                     ck_mat)
                     - np.dot(bk_mat, np.dot(bk_mat.T, xck).dot(zk)))
            obs_ck = -np.dot(bk_mat.T.dot(xck), zk)

        else:
            print('lqg-feedback!!')
            amatk = (ak_mat - np.dot(np.dot(xok, ck_mat.T), ck_mat) -
                     np.dot(bk_mat, np.dot(bk_mat.T, xck)))
            obs_ck = -bk_mat.T.dot(xck)

        obs_bk = np.dot(xok, ck_mat.T)
        sysmatk_inv = np.linalg.inv(np.eye(ak_mat.shape[1]) - DT*amatk)

        def fv_tmdp_redoutpfb(time=None, curvel=None, memory=None,
                              linvel=None,
                              b_mat=None, c_mat=None,
                              ipsysk_mat_inv=None, cts=None,
                              obs_bk=None, obs_ck=None,
                              # xck=None, bk_mat=None,
                              **kw):
            """realizes a reduced static output feedback as a function

            that can be passed to a solution routine for the
            unsteady Navier-Stokes equations

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
            buk = cts*np.dot(obs_bk,
                             lau.mm_dnssps(c_mat, (curvel-linvel)))
            xk_old = np.dot(ipsysk_mat_inv, xk_old + buk)
            memory['xk_old'] = xk_old
            actua = lau.mm_dnssps(b_mat, obs_ck.dot(xk_old))
            # if np.mod(np.int(time/DT), np.int(tE/DT)/100) == 0:
            #     print(('time now: {0}, end time: {1}'.format(time, tE)))
            #     print('\nnorm of deviation', np.linalg.norm(curvel-linvel))
            #     print('norm of actuation {0}'.format(np.linalg.norm(actua)))
            memory['actualist'].append(actua)

            return actua, memory

        fv_rofb_dict = dict(cts=DT, linvel=v_ss_nse,
                            b_mat=b_mat_rgscld, c_mat=c_mat_reg,
                            obs_bk=obs_bk, obs_ck=obs_ck,
                            ipsysk_mat_inv=sysmatk_inv)

        fv_tmdp = fv_tmdp_redoutpfb
        fv_tmdp_params = fv_rofb_dict
        fv_tmdp_memory = dict(xk_old=np.zeros((amatk.shape[1], 1)),
                              actualist=[])

        # soldict.update(dict(verbose=False))

    elif closed_loop == 'full_state_fb':
        shortclstr = 'fsfb'

        mtxb = pru.get_mTzzTtb(stokesmatsc['M'].T, zwc, b_mat_rgscld)

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

        tmdp_fsfb_dict = dict(linv=v_ss_nse, tb_mat=b_mat_rgscld,
                              btxm_mat=mtxb.T)

        fv_tmdp = fv_tmdp_fullstatefb
        fv_tmdp_params = tmdp_fsfb_dict
        fv_tmdp_memory = None

    else:
        fv_tmdp = None
        fv_tmdp_params = {}
        fv_tmdp_memory = {}
        shortclstr = '_'

    shortclstr = shortclstr + 'pm' if pymess else shortclstr

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
        shortinivstr = 'ssd{0}'.format(perturbpara)
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

    # checkdaredmod = True
    # checkdaredmod = False
    # if checkdaredmod:
    #     import spacetime_galerkin_pod.gen_pod_utils as gpu
    #     # akm = basetl.T.dot(amat*basetr)
    #     # nk = basetl.shape[1]

    #     redmod = False
    #     redmod = True
    #     if redmod:
    #         curiniv = basetl.T.dot(mmat*(soldict['iniv']-vinf))
    #         nk = basetr.shape[1]

    #         def rednonl(vvec, t):
    #             inflv = basetr.dot(vvec.reshape((nk, 1)))
    #             curcoeff = get_cur_sdccoeff(vdelta=inflv)
    #             returval = basetl.T.dot(curcoeff.dot(inflv))
    #             return returval.flatten()
    #         curnonl = rednonl
    #         tstrunstr = 'testdaredmod'
    #         mmatforlsoda = None
    #         tstc = ck_mat

    #     else:
    #         curiniv = soldict['iniv'] - vinf
    #         NV = mmat.shape[0]

    #         def fulnonl(vvec, t):
    #             curcoeff = get_cur_sdccoeff(vvec.reshape((NV, 1)))
    #             apconv = curcoeff.dot(vvec.reshape((NV, 1)))
    #             prjapc = lau.app_prj_via_sadpnt(amat=mmat, jmat=jmat,
    #                                             rhsv=apconv)
    #             return prjapc.flatten()
    #         mmatforlsoda = mmat
    #         curnonl = fulnonl
    #         tstrunstr = 'testdafulmod'
    #         tstc = c_mat

    #     print('doing the `lsoda` integration...')
    #     tstsol = gpu.time_int_semil(tmesh=trange, A=None, M=mmatforlsoda,
    #                                 nfunc=curnonl, iniv=curiniv)

    #     print('done with the `lsoda` integration!')
    #     outptlst = []
    #     for kline in range(tstsol.shape[0]):
    #         # outptlst.append((ck_mat.dot(redsol[k, :])).tolist())
    #         outptlst.append((tstc.dot(tstsol[kline, :])).tolist())
    #     dou.save_output_json(dict(tmesh=trange.tolist(), outsig=outptlst),
    #                          fstring=tstrunstr)
    #     import ipdb; ipdb.set_trace()

    outstr = truncstr + '{0}'.format(closed_loop) \
        + 't0{0}tE{1}Nts{2}N{3}Re{4}'.format(t0, tE, Nts, N, Re)
    if paraoutput:
        soldict.update(paraviewoutput=True,
                       vfileprfx='results/vel_'+outstr,
                       pfileprfx='results/p_'+outstr)

    shortstring = (get_fdstr(Re, short=True) +  # shortcontsetupstr +
                   shortclstr + shorttruncstr + shortinivstr + shortfailstr)

    soldict.update(data_prfx=shortstring)
    dictofvelstrs = snu.solve_nse(**soldict)

    yscomplist = cou.extract_output(strdict=dictofvelstrs, tmesh=trange,
                                    c_mat=c_mat_reg, load_data=dou.load_npa)

    if robit:
        robitstr = '_robmgnfac{0}'.format(robmrgnfac)
    else:
        robitstr = ''

    dou.save_output_json(dict(tmesh=trange.tolist(), outsig=yscomplist),
                         fstring=shortstring + robitstr)

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
