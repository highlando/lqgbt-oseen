import dolfin
# import numpy as np
# import scipy.sparse as sps
# import matplotlib.pyplot as plt
import os

import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

import sadptprj_riclyap_adi.lin_alg_utils as lau
import sadptprj_riclyap_adi.proj_ric_utils as pru
import sadptprj_riclyap_adi.bal_trunc_utils as btu

import distr_control_fenics.cont_obs_utils as cou

dolfin.parameters.linear_algebra_backend = 'uBLAS'


def time_int_params(Nts, nu):
    t0 = 0.0
    tE = 1.0
    dt = (tE - t0) / Nts
    tip = dict(t0=t0,
               tE=tE,
               dt=dt,
               Nts=Nts,
               vfile=None,
               pfile=None,
               Residuals=[],
               ParaviewOutput=True,
               proutdir='results/',
               prfprfx='',
               nu=nu,
               nnewtsteps=9,  # n nwtn stps for vel comp
               vel_nwtn_tol=1e-14,
               norm_nwtnupd_list=[],
               # parameters for newton adi iteration
               nwtn_adi_dict=dict(
                   adi_max_steps=250,
                   adi_newZ_reltol=1e-5,
                   nwtn_max_steps=9,
                   nwtn_upd_reltol=4e-8,
                   nwtn_upd_abstol=1e-7,
                   verbose=True,
                   full_upd_norm_check=False,
                   check_lyap_res=False
               ),
               compress_z=True,  # whether or not to compress Z
               comprz_maxc=500,  # compression of the columns of Z by QR
               comprz_thresh=5e-5,  # threshold for trunc of SVD
               save_full_z=False,  # whether or not to save the uncompressed Z
               )

    return tip


def lqgbt(problemname='drivencavity',
          N=10, Nts=10, Re=1e2, plain_bt=True,
          savetomatfiles=False):

    problemdict = dict(drivencavity=dnsps.drivcav_fems,
                       cylinderwake=dnsps.cyl_fems)

    typprb = 'BT' if plain_bt else 'LQG-BT'

    print '\n ### We solve the {0} problem for the {1} at Re={2} ###\n'.\
        format(typprb, problemname, Re)

    problemfem = problemdict[problemname]
    femp = problemfem(N)

    data_prfx = problemname + '__'
    NU, NY = 3, 4

    # specify in what spatial direction Bu changes. The remaining is constant
    if problemname == 'drivencavity':
        charlen = 1.0
        uspacedep = 0
    elif problemname == 'cylinderwake':
        charlen = 0.15
        uspacedep = 1
    nu = charlen/Re

    tip = time_int_params(Nts, nu)
    # output
    ddir = 'data/'
    try:
        os.chdir(ddir)
    except OSError:
        raise Warning('need "' + ddir + '" subdir for storing the data')
    os.chdir('..')

    stokesmats = dts.get_stokessysmats(femp['V'], femp['Q'],
                                       tip['nu'])

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
    NV, DT, INVINDS = len(femp['invinds']), tip['dt'], femp['invinds']

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(rhsd_vfrc)  # adding fvc, fpr
    soldict.update(fv_stbc=rhsd_stbc['fv'], fp_stbc=rhsd_stbc['fp'],
                   N=N, nu=tip['nu'],
                   nnewtsteps=tip['nnewtsteps'],
                   vel_nwtn_tol=tip['vel_nwtn_tol'],
                   ddir=ddir, get_datastring=None,
                   data_prfx=data_prfx,
                   paraviewoutput=tip['ParaviewOutput'],
                   vfileprfx=tip['proutdir']+'vel_',
                   pfileprfx=tip['proutdir']+'p_')

#
# compute the uncontrolled steady state Stokes solution
#
    v_ss_nse, list_norm_nwtnupd = snu.solve_steadystate_nse(**soldict)

#
# Prepare for control
#

    contsetupstr = problemname + '__NV{0}NU{1}NY{2}'.format(NV, NU, NY)

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

    c_mat = lau.apply_massinv(y_masmat, mc_mat, output='sparse')

    # TODO: right choice of norms for y
    #       and necessity of regularization here
    #       by now, we go on number save
#
# setup the system for the correction
#
    (convc_mat, rhs_con,
     rhsv_conbc) = snu.get_v_conv_conts(prev_v=v_ss_nse, invinds=invinds,
                                        V=femp['V'], diribcs=femp['diribcs'])

    f_mat = - stokesmatsc['A'] - convc_mat

    def get_fdstr(Re):
        cdatstr = snu.get_datastr_snu(time=None, meshp=N, nu=charlen/Re)
        return ddir + cdatstr + contsetupstr

    fdstr = get_fdstr(Re)
    if plain_bt:
        data_zwc = fdstr + '__bt_zwc'
        data_zwo = fdstr + '__bt_zwo'
        get_gramians = pru.solve_proj_lyap_stein
        data_tl = fdstr + '__bt_tl'
        data_tr = fdstr + '__bt_tr'
    else:
        data_zwc = fdstr + '__lqgbt_zwc'
        data_zwo = fdstr + '__lqgbt_zwo'
        get_gramians = pru.proj_alg_ric_newtonadi
        data_tl = fdstr + '__lqgbt_tl'
        data_tr = fdstr + '__lqgbt_tr'

    try:
        zwc = dou.load_npa(data_zwc)
        zwo = dou.load_npa(data_zwo)
        print 'loaded the factors of ' + \
              'observability and controllability Gramians'

    except IOError:
        if not plain_bt:
            zinic, zinio = None, None
            for Re in use_ric_ini:
                try:
                    zinic = dou.load_npa(get_fdstr(Re) + '__Z')
                    print 'Initialize Newton ADI by Z from ' + cdatstr
                except IOError:
                    raise Warning('No data for initialization of '
                                  ' Newton ADI -- need ' + cdatstr + '__Z')
                cdatstr = get_datastr(meshp=N, nu=nu, data_prfx=data_prfx)
            else:

            Z = pru.proj_alg_ric_newtonadi(mmat=M, amat=-A-convc_mat,
                                           jmat=stokesmatsc['J'],
                                           bmat=tb_mat, wmat=trct_mat,
                                           nwtn_adi_dict=
                                           tip['nwtn_adi_dict'],
                                           z0=zini)['zfac']
            dou.save_npa(Z, fstring=ddir + cdatstr + cntpstr + '__Z')
            print 'saved ' + ddir + cdatstr + cntpstr + '__Z'
        c_mat = lau.apply_massinv(y_masmat, mc_mat, output='sparse')
        c_mat_reg = lau.app_prj_via_sadpnt(amat=stokesmatsc['M'],
                                           jmat=stokesmatsc['J'],
                                           rhsv=c_mat.T,
                                           transposedprj=True).T

        # solve for the contr gramian: A*Wc*M.T + M*Wc*A.T - ... = -B*B.T
        print 'computing the factors of the ' + \
              'observability and controllability Gramians'
        zwc = get_gramians(mmat=stokesmatsc['M'].T, amat=f_mat.T,
                           jmat=stokesmatsc['J'],
                           bmat=c_mat_reg.T,
                           wmat=b_mat,
                           nwtn_adi_dict=tip['nwtn_adi_dict'])['zfac']

        # solve for the obs gramian: A.T*Wo*M + M.T*Wo*A - ... = -C*C.T
        zwo = get_gramians(mmat=stokesmatsc['M'], amat=f_mat,
                           jmat=stokesmatsc['J'], bmat=b_mat,
                           wmat=c_mat_reg.T,
                           nwtn_adi_dict=tip['nwtn_adi_dict'])['zfac']

        # save the data
        dou.save_npa(zwc, fstring=data_zwc)
        dou.save_npa(zwo, fstring=data_zwo)

    try:
        tl = dou.load_npa(data_tl)
        tr = dou.load_npa(data_tr)
        print 'loaded the left and right transformations: \n' + \
            data_tr
    except IOError:
        print 'computing the left and right transformations' + \
            ' and saving to:\n' + data_tr
        tl, tr = btu.compute_lrbt_transfos(zfc=zwc, zfo=zwo,
                                           mmat=stokesmatsc['M'])
        dou.save_npa(tl, data_tl)
        dou.save_npa(tr, data_tr)

    btu.compare_freqresp(mmat=stokesmatsc['M'], amat=f_mat,
                         jmat=stokesmatsc['J'], bmat=b_mat,
                         cmat=c_mat, tr=tr, tl=tl,
                         plot=True)


if __name__ == '__main__':
    # drivcav_lqgbt(N=10, nu=1e-1, plain_bt=True)
    lqgbt(problemname='cylinderwake', N=2, Re=1e2, plain_bt=False,
          savetomatfiles=True)
