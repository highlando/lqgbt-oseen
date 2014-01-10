import dolfin
#import numpy as np
#import scipy.sparse as sps
# import matplotlib.pyplot as plt
import os

import dolfin_navier_scipy.dolfin_to_sparrays as dts
#import dolfin_navier_scipy.data_output_utils as dou
from dolfin_navier_scipy.problem_setups import drivcav_fems

#import sadptprj_riclyap_adi.lin_alg_utils as lau
#import sadptprj_riclyap_adi.proj_ric_utils as pru

#import cont_obs_utils as cou
import stokes_navier_utils as snu

dolfin.parameters.linear_algebra_backend = 'uBLAS'


def time_int_params(Nts):
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
               proutdir='results',
               prfprfx='',
               nu=1e-2,
               nnewtsteps=9,  # n nwtn stps for vel comp
               vel_nwtn_tol=1e-14,
               norm_nwtnupd_list=[],
               # parameters for newton adi iteration
               nwtn_adi_dict=dict(
                   adi_max_steps=100,
                   adi_newZ_reltol=1e-5,
                   nwtn_max_steps=6,
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


class IOParams():
    """define the parameters of the input output problem

    as there are
    - dimensions of in and output space
    - extensions of the subdomains of control and observation
    """

    def __init__(self):

        self.NU, self.NY = 4, 4

        self.odcoo = dict(xmin=0.45,
                          xmax=0.55,
                          ymin=0.5,
                          ymax=0.7)
        self.cdcoo = dict(xmin=0.4,
                          xmax=0.6,
                          ymin=0.2,
                          ymax=0.3)


def drivcav_lqgbt(N=10, Nts=10):

    tip = time_int_params(Nts)
    femp = drivcav_fems(N)
    iotp = IOParams()

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
    NP = stokesmatsc['J'].shape[0]

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(rhsd_vfrc)  # adding fvc, fpr
    soldict.update(fv_stbc=rhsd_stbc['fv'], fp_stbc=rhsd_stbc['fp'],
                   N=N, nu=tip['nu'],
                   nnewtsteps=tip['nnewtsteps'],
                   vel_nwtn_tol=tip['vel_nwtn_tol'],
                   ddir=ddir, get_datastring=None,
                   paraviewoutput=False, prfdir=tip['proutdir'])

#
# compute the uncontrolled steady state Stokes solution
#
    vp_ss_nse, list_norm_nwtnupd = snu.solve_steadystate_nse(**soldict)

#
# Prepare for control
#

#    # casting some parameters
#    NY, NU = iotp.NY, iotp.NU
#
#    contsetupstr = 'NV{0}NU{1}NY{2}'.format(NV, NU, NY)
#
#    # get the control and observation operators
#    try:
#        b_mat = dou.load_spa(ddir + contsetupstr + '__b_mat')
#        u_masmat = dou.load_spa(ddir + contsetupstr + '__u_masmat')
#        print 'loaded `b_mat`'
#    except IOError:
#        print 'computing `b_mat`...'
#        b_mat, u_masmat = cou.get_inp_opa(cdcoo=iotp.cdcoo,
#                                          V=femp['V'], NU=iotp.NU)
#        dou.save_spa(b_mat, ddir + contsetupstr + '__b_mat')
#        dou.save_spa(u_masmat, ddir + contsetupstr + '__u_masmat')
#    try:
#        mc_mat = dou.load_spa(ddir + contsetupstr + '__mc_mat')
#        y_masmat = dou.load_spa(ddir + contsetupstr + '__y_masmat')
#        print 'loaded `c_mat`'
#    except IOError:
#        print 'computing `c_mat`...'
#        mc_mat, y_masmat = cou.get_mout_opa(odcoo=iotp.odcoo,
#                                            V=femp['V'], NY=iotp.NY)
#        dou.save_spa(mc_mat, ddir + contsetupstr + '__mc_mat')
#        dou.save_spa(y_masmat, ddir + contsetupstr + '__y_masmat')
#
#    # restrict the operators to the inner nodes
#    mc_mat = mc_mat[:, invinds][:, :]
#    b_mat = b_mat[invinds, :][:, :]
#
#    mct_mat_reg = lau.app_prj_via_sadpnt(amat=stokesmatsc['M'],
#                                         jmat=stokesmatsc['J'],
#                                         rhsv=mc_mat.T,
#                                         transposedprj=True)
#
#    # set the weighing matrices
#    # if iotp.R is None:
#    iotp.R = iotp.alphau * u_masmat
#    # TODO: by now we tacitly assume that V, W = MyC.T My^-1 MyC
#    # if iotp.V is None:
#    #     iotp.V = My
#    # if iotp.W is None:
#    #     iotp.W = My
#
##
## solve the differential-alg. Riccati eqn for the feedback gain X
## via computing factors Z, such that X = -Z*Z.T
##
## at the same time we solve for the affine-linear correction w
##
#
#    # tilde B = BR^{-1/2}
#    tb_mat = lau.apply_invsqrt_fromleft(iotp.R, b_mat,
#                                        output='sparse')
#
#    trct_mat = lau.apply_invsqrt_fromleft(iotp.endpy*y_masmat,
#                                          mct_mat_reg, output='dense')
#
#    cntpstr = 'NY{0}NU{1}alphau{2}'.format(iotp.NU, iotp.NY, iotp.alphau)
#
#    # set/compute the terminal values aka starting point
#    Zc = lau.apply_massinv(stokesmatsc['M'], trct_mat)
#    wc = -lau.apply_massinv(stokesmatsc['MT'],
#                            np.dot(mct_mat_reg, iotp.ystarvec(tip['tE'])))
#
#    cdatstr = get_datastr(nwtn=newtk, time=tip['tE'], meshp=N, timps=tip)
#
#    dou.save_npa(Zc, fstring=ddir + cdatstr + cntpstr + '__Z')
#    dou.save_npa(wc, fstring=ddir + cdatstr + cntpstr + '__w')
#
#    # we will need transposes, and explicit is better than implicit
#    # here, the coefficient matrices are symmetric
#    stokesmatsc.update(dict(MT=stokesmatsc['M'],
#                            AT=stokesmatsc['A']))
#
#    # we gonna use this quite often
#    MT, AT = stokesmatsc['MT'], stokesmatsc['AT']
#    M, A = stokesmatsc['M'], stokesmatsc['A']
#
#    for t in np.linspace(tip['tE'] - DT, tip['t0'], Nts):
#        print 'Time is {0}'.format(t)
#
#        # get the previous time convection matrices
#        pdatstr = get_datastr(nwtn=newtk, time=t, meshp=N, timps=tip)
#        prev_v = dou.load_npa(ddir + pdatstr + '__vel')
#        convc_mat, rhs_con, rhsv_conbc = get_v_conv_conts(prev_v,
#                                                          femp, tip)
#
#        try:
#            Zc = dou.load_npa(ddir + pdatstr + cntpstr + '__Z')
#        except IOError:
#
#            # coeffmat for nwtn adi
#            ft_mat = -(0.5*stokesmatsc['MT'] + DT*(stokesmatsc['AT'] +
#                                                   convc_mat.T))
#            # rhs for nwtn adi
#            w_mat = np.hstack([stokesmatsc['MT']*Zc, np.sqrt(DT)*trct_mat])
#
#            Zp = pru.proj_alg_ric_newtonadi(mmat=stokesmatsc['MT'],
#                                            fmat=ft_mat, transposed=True,
#                                            jmat=stokesmatsc['J'],
#                                            bmat=np.sqrt(DT)*tb_mat,
#                                            wmat=w_mat, z0=Zc,
#                                            nwtn_adi_dict=tip['nwtn_adi_dict']
#                                            )['zfac']
#
#            if tip['compress_z']:
#                # Zc = pru.compress_ZQR(Zp, kmax=tip['comprz_maxc'])
#                Zc = pru.compress_Zsvd(Zp, thresh=tip['comprz_thresh'])
#                # monitor the compression
#                vec = np.random.randn(Zp.shape[0], 1)
#                print 'dims of Z and Z_red: ', Zp.shape, Zc.shape
#                print '||(ZZ_red - ZZ )*testvec|| / ||ZZ_red*testvec|| = {0}'.\
#                    format(np.linalg.norm(np.dot(Zp, np.dot(Zp.T, vec)) -
#                           np.dot(Zc, np.dot(Zc.T, vec))) /
#                           np.linalg.norm(np.dot(Zp, np.dot(Zp.T, vec))))
#            else:
#                Zc = Zp
#
#            if tip['save_full_z']:
#                dou.save_npa(Zp, fstring=ddir + pdatstr + cntpstr + '__Z')
#            else:
#                dou.save_npa(Zc, fstring=ddir + pdatstr + cntpstr + '__Z')
#
#        ### and the affine correction
#        ftilde = rhs_con[INVINDS, :] + rhsv_conbc + rhsd_vfstbc['fv']
#        at_mat = MT + DT*(AT + convc_mat.T)
#        rhswc = MT*wc + DT*(mc_mat.T*iotp.ystarvec(t) -
#                            MT*np.dot(Zc, np.dot(Zc.T, ftilde)))
#
#        amat, currhs = dts.sadpnt_matsrhs(at_mat, stokesmatsc['J'], rhswc)
#
#        umat = DT*MT*np.dot(Zc, Zc.T*tb_mat)
#        vmat = tb_mat.T
#
#        vmate = sps.hstack([vmat, sps.csc_matrix((vmat.shape[0], NP))])
#        umate = np.vstack([umat, np.zeros((NP, umat.shape[1]))])
#
#        wc = lau.app_smw_inv(amat, umat=-umate, vmat=vmate, rhsa=currhs)[:NV]
#        dou.save_npa(wc, fstring=ddir + pdatstr + cntpstr + '__w')
#
#    # solve the closed loop system
#    set_vpfiles(tip, fstring=('results/' + 'closedloop' + cntpstr +
#                              'NewtonIt{0}').format(newtk))
#
#    v_old = inivalvec
#    for t in np.linspace(tip['t0']+DT, tip['tE'], Nts):
#
#        # t for implicit scheme
#        ndatstr = get_datastr(nwtn=newtk, time=t,
#                              meshp=N, timps=tip)
#
#        # convec mats
#        next_v = dou.load_npa(ddir + ndatstr + '__vel')
#        convc_mat, rhs_con, rhsv_conbc = get_v_conv_conts(next_v,
#                                                          femp, tip)
#
#        # feedback mats
#        next_zmat = dou.load_npa(ddir + ndatstr + cntpstr + '__Z')
#        next_w = dou.load_npa(ddir + ndatstr + cntpstr + '__w')
#        print 'norm of w:', np.linalg.norm(next_w)
#
#        umat = DT*MT*np.dot(next_zmat, next_zmat.T*tb_mat)
#        vmat = tb_mat.T
#
#        vmate = sps.hstack([vmat, sps.csc_matrix((vmat.shape[0], NP))])
#        umate = DT*np.vstack([umat, np.zeros((NP, umat.shape[1]))])
#
#        fvn = rhs_con[INVINDS, :] + rhsv_conbc + rhsd_vfstbc['fv']
#        # rhsn = M*next_v + DT*(fvn + tb_mat * (tb_mat.T * next_w))
#        rhsn = M*v_old + DT*(fvn + 0*tb_mat * (tb_mat.T * next_w))
#
#        amat = M + DT*(A + convc_mat)
#        rvec = np.random.randn(next_zmat.shape[0], 1)
#        print 'norm of amat', np.linalg.norm(amat*rvec)
#        print 'norm of gain mat', np.linalg.norm(np.dot(umat, vmat*rvec))
#
#        amat, currhs = dts.sadpnt_matsrhs(amat, stokesmatsc['J'], rhsn)
#
#        vpn = lau.app_smw_inv(amat, umat=-umate, vmat=vmate, rhsa=currhs)
#        # vpn = np.atleast_2d(sps.linalg.spsolve(amat, currhs)).T
#        v_old = vpn[:NV]
#
#        yn = lau.apply_massinv(y_masmat, mc_mat*vpn[:NV])
#        print 'current y: ', yn
#
#        dou.save_npa(vpn[:NV], fstring=ddir + cdatstr + '__cont_vel')
#
#        dou.output_paraview(tip, femp, vp=vpn, t=t),
#
#    print 'dim of v :', femp['V'].dim()

if __name__ == '__main__':
    drivcav_lqgbt(N=15, Nts=2)
