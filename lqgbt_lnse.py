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

dolfin.parameters.linear_algebra_backend = 'uBLAS'


def nwtn_adi_params():
    return dict(nwtn_adi_dict=dict(
                adi_max_steps=300,
                adi_newZ_reltol=1e-7,
                nwtn_max_steps=20,
                nwtn_upd_reltol=4e-8,
                nwtn_upd_abstol=1e-7,
                verbose=True,
                full_upd_norm_check=False,
                check_lyap_res=False))


def lqgbt(problemname='drivencavity',
          N=10, Re=1e2, plain_bt=True,
          use_ric_ini=None, t0=0.0, tE=1.0, Nts=10,
          plot_freqresp=False, plot_stepresp=True):

    problemdict = dict(drivencavity=dnsps.drivcav_fems,
                       cylinderwake=dnsps.cyl_fems)

    typprb = 'BT' if plain_bt else 'LQG-BT'

    print '\n ### We solve the {0} problem for the {1} at Re={2} ###\n'.\
        format(typprb, problemname, Re)

    problemfem = problemdict[problemname]
    femp = problemfem(N)

    NU, NY = 3, 3

    # specify in what spatial direction Bu changes. The remaining is constant
    if problemname == 'drivencavity':
        charlen = 1.0
        uspacedep = 0
    elif problemname == 'cylinderwake':
        charlen = 0.15
        uspacedep = 1
    nu = charlen/Re

    nap = nwtn_adi_params()
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
    NV, INVINDS = len(femp['invinds']), femp['invinds']

    prbstr = '_bt' if plain_bt else '_lqgbt'
    contsetupstr = 'NV{0}NU{1}NY{2}'.format(NV, NU, NY)

    def get_fdstr(Re):
        return ddir + problemname + '_Re{0}_'.format(Re) + \
            contsetupstr + prbstr

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(rhsd_vfrc)  # adding fvc, fpr
    soldict.update(fv_stbc=rhsd_stbc['fv'], fp_stbc=rhsd_stbc['fp'],
                   N=N, nu=nu, ddir=ddir)

#
# compute the uncontrolled steady state Stokes solution
#
    v_ss_nse, list_norm_nwtnupd = snu.solve_steadystate_nse(**soldict)

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

    fdstr = get_fdstr(Re)
    try:
        tl = dou.load_npa(fdstr + '__tl')
        tr = dou.load_npa(fdstr + '__tr')
        print 'loaded the left and right transformations: \n' + \
            fdstr + '__tl/__tr'
    except IOError:
        print 'computing the left and right transformations' + \
            ' and saving to: \n' + fdstr + '__tl/__tr'

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
            zwc = get_gramians(mmat=mmat.T, amat=f_mat.T,
                               jmat=stokesmatsc['J'],
                               bmat=c_mat_reg.T, wmat=b_mat,
                               nwtn_adi_dict=nap['nwtn_adi_dict'],
                               z0=zinic)['zfac']
            dou.save_npa(zwc, fdstr + '__zwc')
            zwo = get_gramians(mmat=mmat, amat=f_mat,
                               jmat=stokesmatsc['J'],
                               bmat=b_mat, wmat=c_mat_reg.T,
                               nwtn_adi_dict=nap['nwtn_adi_dict'],
                               z0=zinio)['zfac']
            dou.save_npa(zwo, fdstr + '__zwo')

        print 'computing the left and right transformations' + \
            ' and saving to:\n' + fdstr + '__tr/__tl'
        tl, tr = btu.compute_lrbt_transfos(zfc=zwc, zfo=zwo,
                                           mmat=stokesmatsc['M'])
        dou.save_npa(tl, fdstr + '__tl')
        dou.save_npa(tr, fdstr + '__tr')

    if plot_freqresp:
        btu.compare_freqresp(mmat=stokesmatsc['M'], amat=f_mat,
                             jmat=stokesmatsc['J'], bmat=b_mat,
                             cmat=c_mat, tr=tr, tl=tl,
                             plot=True)

    def fullstepresp_lnse(bcol=None, trange=None, ini_vel=None,
                          cmat=None, soldict=None):
        soldict.update(fv_stbc=rhsd_stbc['fv']+bcol,
                       vel_nwtn_stps=1, trange=trange,
                       iniv=ini_vel, lin_vel_point=ini_vel,
                       clearprvdata=True, data_prfx='stepresp_',
                       return_dictofvelstrs=True)

        dictofvelstrs = snu.solve_nse(**soldict)

        return cou.extract_output(strdict=dictofvelstrs, tmesh=trange,
                                  c_mat=cmat, load_data=dou.load_npa)

    print np.dot(c_mat_reg, v_ss_nse)
    print np.dot(np.dot(c_mat_reg, tr),
                 np.dot(tl.T, stokesmatsc['M']*v_ss_nse))

    if plot_stepresp:
        btu.compare_stepresp(tmesh=np.linspace(t0, tE, Nts),
                             a_mat=f_mat, c_mat=c_mat_reg, b_mat=b_mat,
                             m_mat=stokesmatsc['M'],
                             tr=tr, tl=tl, iniv=v_ss_nse,
                             # ss_rhs=ssv_rhs,
                             fullresp=fullstepresp_lnse, fsr_soldict=soldict,
                             plot=True)

if __name__ == '__main__':
    # lqgbt(N=10, Re=500, use_ric_ini=None, plain_bt=False)
    lqgbt(problemname='cylinderwake', N=2,  # use_ric_ini=1e2,
          Re=1e2, plain_bt=False,
          t0=0.0, tE=2.0, Nts=1e2+1,
          plot_freqresp=False, plot_stepresp=True)
