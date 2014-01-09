import os
import glob

import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.data_output_utils as dou

import sadptprj_riclyap_adi.lin_alg_utils as lau


def get_v_conv_conts(prev_v=None, V=None, invinds=None, diribcs=None):
    """ get and condense the linearized convection

    rhsv_con += (u_0*D_x)u_0 from the Newton scheme"""

    N1, N2, rhs_con = dts.get_convmats(u0_vec=prev_v,
                                       V=V,
                                       invinds=invinds,
                                       diribcs=diribcs)
    convc_mat, rhsv_conbc = \
        dts.condense_velmatsbybcs(N1 + N2, diribcs)

    return convc_mat, rhs_con, rhsv_conbc


def solve_steadystate_nse(A=None, J=None, JT=None,
                          fvc=None, fpr=None,
                          fv_stbc=None, fp_stbc=None,
                          V=None, Q=None, invinds=None, diribcs=None,
                          N=None, nu=None,
                          nnewtsteps=None, vel_nwtn_tol=None,
                          ddir=None, get_datastring=None,
                          paraviewoutput=False, prfdir='', prfprfx=''
                          ):
    """
    :param fvc, fpr:
        right hand sides restricted via removing the boundary nodes in the
        momentum and the pressure freedom in the continuity equation
    :param fv_stbc, fp_stbc:
        contributions to the right hand side by the Dirichlet boundary
        conditions in the stokes equations. TODO: time dependent conditions
        are not handled by now
    :param ddir:
        path to directory where the data is stored
    :param get_datastring:
        routine that returns a string describing the data
    :param paraviewoutput:
        boolean control whether paraview output is produced
    :param prfdir:
        path to directory where the paraview output is stored
    :param prfprfx:
        prefix for the output files
    """

    if get_datastring is None:
        def get_datastr(nwtn=None, time=None,
                        meshp=None, nu=None, Nts=None, dt=None):

            return ('Nwtnit{0}_time{1}_nu{2}_mesh{3}_Nts{4}_dt{5}').format(
                nwtn, time, nu, meshp, Nts, dt)

    if paraviewoutput:
        curwd = os.getcwd()
        try:
            os.chdir(prfdir)
            for fname in glob.glob(prfprfx + '*'):
                os.remove(fname)
            os.chdir(curwd)
            prvoutdict = dict(V=V, Q=Q, fstring=prfdir+prfprfx,
                              invinds=invinds, diribcs=diribcs,
                              vp=None, t=None)
        except OSError:
            raise Warning('the ' + prfdir + 'subdir for storing the' +
                          ' output does not exist. Make it yourself' +
                          'or set paraviewoutput=False')
    else:
        prvoutdict = dict(writeoutput=False)  # save 'if statements' here

    norm_nwtnupd_list = []

    NV = A.shape[0]

    newtk = 0
    vp_stokes = lau.solve_sadpnt_smw(amat=A, jmat=J, jmatT=JT,
                                     rhsv=fv_stbc + fvc,
                                     rhsp=fp_stbc + fpr
                                     )

    # a dict to be passed to the get_datastring function
    datastrdict = dict(nwtn=newtk, time=None,
                       meshp=N, nu=nu, Nts=None, dt=None)

    # save the data
    cdatstr = get_datastr(**datastrdict)

    dou.save_npa(vp_stokes[:NV, ], fstring=ddir + cdatstr + '__vel')

    prvoutdict.update(dict(vp=vp_stokes, fstring=prfdir+prfprfx+cdatstr))
    dou.output_paraview(**prvoutdict)

#
# Compute the uncontrolled steady state Navier-Stokes solution
#

    # Stokes solution as starting value
    vel_k = vp_stokes[:NV, ]

    norm_nwtnupd = 1
    while newtk < nnewtsteps:
        newtk += 1
        datastrdict.update(dict(nwtn=newtk))
        # check for previously computed velocities
        try:
            cdatstr = get_datastr(**datastrdict)

            norm_nwtnupd = dou.load_npa(ddir + cdatstr + '__norm_nwtnupd')
            vel_k = dou.load_npa(ddir + cdatstr + '__vel')

            norm_nwtnupd_list.append(norm_nwtnupd)
            print 'found vel files of Newton iteration {0}'.format(newtk)
            print 'norm of current Nwtn update: {0}'.format(norm_nwtnupd[0])

        except IOError:
            newtk -= 1
            break

    while (newtk < nnewtsteps and norm_nwtnupd > vel_nwtn_tol):
        newtk += 1

        datastrdict.update(dict(nwtn=newtk))
        cdatstr = get_datastr(**datastrdict)

        print 'Computing Newton Iteration {0} -- steady state'.\
            format(newtk)

        (convc_mat,
         rhs_con, rhsv_conbc) = get_v_conv_conts(vel_k, invinds=invinds,
                                                 V=V, diribcs=diribcs)

        vp_k = lau.solve_sadpnt_smw(amat=A+convc_mat, jmat=J, jmatT=JT,
                                    rhsv=fv_stbc+fvc+rhs_con+rhsv_conbc,
                                    rhsp=fp_stbc + fpr)

        vel_k = vp_k[:invinds, ]

        dou.save_npa(vel_k, fstring=ddir + cdatstr + '__vel')

        prvoutdict.update(dict(vp=vp_k, fstring=prfdir+prfprfx+cdatstr))
        dou.output_paraview(**prvoutdict)

        dou.save_npa(norm_nwtnupd, ddir + cdatstr + '__norm_nwtnupd')
        norm_nwtnupd_list.append(norm_nwtnupd[0])

        print 'norm of current Newton update: {}'.format(norm_nwtnupd)

    return vp_k, norm_nwtnupd_list
