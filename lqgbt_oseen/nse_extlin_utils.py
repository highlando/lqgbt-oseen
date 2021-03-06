import numpy as np
import scipy.linalg as spla
import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.data_output_utils as dou


def get_get_cur_extlin(vinf=None, V=None, diribcs=None, invinds=None,
                       reducedmodel=False, tl=None, tr=None,
                       picrdvsnwtn=0.,
                       amat=None, akmat=None, **kwargs):

    ''' returns a function to compute the current extended linearization

    Parameters
    ---
    vinf: array
        the set point
    use_ric_ini : string, optional
        path to a stabilizing initial guess
    '''

    convc_mat, vinablavi, _ = \
        snu.get_v_conv_conts(prev_v=vinf, invinds=invinds,
                             retparts=False,
                             V=V, diribcs=diribcs)

    def get_cur_extlin(vcur=None, vdelta=None):
        ''' compute the current extended linearization

        dot vd + A(vd)vd = rhs

        for the NSE momentum eqn (shifted by the setpoint vinf)
        which basically reads

        A(vd)(.) = amat*(.) + (.*nabla)*vinf + ([vinf+vd]*nabla)*(.)

        or

        A(vd)(.) = amat*(.) + (.*nabla)*[vinf+vd] + (vinf*nabla)*(.)

        or anything in between

        Parameters
        ---
        vdelta: array
            the difference between the (target) set point and the current state
        vcur: nparray
            the current state
        picrdvsnwtn: float, optional
            blends between the picard (0.) and antipicard (1.) part
            of the extended linearization
        '''

        if vdelta is None:
            vdelta = vcur - vinf

        fullnv = V.dim()
        vdpbc = np.zeros((fullnv, 1))  # vdelta has zero bcs
        vdpbc[invinds] = vdelta
        delta_convmats, vdnablavd, _ = \
            snu.get_v_conv_conts(prev_v=vdpbc, invinds=invinds,
                                 retparts=True, zerodiribcs=True,
                                 V=V, diribcs=None)

        # testing to see the effect of the zero dirichlets...
        # tstdelta_convmats, tstvdnablavd, tstrhs = \
        #     snu.get_v_conv_conts(prev_v=vdelta, invinds=invinds,
        #                          retparts=True,
        #                          V=V, diribcs=diribcs)

        # delta_convmats[0] -- Picard part
        # delta_convmats[1] -- Anti Picard (Newton minus Picard)
        # one may think of convex combining both parts
        if picrdvsnwtn < 0 or picrdvsnwtn > 1:
            raise UserWarning('interesting parameter value -- good luck!!!')

        curfmat = amat + convc_mat + \
            picrdvsnwtn*delta_convmats[0] + (1-picrdvsnwtn)*delta_convmats[1]

        return curfmat

    return get_cur_extlin


def _pinsvd(jmat=None, Minv=None):

    Np, Nv = jmat.shape

    minvjt = Minv(jmat.T).todense()
    S = jmat*minvjt
    Sinv = spla.inv(S)
    Pi = np.eye(Nv) - minvjt.dot(Sinv*jmat)

    umat, svec, vmat = spla.svd(Pi)
    uk = umat[:, :Nv-Np]
    vk = vmat.T[:, :Nv-Np]

    thl = uk*svec[:Nv-Np]
    thr = vk

    return thl, thr


def decomp_leray(mmat, jmat, dataprfx='', testit=False):
    from spacetime_galerkin_pod.ldfnp_ext_cholmod import SparseFactorMassmat

    facmy = SparseFactorMassmat(mmat)
    Minv = facmy.solve_M

    pinsvddict = dict(jmat=jmat, Minv=Minv)
    thrthlstrl = [dataprfx+'_pithl', dataprfx+'_pithr']
    thl, thr = dou.load_or_comp(arraytype='dense', filestr=thrthlstrl,
                                numthings=2,
                                comprtn=_pinsvd, comprtnargs=pinsvddict)
    minvthr = Minv(thr)

    if testit:
        Np, Nv = jmat.shape
        print(np.allclose(np.eye(Nv-Np), thr.T.dot(thl)))
        minvjt = Minv(jmat.T).todense()
        S = jmat*minvjt
        Sinv = spla.inv(S)
        Pi = np.eye(Nv) - minvjt.dot(Sinv*jmat)
        print(np.allclose(Pi, thl.dot(thr.T)))

    return thl, thr, minvthr
