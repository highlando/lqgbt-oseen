import dolfin_navier_scipy.stokes_navier_utils as snu


def get_get_cur_extlin(vinf=None, V=None, diribcs=None, invinds=None,
                       reducedmodel=False, tl=None, tr=None,
                       amat=None, akmat=None, **kwargs):

    ''' returns a function to compute the current extended linearization

    Parameters
    ---
    vinf: array
        the set point
    use_ric_ini : string, optional
        path to a stabilizing initial guess
    '''

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
        '''

        if vdelta is None:
            vdelta = vcur - vinf

        convc_mat, _, _ = \
            snu.get_v_conv_conts(prev_v=vinf, invinds=invinds,
                                 retparts=False,
                                 V=V, diribcs=diribcs)

        delta_convmats, _, _ = \
            snu.get_v_conv_conts(prev_v=vinf, invinds=invinds,
                                 retparts=False,
                                 V=V, diribcs=diribcs)
        # delta_convmats[0] -- Picard part
        # delta_convmats[1] -- Anti Picard (Newton minus Picard)
        # one may think of convex combining both parts

        curfmat = - amat - convc_mat - delta_convmats[0]

        return curfmat

    return get_cur_extlin
