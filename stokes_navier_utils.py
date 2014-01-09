import dolfin_navier_scipy.dolfin_to_sparrays as dts

import sadptprj_riclyap_adi.lin_alg_utils as lau


def get_v_conv_conts(prev_v, femp=dict(V=None,
                                       invinds=None,
                                       diribcs=None)):
    """ get and condense the linearized convection

    rhsv_con += (u_0*D_x)u_0 from the Newton scheme"""

    N1, N2, rhs_con = dts.get_convmats(u0_vec=prev_v,
                                       V=femp['V'],
                                       invinds=femp['invinds'],
                                       diribcs=femp['diribcs'])
    convc_mat, rhsv_conbc = \
        dts.condense_velmatsbybcs(N1 + N2, femp['diribcs'])

    return convc_mat, rhs_con, rhsv_conbc


def solve_steadystate_nse(stokesmatsc=dict(A=None,
                                           J=None,
                                           JT=None
                                           ),
                          rhsd_vfrc=dict(fv=None, fp=None),
                          rhsd_stbc=dict(fv=None, fp=None),
                          femp=dict(V=None,
                                    invinds=None,
                                    diribcs=None
                                    ),
                          tip=dict(nu=None,
                                   Nts=None,
                                   dt=None,
                                   nnewtsteps=None),
                          get_datastring=None):

    if get_datastring is None:
        def get_datastr(nwtn=None, time=None,
                        meshp=None, tip=dict(nu=None, Nts=None, dt=None)):

            return ('Nwtnit{0}_time{1}_nu{2}_mesh{3}_Nts{4}_dt{5}').format(
                nwtn, time, tip['nu'], meshp,
                tip['Nts'], tip['dt'])

    newtk = 0
    vp_stokes = lau.solve_sadpnt_smw(amat=stokesmatsc['A'],
                                     jmat=stokesmatsc['J'],
                                     jmatT=stokesmatsc['JT'],
                                     rhsv=rhsd_stbc['fv'] + rhsd_vfrc['fv'],
                                     rhsp=rhsd_stbc['fp'] + rhsd_vfrc['fp']
                                     )
