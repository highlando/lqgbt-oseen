import numpy as np

import sadptprj_riclyap_adi.lin_alg_utils as lau
import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.stokes_navier_utils as snu


def set_inival(whichinival='sstokes', soldict=None, perturbpara=None,
               v_ss_nse=None, trange=None, tpp=None, fdstr=None,
               retvssnse=False):
    ''' compute the wanted initial value and set it in the soldict

    '''

    if (retvssnse or whichinival == 'sstate+d') and v_ss_nse is None:
        ret_v_ss_nse = snu.solve_steadystate_nse(**soldict)
    elif v_ss_nse is not None:
        ret_v_ss_nse = v_ss_nse
    else:
        ret_v_ss_nse is None

    if whichinival == 'sstokes':
        print('we start with Stokes -- `perturbpara` is not considered')
        soldict.update(dict(iniv=None, start_ssstokes=True))
        shortinivstr = 'sks'
        return shortinivstr, ret_v_ss_nse

    if whichinival == 'sstate+d' or whichinival == 'snse+d++':
        perturbini = perturbpara*np.ones((soldict['M'].shape[0], 1))
        reg_pertubini = lau.app_prj_via_sadpnt(amat=soldict['M'],
                                               jmat=soldict['J'],
                                               rhsv=perturbini)
        if whichinival == 'sstate+d':
            soldict.update(dict(iniv=ret_v_ss_nse + reg_pertubini))
            shortinivstr = 'ssd{0}'.format(perturbpara)
            return shortinivstr, ret_v_ss_nse

    if whichinival == 'sstokes++' or whichinival == 'snse+d++':
        lctrng = (trange[trange < tpp]).tolist()
        lctrng.append(tpp)

        stksppdtstr = fdstr + 't0{0:.1f}tE{1:.4f}'.\
            format(trange[0], tpp) + whichinival
        try:
            sstokspp = dou.load_npa(stksppdtstr)
            print('loaded ' + stksppdtstr + ' for inival')
        except IOError:
            inivsoldict = {}
            inivsoldict.update(soldict)  # containing A, J, JT
            inivsoldict['fv_tmdp'] = None  # don't want control here
            # import ipdb; ipdb.set_trace()
            inivsoldict.update(trange=np.array(lctrng),
                               comp_nonl_semexp=True,
                               return_dictofvelstrs=True)
            if whichinival == 'sstokes++':
                print('solving for `stokespp({0})` as inival'.format(tpp))
                inivsoldict.update(iniv=None, start_ssstokes=True)
            else:
                inivsoldict.update(iniv=ret_v_ss_nse+reg_pertubini)
                print('solving for `nse+d+pp({0})` as inival'.format(tpp))
            dcvlstrs = snu.solve_nse(**inivsoldict)
            sstokspp = dou.load_npa(dcvlstrs[tpp])
            dou.save_npa(sstokspp, stksppdtstr)
        soldict.update(dict(iniv=sstokspp))
        shortinivstr = 'sk{0}'.format(tpp) if whichinival == 'sstokes++' \
            else 'nsk{0}'.format(tpp)
        return shortinivstr, ret_v_ss_nse
