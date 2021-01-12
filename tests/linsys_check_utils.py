import numpy as np

from scipy.optimize import bisect

import sadptprj_riclyap_adi.bal_trunc_utils as btu


def hinf_rom_stable(zwc=None, zwo=None, mmat=None, gamma=None,
                    iniint=(1e-5, 10), maxiter=20, xtol=1e-4):

    beta = np.sqrt(1-1/gamma**2)

    def stblyes(threshh):
        tl, tr, svs = btu.\
            compute_lrbt_transfos(zfc=zwc, zfo=zwo, mmat=mmat,
                                  trunck={'threshh': threshh})
        romdim = tl.shape[1]
        trnctsvs = svs[romdim:].flatten()
        epsilon = 2*(trnctsvs / np.sqrt(1+beta**2*trnctsvs**2)).sum()
        print('k={0}: -- threshh: {1:.2e} -- stable if <0`: {2:.5f}'.
              format(romdim, threshh, epsilon*beta-1/gamma))
        return epsilon*beta - 1/gamma

    optithresh = bisect(stblyes, iniint[0], iniint[1],
                        maxiter=maxiter, xtol=xtol)

    tl, tr, svs = btu.\
        compute_lrbt_transfos(zfc=zwc, zfo=zwo, mmat=mmat,
                              trunck={'threshh': optithresh})
    romdim = tl.shape[1]
    trnctsvs = svs[romdim:].flatten()
    epsilon = 2*(trnctsvs / np.sqrt(1+beta**2*trnctsvs**2)).sum()

    print('kopti: {0} -- threshhold: {1:.3e} -- stable if `<0`: {2:.5f}'.
          format(romdim, optithresh, epsilon*beta-1/gamma))

    return romdim, optithresh, tl, tr
