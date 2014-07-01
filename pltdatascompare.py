import dolfin_navier_scipy.data_output_utils as dou
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
import numpy as np


def compplot_deviations(jsonfilelist, fignum=333, figname='figdiff',
                        compress=10):
    """ compute the deviations of the measured output from the stable

    state output
    """

    import matplotlib.cm as cm
    fig = plt.figure(fignum)
    ax1 = fig.add_subplot(111)
    diffnormlist = []
    collin = np.linspace(0.2, 0.7, len(jsonfilelist))

    for k, jsfstr in enumerate(jsonfilelist):
        jsf = dou.load_json_dicts(jsfstr)
        tmesh = jsf['tmesh']
        redinds = range(0, len(tmesh), compress)
        redina = np.array(redinds)
        outsig = np.array(jsf['outsig'])
        # two norm at the differences
        outsigdiff = np.sqrt(((outsig - outsig[0, :]) *
                             (outsig - outsig[0, :])).sum(axis=1))
        diffnormlist.append((tmesh[-1]-tmesh[0])/len(tmesh) * outsigdiff.sum())
        curline, = ax1.plot(np.array(tmesh)[redina], outsigdiff[redina],
                            c=cm.CMRmap(collin[k]), linewidth=2.0,
                            label='{0}'.format(k))

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='lower right')

    tikz_save(figname + '{0}'.format(fignum) + '.tikz',
              figureheight='\\figureheight',
              figurewidth='\\figurewidth'
              )
    print 'saved to ' + figname + '{0}'.format(fignum) + '.tikz'
    fig.show()
    print diffnormlist
    print len(tmesh)


def compplotfreqresps(jsonfilelist, fignum=444, figname='freqfigdiff',
                      compress=1):
    """ plot the errors in the freqresp for various configurations

    Parameters:
    -----------
    jsonfilelist : list
        of json files as produced by `btu.compare_freqresp`
    """
    import matplotlib.cm as cm
    fig = plt.figure(fignum)
    ax1 = fig.add_subplot(111)
    collin = np.linspace(0.2, 0.7, len(jsonfilelist))

    for k, jsfstr in enumerate(jsonfilelist):
        jsf = dou.load_json_dicts(jsfstr)
        tmesh = jsf['tmesh']
        redinds = range(0, len(tmesh), compress)
        redina = np.array(redinds)
        outsig = np.array(jsf['diffsysfr'])
        curline, = ax1.plot(np.array(tmesh)[redina], outsig[redina],
                            c=cm.CMRmap(collin[k]), linewidth=2.0,
                            label='{0}'.format(k))

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='lower left')

    tikz_save(figname + '{0}'.format(fignum) + '.tikz',
              figureheight='\\figureheight',
              figurewidth='\\figurewidth'
              )

    print 'saved to ' + figname + '{0}'.format(fignum) + '.tikz'
    ax1.semilogx()
    ax1.semilogy()
    fig.show()

if __name__ == '__main__':

    freqresplist = [
        "data/cylinderwake_Re200.0_NV9356NU3NY3_lqgbt__tl__lqgbtcv1e-05forfreqrespplot"
        ,
        "data/cylinderwake_Re200.0_NV9356NU3NY3_lqgbt__tl__lqgbtcv0.001forfreqrespplot"
        ,
        "data/cylinderwake_Re200.0_NV9356NU3NY3_lqgbt__tl__lqgbtcv0.01forfreqrespplot"
        ,
        "data/cylinderwake_Re200.0_NV9356NU3NY3_lqgbt__tl__lqgbtcv0.1forfreqrespplot"
                ]

    compplotfreqresps(freqresplist)
    jsfl = ["data/cylinderwake_Re200.0_NV9356NU3NY3_lqgbt__lqgbtcv1e-05red_output_fbt00.0tE12.0Nts2401.0",
            "data/cylinderwake_Re200.0_NV9356NU3NY3_lqgbt__lqgbtcv0.0001red_output_fbt00.0tE12.0Nts2401.0",
            "data/cylinderwake_Re200.0_NV9356NU3NY3_lqgbt__lqgbtcv0.001red_output_fbt00.0tE12.0Nts2401.0",
            "data/cylinderwake_Re200.0_NV9356NU3NY3_lqgbt__lqgbtcv0.01red_output_fbt00.0tE12.0Nts2401.0",
            "data/cylinderwake_Re200.0_NV9356NU3NY3_lqgbt__lqgbtcv0.1red_output_fbt00.0tE12.0Nts2401.0",
             "data/cylinderwake_Re200.0_NV9356NU3NY3_lqgbt__lqgbtcv1.0red_output_fbt00.0tE12.0Nts2401.0"]
    compplot_deviations(jsfl)
