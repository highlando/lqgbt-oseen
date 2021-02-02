import numpy as np
import json
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--cw", help="Check the cylinder wake data",
                    action='store_true')
parser.add_argument("--dbrc", help="Check the double cylinder data",
                    action='store_true')
parser.add_argument("--showplots", help="Whether to show the plots",
                    action='store_true')
args = parser.parse_args()


def plot_outp_sig(str_to_json=None, tmeshkey='tmesh', sigkey='outsig',
                  plotplease=True,
                  outsig=None, tmesh=None, fignum=222, ttlstr=''):

    try:
        with open(str_to_json) as fjs:
            jsdict = json.load(fjs)
    except IOError:
        print('not found:', str_to_json)
        return None

    tmesh = jsdict[tmeshkey]
    outsig = jsdict[sigkey]
    ntt = np.int(len(outsig)/2)
    ntf = np.int(len(outsig)/4)

    redinds = list(range(ntt, len(outsig), 8))
    redina = np.array(redinds)

    # NY = np.int(len(outsig[0])/2)
    # NY = 1

    frstcmp = np.array(outsig)[:, 0]
    if np.isnan(frstcmp[-1]):
        return False
    frstcmpm = np.mean(frstcmp[ntt:])
    vardiff = np.var(frstcmp[ntt:ntt+ntf]) - np.var(frstcmp[ntt+ntf:])
    stable = vardiff > -1e-15 and vardiff < 1e-5
    if plotplease:
        fig = plt.figure(fignum)
        ax1 = fig.add_subplot(111)
        ax1.plot(np.array(tmesh)[redina], frstcmp[redina]-frstcmpm,
                 color='b', linewidth=2.0)
        # ax1.plot(np.array(tmesh)[redina],
        #          np.array(outsig)[redina, :NY]-ousm,
        #          color='b', linewidth=2.0)
        if stable:
            ttlstr = 'STABLE: ' + ttlstr
        ax1.set_title(ttlstr)
    return stable


problem = 'dbrc'
problem = 'cw'

if args.cw and args.dbrc:
    raise UserWarning("Cannot do both 'cw' and 'dbrc' at a time")

if args.cw:
    problem = 'cw'
if args.dbrc:
    problem = 'dbrc'

if problem == 'cw':
    probstr = '../data/cw60.01.0_4171831-16hinfrofb'
    afterstr = 't0.030.0000Nts40000ab0.01.0A1e-06'
    truncatsl = [0.0064, 0.0032, 0.0016, 0.0004, 0.0001]
    mafl = [17, 20, 24, 28, 34, 37, 40, 48]
    sstblfile = 'cw60sstbl.dat'
    sntblfile = 'cw60sntbl.dat'
    nntblfile = 'cw60nntbl.dat'
    nstblfile = 'cw60nstbl.dat'
    trcthrsh = 0.00166
    ellthrsh = 45

    def getsustr(trcstr, maf):
        return 'pm{0}ssd0.0mfpi{1}'.format(trcstr, maf)

elif problem == 'dbrc':
    probstr = '../data/drc60.01.0_4552841-16hinfrofb'
    afterstr = 't0.0300.0000Nts76800ab0.01.0A1e-06'
    truncatsl = [0.512, 0.128, 0.064, 0.045, 0.032, 0.008]
    # truncatsl = [0.128, 0.09, 0.064]
    sstblfile = 'db60sstbl.dat'
    sntblfile = 'db60sntbl.dat'
    nstblfile = 'db60nstbl.dat'
    nntblfile = 'db60nntbl.dat'
    mafl = [192, 135, 96, 48, 24, 12, 6]
    trcthrsh = 0.00166
    ellthrsh = 45
    trcthrsh = 0.155
    ellthrsh = 25

    def getsustr(trcstr, maf):
        return 'pm{0}ssd0.0mfpr-{1}'.format(trcstr, maf)

# bpm0.512ssd0.0mfpr-6t0.0300.0000Nts76800ab0.01.0A1e-06
# mafl = [34]
sstbllst = []
snstblst = []
nstbllst = []
nnstblst = []

checkplots = True
checkplots = False
if args.showplots:
    checkplots = True


k = 0

for ctrc in truncatsl:
    for maf in mafl:
        # sustr = 'pm{0}ssd0.0maf{1}'.format(trcstr, maf)
        sustr = getsustr('{0}'.format(ctrc), maf)
        # afterstr = 't0.018.0000Nts24000ab0.01.0A1e-06'
        tltstr = problem + 'pm: {0} -- maf: {1}'.format(ctrc, maf)
        stbl = plot_outp_sig(probstr + sustr + afterstr,
                             plotplease=checkplots,
                             fignum=k, ttlstr=tltstr)
        print('{0}: '.format(k) + tltstr, stbl)
        idx = ['{0}'.format(ctrc), '{0}'.format(maf)]
        if problem == 'dbrc':
            mafcheck = maf < ellthrsh
        else:
            mafcheck = maf > ellthrsh
        if ctrc < trcthrsh and mafcheck:
            sstbllst.append(idx) if stbl else snstblst.append(idx)
        else:
            nstbllst.append(idx) if stbl else nnstblst.append(idx)
        k += 1
if checkplots:
    plt.show()

with open(sstblfile, 'w') as sfile:
    sfile.write('trc l\n')
    for idx in sstbllst:
        sfile.write(idx[0]+' '+idx[1]+'\n')

with open(sntblfile, 'w') as nfile:
    nfile.write('trc l\n')
    for idx in snstblst:
        nfile.write(idx[0]+' '+idx[1]+'\n')

with open(nntblfile, 'w') as nfile:
    nfile.write('trc l\n')
    for idx in nnstblst:
        nfile.write(idx[0]+' '+idx[1]+'\n')

with open(nstblfile, 'w') as nfile:
    nfile.write('trc l\n')
    for idx in nstbllst:
        nfile.write(idx[0]+' '+idx[1]+'\n')
