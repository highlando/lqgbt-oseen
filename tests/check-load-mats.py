import mat73
import numpy as np

gramsfile = '/scratch/owncloud-gwdg/mpi-projects/18-hinf-lqgbt/results/cylinderwake_re50_hinf.mat%outRegulator.Z%outFilter.Z%gam'
(fname, zwcaddress, zwoaddress, gammadress) = gramsfile.split(sep='%')
lmd = mat73.loadmat(fname)
zwc, zwo, gamma = lmd, lmd, lmd
for isitthis in zwcaddress.split(sep='.'):
    zwc = zwc[isitthis]
for isitthis in zwoaddress.split(sep='.'):
    zwo = zwo[isitthis]
for isitthis in gammadress.split(sep='.'):
    gamma = gamma[isitthis]
gamma = np.atleast_1d(gamma).flatten()[0]
