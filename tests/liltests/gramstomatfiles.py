import scipy.io as sio
import numpy as np

Re = 40

gramspath = '/scratch/tbd/dnsdata/'
gramsspec = 'cylinderwake_Re{0}.0'.format(Re) + \
        '_gamma1.0_NV41718_Bbcc_C31_palpha1e-05__'
zwc = np.load(gramspath + gramsspec + 'zwc.npy')
zwo = np.load(gramspath + gramsspec + 'zwo.npy')

savepath = '/scratch/owncloud-gwdg/mpi-projects/18-hinf-lqgbt/data/'

sio.savemat(savepath+gramsspec+'zwczwo.mat', dict(zwc=zwc, zwo=zwo))
