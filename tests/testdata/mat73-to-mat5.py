import mat73
import scipy.io


matlist = ['cylinderwake_re60_hinf',
           'doublecylinder_re60_hinf']

for matstr in matlist:
    lmd = mat73.loadmat(matstr+'.mat')
    try:
        scipy.io.savemat(matstr+'v5.mat', lmd, do_compression=True)
    except TypeError:
        jtm = {'gam': lmd['gam'],
               'outFilter': lmd['outFilter']['Z'],
               'outRegulator': lmd['outRegulator']['Z'],
               }
        scipy.io.savemat(matstr+'v5.mat', jtm, do_compression=True)
        print(matstr + ': stripped data since cannot handle `None`')
