import numpy as np
import mat73
from scipy.io import loadmat
import sadptprj_riclyap_adi.bal_trunc_utils as btu
import sadptprj_riclyap_adi.lin_alg_utils as lau
import matplotlib.pyplot as plt
import lqgbt_oseen.nse_riccont_utils as nru

from scipy.linalg import solve_continuous_are

# %% CYLINDERWAKE_RE20_ROM_CONTROL
# % Script for computing the ROM and controller for:
# %
# %   cylinderwake_Re20.0_gamma1.0_NV41700_Bbcc_C31_palpha1e-05__mats.mat

# %% Setup problem data.
print('Setup problem data.')
print('-------------------')

fname = '/scratch/owncloud-gwdg/mpi-projects/18-hinf-lqgbt/data/' + \
        'cylinderwake_Re40.0_gamma1.0_NV41718_Bbcc_C31_palpha1e-05__mats'
matdict = loadmat(fname)
mmat = matdict['mmat']
amat = matdict['amat']
cmat = matdict['cmat']
bmat = matdict['bmat']
print('|M|: ', np.linalg.norm(mmat.data))
print('|A|: ', np.linalg.norm(amat.data))
print('|B|: ', np.linalg.norm(bmat.data))
print('|C|: ', np.linalg.norm(cmat.data))

# % System sizes.
# st = size(mmat, 1);
# nz = size(jmat, 1);
# m  = size(bmat, 2);
# p  = size(cmat, 1);

# %% Load Riccati results.
print('Load Riccati results.')
print('---------------------')

fname = '/scratch/owncloud-gwdg/mpi-projects/18-hinf-lqgbt/results/' + \
        'cylinderwake_re40_hinf.mat'
lmd = mat73.loadmat(fname)
zwc = lmd['outRegulator']['Z']
zwo = lmd['outFilter']['Z']
gam = lmd['gam']

scfc = np.sqrt(gam**2-1)/gam  # [1]
scfc = np.sqrt(1-1/gam**2)  # [2]

print('Check the Riccati Residuals')
print('------------')

cricres = lau.comp_sqfnrm_factrd_riccati_res(M=mmat, A=amat, B=scfc*bmat,
                                             C=cmat, Z=zwc)
normxc = ((zwc.T@zwc) ** 2).sum(-1).sum()
print('c-ric-res^2: ', cricres)
print('norm-xc^2: ', normxc)
print('relres: ', np.sqrt(cricres/normxc))
oricres = lau.comp_sqfnrm_factrd_riccati_res(M=mmat, A=amat.T, B=scfc*cmat.T,
                                             C=bmat.T, Z=zwo)
normxo = ((zwo.T@zwo) ** 2).sum(-1).sum()
print('oricres^2: ', oricres)
print('normxo^2: ', normxo)
print('relres: ', np.sqrt(oricres/normxo))

print('Compute ROM.')
print('------------')

# # % SR method.
# [U, S, V] = svd(outRegulator.Z' * (mmat * outFilter.Z), 'econ');

print('|zfo|: ', np.linalg.norm(zwo))
print('|zfc|: ', np.linalg.norm(zwc))
tl, tr, svs = btu.\
    compute_lrbt_transfos(zfc=zwc, zfo=zwo, mmat=mmat,
                          trunck={'threshh': 0.1})
print('|tl|: ', np.linalg.norm(tl))
print('|tr|: ', np.linalg.norm(tr))
cntrlsz = tl.shape[1]
plt.figure(1)
plt.semilogy(svs[:cntrlsz], 'x')
plt.show(block=False)

# r = 25;
# S = sparse(1:r, 1:r, 1 ./ sqrt(hsv(1:r)));
# W = S * (outRegulator.Z * U(: , 1:r))';
# T = (outFilter.Z * V(: , 1:r)) * S;

# % Compute ROM.
Er = tl.T.dot(mmat * tr)
Ar = tl.T.dot(amat * tr)
Br = tl.T.dot(bmat)
Cr = cmat.dot(tr)

ak_mat, bk_mat, ck_mat, xok, xck = nru.\
    get_prj_model(mmat=mmat, fmat=amat, jmat=None,
                  zwo=zwo, zwc=zwc,
                  tl=tl, tr=tr,
                  bmat=bmat, cmat=cmat)
riccres = ak_mat.T.dot(xck) + xck.dot(ak_mat) - \
    (1-1/gam**2)*xck.dot(bk_mat).dot(bk_mat.T.dot(xck)) +\
    ck_mat.T.dot(ck_mat)
print('red c-ric-res', np.linalg.norm(riccres))

ricores = ak_mat.dot(xok) + xok.dot(ak_mat.T) - \
    (1-1/gam**2)*xok.dot(ck_mat.T).dot(ck_mat.dot(xok)) + \
    bk_mat.dot(bk_mat.T)

print('red o-ric-res', np.linalg.norm(ricores))

# Recompute the red Grams
rcxok = solve_continuous_are(ak_mat.T, scfc*ck_mat.T, bk_mat.dot(bk_mat.T),
                             np.eye(ck_mat.shape[0]))
rcxokres = ak_mat.dot(rcxok) + rcxok.dot(ak_mat.T) - \
    (1-1/gam)*(1+1/gam)*rcxok.dot(ck_mat.T).dot(ck_mat.dot(rcxok)) + \
    bk_mat.dot(bk_mat.T)
# print(np.linalg.norm(riccres))
# print(np.linalg.norm(ricores))
print('rc-o-ric-res|', np.linalg.norm(rcxokres))

rcxck = solve_continuous_are(ak_mat, scfc*bk_mat, ck_mat.T.dot(ck_mat),
                             np.eye(bk_mat.shape[1]))
rcxckres = ak_mat.T.dot(rcxck) + rcxck.dot(ak_mat) - \
    (1-1/gam)*(1+1/gam)*rcxck.dot(bk_mat).dot(bk_mat.T@rcxok) + ck_mat.T@ck_mat
print('rc-c-ric-res|', np.linalg.norm(rcxokres))

zk = np.linalg.inv(np.eye(xck.shape[0])
                   - 1./gam**2*xok.dot(xck))
# amatk = (ak_mat
obsvakmat = (ak_mat
             - (1. - 1./gam**2)*np.dot(np.dot(xok, ck_mat.T), ck_mat)
             - np.dot(bk_mat, np.dot(bk_mat.T, xck).dot(zk)))
evls = np.linalg.eigvals(obsvakmat)
# obs_ck = -np.dot(bk_mat.T.dot(xck), zk)
print('`Ak-cl`-evls:', evls)
# xck, xok = rcxck, rcxok
# zk = np.linalg.inv(np.eye(xck.shape[0])
#                    - 1./gam**2*xok.dot(xck))
# obsvakmat = (ak_mat
#              - (1. - 1./gam**2)*np.dot(np.dot(xok, ck_mat.T), ck_mat)
#              - np.dot(bk_mat, np.dot(bk_mat.T, xck).dot(zk)))
# evls = np.linalg.eigvals(obsvakmat)
# print('`rc-Ak-cl`-evls:', evls)

fullrmmat = np.vstack([np.hstack([obsvakmat, (xok@ck_mat.T)@ck_mat]),
                       np.hstack([-bk_mat@((bk_mat.T@xok)@zk), ak_mat])])
evls = np.linalg.eigvals(fullrmmat)
print('`cl`-evls:', evls)
