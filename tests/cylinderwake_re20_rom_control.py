import numpy as np

import mat73
from scipy.io import loadmat
import sadptprj_riclyap_adi.bal_trunc_utils as btu
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
        'cylinderwake_Re20.0_gamma1.0_NV41700_Bbcc_C31_palpha1e-05__mats.mat'
matdict = loadmat(fname)
mmat = matdict['mmat']
amat = matdict['amat']
cmat = matdict['cmat']
bmat = matdict['bmat']

# % System sizes.
# st = size(mmat, 1);
# nz = size(jmat, 1);
# m  = size(bmat, 2);
# p  = size(cmat, 1);

# %% Load Riccati results.
print('Load Riccati results.')
print('---------------------')

fname = '/scratch/owncloud-gwdg/mpi-projects/18-hinf-lqgbt/results/' + \
        'cylinderwake_re20_hinf.mat'
lmd = mat73.loadmat(fname)
zwo = lmd['outRegulator']['Z']
zwc = lmd['outFilter']['Z']
gam = lmd['gam']

print('Compute ROM.')
print('------------')

# # % SR method.
# [U, S, V] = svd(outRegulator.Z' * (mmat * outFilter.Z), 'econ');

tl, tr, svs = btu.\
    compute_lrbt_transfos(zfc=zwc, zfo=zwo, mmat=mmat,
                          trunck={'threshh': 0.5})
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

ricores = ak_mat.dot(xok) + xok.dot(ak_mat.T) - \
    (1-1/gam**2)*xok.dot(ck_mat.T).dot(ck_mat.dot(xok)) + \
    bk_mat.dot(bk_mat.T)

scfc = np.sqrt(gam**2-1)/gam  # [1]
scfc = np.sqrt(1-1/gam**2)  # [2]
rsxok = solve_continuous_are(ak_mat.T, scfc*ck_mat.T, bk_mat.dot(bk_mat.T),
                             np.eye(ck_mat.shape[0]))

ricrsores = ak_mat.dot(rsxok) + rsxok.dot(ak_mat.T) - \
    (1-1/gam)*(1+1/gam)*rsxok.dot(ck_mat.T).dot(ck_mat.dot(rsxok)) + \
    bk_mat.dot(bk_mat.T)

print(np.linalg.norm(riccres))
print(np.linalg.norm(ricores))
print(np.linalg.norm(ricrsores))

zk = np.linalg.inv(np.eye(xck.shape[0])
                   - 1./gam**2*rsxok.dot(xck))
amatk = (ak_mat
         - (1. - 1./gam**2)*np.dot(np.dot(rsxok, ck_mat.T), ck_mat)
         - np.dot(bk_mat, np.dot(bk_mat.T, xck).dot(zk)))
obs_ck = -np.dot(bk_mat.T.dot(xck), zk)
evls = np.linalg.eigvals(amatk)
plt.figure(2)
plt.plot(np.real(evls), np.imag(evls), 'x')
plt.show()
# print(evls)

# fprintf(1, '\n');
# %% Compute controller.
# fprintf(1, 'Compute controller.\n');
# fprintf(1, '-------------------\n');
# 
# scale = sqrt(gam^2 - 1) / gam;
# Xinf  = icare(Ar, scale * Br, Cr' * Cr);
# Yinf  = icare(Ar', scale * Cr', Br * Br');
# Zinf  = eye(r) - (1 / gam^(2)) * (Xinf * Yinf);
# 
# Ak = Ar - scale^2 * (Br * (Br' * Xinf)) - Zinf \ (Yinf * (Cr' * Cr));
# Bk = Zinf \ (Yinf * Cr');
# Ck = -Br' * Xinf;
# 
# fprintf(1, '\n');
# 
# 
# %% Save results.
# fprintf(1, 'Save results.\n');
# fprintf(1, '-------------\n');
# 
# % save('results/cylinderwake_re20_rom_control.mat', ...
# %     'Ar', 'Br', 'Cr', 'Ak', 'Bk', 'Ck', 'gam', ...
# %     '-v7.3');
# 
# fprintf(1, '\n');
# 
# 
# %% Finished script.
# fprintf(1, 'FINISHED SCRIPT.\n');
# fprintf(1, '================\n');
# fprintf(1, '\n');
