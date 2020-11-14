%% CYLINDERWAKE_RE20_ROM_CONTROL
% Script for computing the ROM and controller for:
%
%   cylinderwake_Re20.0_gamma1.0_NV41700_Bbcc_C31_palpha1e-05__mats.mat

clear;
close all;
clc;

% Get rootpath of runnning script.
[rootpath, name, ~] = fileparts(mfilename('fullpath'));

fprintf(1, ['SCRIPT: ' name '\n']);
fprintf(1, ['========' repmat('=', 1, length(name)) '\n']);
fprintf(1, '\n');


%% Setup problem data.
fprintf(1, 'Setup problem data.\n');
fprintf(1, '-------------------\n');

load('/scratch/owncloud-gwdg/mpi-projects/18-hinf-lqgbt/data/cylinderwake_Re20.0_gamma1.0_NV41700_Bbcc_C31_palpha1e-05__mats.mat');

% System sizes.
st = size(mmat, 1);
nz = size(jmat, 1);
m  = size(bmat, 2);
p  = size(cmat, 1);

fprintf(1, '\n');


%% Load Riccati results.
fprintf(1, 'Load Riccati results.\n');
fprintf(1, '---------------------\n');

load('/scratch/owncloud-gwdg/mpi-projects/18-hinf-lqgbt/results/cylinderwake_re20_hinf');

fprintf(1, '\n');


%% Compute ROM.
fprintf(1, 'Compute ROM.\n');
fprintf(1, '------------\n');

% SR method.
[U, S, V] = svd(outRegulator.Z' * (mmat * outFilter.Z), 'econ');

hsv      = diag(S);
errtol   = 1 / gam;
r        = length(hsv) + 1;
err      = 0;
errbound = 0;
figure(1);
r = 40;
semilogy(hsv(1:r), 'x');

S = sparse(1:r, 1:r, 1 ./ sqrt(hsv(1:r)));
W = S * (outRegulator.Z * U(: , 1:r))';
T = (outFilter.Z * V(: , 1:r)) * S;

% Compute ROM.
Er = W * (mmat * T);
Ar = W * (amat * T);
Br = W * bmat;
Cr = cmat * T;

fprintf(1, '\n');


%% Compute controller.
fprintf(1, 'Compute controller.\n');
fprintf(1, '-------------------\n');

scale = sqrt(gam^2 - 1) / gam;
Xinf  = care(Ar, scale * Br, Cr' * Cr, eye(2));
Yinf  = care(Ar', scale * Cr', Br * Br', eye(6));
Zinf  = eye(r) - (1 / gam^(2)) * (Xinf * Yinf);

Ak = Ar - scale^2 * (Br * (Br' * Xinf)) - Zinf \ (Yinf * (Cr' * Cr));
Bk = Zinf \ (Yinf * Cr');
Ck = -Br' * Xinf;

fprintf(1, '\n');


%% Save results.
fprintf(1, 'Save results.\n');
fprintf(1, '-------------\n');

% save('results/cylinderwake_re20_rom_control.mat', ...
%     'Ar', 'Br', 'Cr', 'Ak', 'Bk', 'Ck', 'gam', ...
%     '-v7.3');

fprintf(1, '\n');


%% Finished script.
fprintf(1, 'FINISHED SCRIPT.\n');
fprintf(1, '================\n');
fprintf(1, '\n');

