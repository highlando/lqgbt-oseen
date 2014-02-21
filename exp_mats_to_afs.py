from exp_cylinder_mats import lqgbt
import numpy as np

charlen = 0.15  # cylinder
relist = np.array([100, 200, 300, 400, 500], dtype=float)
nulist = charlen/relist
Nlist = [0, 1, 2, 3, 4]
for nu in nulist:
    for N in Nlist:
        lqgbt(problemname='cylinderwake', N=N, nu=nu, plain_bt=True,
              savetomatfiles=True)
