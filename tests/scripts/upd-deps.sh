#/bin/bash

rsync -avC ~/software/gits/mypys/dolfin_navier_scipy/dolfin_navier_scipy/dolfin_to_sparrays.py dolfin_navier_scipy/
rsync -avC ~/software/gits/mypys/dolfin_navier_scipy/dolfin_navier_scipy/stokes_navier_utils.py dolfin_navier_scipy/
rsync -avC ~/software/gits/mypys/dolfin_navier_scipy/dolfin_navier_scipy/data_output_utils.py dolfin_navier_scipy/
rsync -avC ~/software/gits/mypys/dolfin_navier_scipy/dolfin_navier_scipy/problem_setups.py dolfin_navier_scipy/

rsync -avC ~/software/gits/mypys/sadptprj_riclyap_adi/sadptprj_riclyap_adi/lin_alg_utils.py sadptprj_riclyap_adi/
rsync -avC ~/software/gits/mypys/sadptprj_riclyap_adi/sadptprj_riclyap_adi/proj_ric_utils.py sadptprj_riclyap_adi/
rsync -avC ~/software/gits/mypys/sadptprj_riclyap_adi/sadptprj_riclyap_adi/bal_trunc_utils.py sadptprj_riclyap_adi/

rsync -avC ~/software/gits/mypys/distr_control_fenics/distr_control_fenics/cont_obs_utils.py distr_control_fenics/
