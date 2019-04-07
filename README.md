LQG-BT for linearized Navier-Stokes Equations
---------------------------------------------

Python module for application of LQG-balanced truncation for low-order controllers for the stabilization of Navier-Stokes

As an example we consider the stabilization of the cylinder wake at moderate Reynoldsnumbers via distributed control and observation.

Documentation goes [here](http://lqgbt-for-flow-stabilization.readthedocs.org/en/latest/).

Dependencies:

* numpy (1.13.1) and scipy (1.1.0)
* dolfin (2018.1.0)

and my home-brew modules that are available via github.

* dolfin_navier_scipy
* sadptprj_riclyap_adi
* distr_control_fenics

The branch `deps-included` already contains my home-brew modules.
