Hinf-LQG-BT for linearized Navier-Stokes Equations
---------------------------------------------

Python module for application of (Hinf-)LQG-balanced truncation for low-order controllers for the stabilization of Navier-Stokes equations.

As an example we consider the stabilization of the cylinder wake at moderate Reynoldsnumbers via boundary control and distributed observation.

Documentation goes [here](http://lqgbt-for-flow-stabilization.readthedocs.org/en/latest/).

Installation (including all dependencies, **except from** *FEniCS*, *mat73*):

```sh
pip install .
```

Dependencies:

* `numpy` (tested with 1.13.1) and `scipy` (tested with 1.1.0)
* `dolfin` (python interface to FEniCS, tested with 2019.2.0 and 2018.1.0)

and my home-brew modules that are available via pip 

```bash
pip install dolfin_navier_scipy
pip install sadptprj_riclyap_adi
pip install distr_control_fenics 
```

and on github:

* [dolfin_navier_scipy](https://github.com/highlando/dolfin_navier_scipy)
* [sadptprj_riclyap_adi](https://github.com/highlando/sadptprj_riclyap_adi)
* [distr_control_fenics](https://github.com/highlando/distr_control_fenics)
