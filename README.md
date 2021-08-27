Hinf-LQG-BT for linearized Navier-Stokes Equations
---------------------------------------------

Python module for application of (Hinf-)LQG-balanced truncation for low-order controllers for the stabilization of Navier-Stokes equations.

As an example we consider the stabilization of the cylinder wake at moderate Reynoldsnumbers via boundary control and distributed observation.

Documentation of the module that realizes the controllers goes [here](http://lqgbt-for-flow-stabilization.readthedocs.org/en/latest/) -- has not been updated recently.

To reproduce the results of our recent preprint

> [Benner, Heiland, Werner: *Robust output-feedback stabilization for incompressible flows using low-dimensional H∞-controllers*](https://arxiv.org/abs/2103.01608)

see the [`tests/RUNME.md`](tests/RUNME.md)

Installation (including all dependencies, **except from** *FEniCS*, *mat73*):

```sh
pip install .
```

To make *FEniCS* and *mat73* available to you, you may want to use a *docker container* as it can be generated with the `Dockerfile` in this repo.

## Dependencies

* `numpy` (tested with 1.13.1, 1.18.0, 1.19.4) and `scipy` (tested with 1.1.0, 1.5.2)
* `dolfin` (python interface to FEniCS, tested with 2019.2.0 and 2018.1.0)
* [dolfin_navier_scipy](https://github.com/highlando/dolfin_navier_scipy) (tested with `pip install dolfin_navier_scipy==1.1.1`)
* [sadptprj_riclyap_adi](https://github.com/highlando/sadptprj_riclyap_adi) (tested with `sadptprj_riclyap_adi==1.0.3`)
* [distr_control_fenics](https://github.com/highlando/distr_control_fenics) (tested with `distributed_control_fenics==1.0.0`)
