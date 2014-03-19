.. lqgbt-nse documentation master file, created by
   sphinx-quickstart on Fri Mar 14 15:17:45 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to lqgbt-nse's documentation!
=====================================

This is a Python module for application of LQG-balanced truncation for low-order controllers for the stabilization of Navier-Stokes

As an example we consider the stabilization of the cylinder wake at moderate Reynoldsnumbers via distributed control and observation.

To get started, install all dependencies, download the code from the linked github repository, create the subdirectory `data`, and run `compscript.py`. Since the problem already is unstable, the Newton-ADI iterations will not simply converge. You will need to compute stabilizing initial guesses first. Therefore, uncomment the respective line in the header of `compscript.py`.

Dependencies:

* `numpy and scipy (v0.13) <http://scipy.org/>`_
* `dolfin (v1.3) <http://fenicsproject.org/>`_
* `dolfin_navier_scipy <https://github.com/highlando/dolfin_navier_scipy/releases/tag/v1.0-lqgbtpaper>`_
* `sadptprj_riclyap_adi <https://github.com/highlando/sadptprj_riclyap_adi/releases/tag/v1.0-lqgbtpaper>`_
* `distr_control_fenics <https://github.com/highlando/distr_control_fenics/releases/tag/v1.0-lqgbtpaper>`_

Alternatively:
If you have dolfin and scipy installed, all you need is in tar file attached to the github release `v1.0-lqgbtpaper <https://github.com/highlando/lqgbt-oseen/releases/tag/v1.0-lqgbtpaper>`_.

Contents:

.. toctree::
   :maxdepth: 2

   code


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

