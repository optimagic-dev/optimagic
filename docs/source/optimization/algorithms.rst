.. _list_of_algorithms:

algo_options
============


``algo_options`` is a dictionary with optional keyword arguments that are passed
to the optimizer. This includes tolerances for the termination criteria,
parameters that determine how greedy the optimizer how large the stepsize for
a numerical gradient is. It is the only thing in estimagic that is specific to
each algorithm.


Typically you will leave all of those parameters at their default, unless you
have a very difficult optimization problem. If so, you can find all available
options at the following links (depending on the origin of the algorithm.):

- `pygmo <https://tinyurl.com/y3bgsl4z>`__
- `nlopt <https://tinyurl.com/y3dsmaz7>`__
- `scipy <https://tinyurl.com/y54nmedo>`__


``estimagic`` supports the following algorithms for optimization. To learn more
about the corresponding ``algo_options``, follow the accompanying links.


Supported algorithms
====================

pygmo
-----

- ``"pygmo_gaco"``
- ``"pygmo_de"``
- ``"pygmo_sade"``
- ``"pygmo_de1220"``
- ``"pygmo_ihs"``
- ``"pygmo_pso"``
- ``"pygmo_pso_gen"``
- ``"pygmo_sea"``
- ``"pygmo_sga"``
- ``"pygmo_simulated_annealing"``
- ``"pygmo_bee_colony"``
- ``"pygmo_cmaes"``
- ``"pygmo_xnes"``
- ``"pygmo_nsga2"``
- ``"pygmo_moead"``

nlopt
-----

- ``"nlopt_cobyla"``
- ``"nlopt_bobyqa"``
- ``"nlopt_newuoa"``
- ``"nlopt_newuoa_bound"``
- ``"nlopt_praxis"``
- ``"nlopt_neldermead"``
- ``"nlopt_sbplx"``
- ``"nlopt_mma"``
- ``"nlopt_ccsaq"``
- ``"nlopt_slsqp"``
- ``"nlopt_lbfgs"``
- ``"nlopt_tnewton_precond_restart"``
- ``"nlopt_tnewton_precond"``
- ``"nlopt_tnewton_restart"``
- ``"nlopt_tnewton"``
- ``"nlopt_var2"``
- ``"nlopt_var1"``
- ``"nlopt_auglag"``
- ``"nlopt_auglag_eq"``

scipy
-----

- ``"scipy_L-BFGS-B"``
- ``"scipy_TNC"``
- ``"scipy_SLSQP"``

