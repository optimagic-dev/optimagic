.. _algorithms:

========================================================
How to specify algorithms and algorithm specific options
========================================================

The *algorithm* argument
========================

The ``algorithm`` argument can either be string with the name of a algorithm that is
implemented in estimagic, or a function that fulfills the interface laid out in
:ref:`internal_optimizer_interface`.

Which algorithms are available in estimagic depends on the packages a user has
installed. We list all implemented algorithms :ref:`below <list_of_algorithms>`.


The *algo_options* argument
===========================

``algo_options`` is a dictionary with optional keyword arguments that are passed to the
optimization algorithm.

We align the names of all ``algo_options`` across algorithms.

Since some optimizers support many tuning parameters we group some of them using the
first part of their name (e.g. all convergence criteria names start with
``convergence_``).

All option names only contain ``_``. However, to make the group membership more visible,
you can also specify them separating the group with a ``.`` from the rest of the
option's name. For example, if you wanted to set some tuning parameters of ``nag_dfols``
you could specify your ``algo_options`` like this:

.. code-block:: python

    algo_options = {
        "trustregion.threshold_successful": 0.2,
        "trustregion.threshold_very_successful": 0.9,
        "trustregion.shrinking_factor.not_successful": 0.4,
        "trustregion.shrinking_factor.lower_radius": 0.2,
        "trustregion.shrinking_factor.upper_radius": 0.8,
        "convergence.scaled_criterion_tolerance": 0.0,
        "convergence.noise_corrected_criterion_tolerance": 1.1,
    }

Estimagic then automatically replaces the ``.`` with ``_`` before passing them to the
internal optimizer.

However, not all algorithms support all options. Which options are supported and
very specific details of what the options mean are documented for each algorithm.

To make it easier to switch between algorithms, we simply ignore non-supported options
and issue a warning that explains which options have been ignored.

To find more information on ``algo_options`` that more than one algorithm allows for
see :ref:`algo_options`.



.. _list_of_algorithms:

Available Optimizers
====================


.. _list_of_scipy_algorithms:

Optimizers from scipy
---------------------


.. _scipy_algorithms:


estimagic supports most ``scipy`` algorithms. You do not need to install additional
dependencies to use them:

.. dropdown::  scipy_lbfgsb

    Minimize a scalar function of one or more variables using the L-BFGS-B algorithm.

    The optimizer is taken from scipy, which calls the Fortran code written by the
    original authors of the algorithm. The Fortran code includes the corrections
    and improvements that were introduced in a follow up paper.

    lbfgsb is a limited memory version of the original bfgs algorithm, that deals with
    lower and upper bounds via an active set approach.

    The lbfgsb algorithm is well suited for differentiable scalar optimization problems
    with up to several hundred parameters.

    It is a quasi-newton line search algorithm. At each trial point it evaluates the
    criterion function and its gradient to find a search direction. It then approximates
    the hessian using the stored history of gradients and uses the hessian to calculate
    a candidate step size. Then it uses a gradient based line search algorithm to
    determine the actual step length. Since the algorithm always evaluates the gradient
    and criterion function jointly, the user should provide a
    ``criterion_and_derivative`` function that exploits the synergies in the
    calculation of criterion and gradient.

    The lbfgsb algorithm is almost perfectly scale invariant. Thus, it is not necessary
    to scale the parameters.


    - convergence_relative_criterion_tolerance (float): Stop when the relative
      improvement between two iterations is smaller than this.
      More formally, this is expressed as

      .. math::

        \\frac{(f^k - f^{k+1})}{\\max{{|f^k|, |f^{k+1}|, 1}}} \\leq
        \\text{relative_criterion_tolerance}

    - convergence_absolute_gradient_tolerance (float): Stop if all elements of the
      projected gradient are smaller than this.
    - stopping_max_criterion_evaluations (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count
      this as convergence.
    - stopping_max_iterations (int): If the maximum number of iterations is reached,
      the optimization stops, but we do not count this as convergence.
    - limited_memory_storage_length (int): Maximum number of saved gradients used to
      approximate the hessian matrix.


.. dropdown::  scipy_slsqp

    .. autofunction:: estimagic.optimization.scipy_optimizers.scipy_slsqp


.. dropdown::  scipy_neldermead

    .. autofunction:: estimagic.optimization.scipy_optimizers.scipy_neldermead


.. dropdown::  scipy_powell

    .. autofunction:: estimagic.optimization.scipy_optimizers.scipy_powell


.. dropdown::  scipy_bfgs

    .. autofunction:: estimagic.optimization.scipy_optimizers.scipy_bfgs


.. dropdown::  scipy_conjugate_gradient

    .. autofunction:: estimagic.optimization.scipy_optimizers.scipy_conjugate_gradient


.. dropdown::  scipy_newton_cg

    .. autofunction:: estimagic.optimization.scipy_optimizers.scipy_newton_cg


.. dropdown::  scipy_cobyla

    .. autofunction:: estimagic.optimization.scipy_optimizers.scipy_cobyla


.. dropdown::  scipy_truncated_newton

    .. autofunction:: estimagic.optimization.scipy_optimizers.scipy_truncated_newton


.. dropdown::  scipy_trust_constr

    .. autofunction:: estimagic.optimization.scipy_optimizers.scipy_trust_constr



.. _tao_algorithms:

Optimizers from the Toolkit for Advanced Optimization (TAO)
-----------------------------------------------------------

At the moment, estimagic only supports
`TAO's <https://www.anl.gov/mcs/tao-toolkit-for-advanced-optimization>`_
POUNDERs algorithm.

The `POUNDERs algorithm <https://www.mcs.anl.gov/papers/P5120-0414.pdf>`_
by Stefan Wild is tailored to minimize a non-linear sum of squares
objective function. Remember to cite :cite:`Wild2015` when using POUNDERs in
addition to estimagic.

To use POUNDERs you need to have
`petsc4py <https://pypi.org/project/petsc4py/>`_ installed.

.. dropdown::  tao_pounders

    .. autofunction:: estimagic.optimization.tao_optimizers.tao_pounders



.. _nag_algorithms:


Optimizers from the Numerical Algorithms Group (NAG)
----------------------------------------------------

Currently, estimagic supports the
`Derivative-Free Optimizer for Least-Squares Minimization (DF-OLS)
<https://numericalalgorithmsgroup.github.io/dfols/>`_ and
`BOBYQA <https://numericalalgorithmsgroup.github.io/pybobyqa/>`_
by the `Numerical Algorithms Group <https://www.nag.com/>`_.

To use DF-OLS you need to have `the dfols package
<https://tinyurl.com/y5ztv4yc>`_ installed (``pip install DFO-LS``). BOBYQA
requires `the pybobyqa package <https://tinyurl.com/y67foub7>`_ (``pip install
Py-BOBYQA``).

.. dropdown::  nag_dfols

    .. autofunction:: estimagic.optimization.nag_optimizers.nag_dfols

.. dropdown::  nag_pybobyqa

    .. autofunction:: estimagic.optimization.nag_optimizers.nag_pybobyqa



.. _pygmo_algorithms:

PYGMO2 Optimizers
------------------

Please cite :cite:`Biscani2020` in addition to estimagic when using pygmo.
estimagic supports the following `pygmo2 <https://esa.github.io/pygmo2>`_
optimizers.

.. dropdown::  pygmo_gaco

    .. autofunction:: estimagic.optimization.pygmo_optimizers.pygmo_gaco

.. dropdown::  pygmo_bee_colony

    .. autofunction:: estimagic.optimization.pygmo_optimizers.pygmo_bee_colony

.. dropdown::  pygmo_de

    .. autofunction:: estimagic.optimization.pygmo_optimizers.pygmo_de

.. dropdown::  pygmo_sea

    .. autofunction:: estimagic.optimization.pygmo_optimizers.pygmo_sea

.. dropdown::  pygmo_sga

    .. autofunction:: estimagic.optimization.pygmo_optimizers.pygmo_sga

.. dropdown::  pygmo_sade

    .. autofunction:: estimagic.optimization.pygmo_optimizers.pygmo_sade


.. dropdown::  pygmo_cmaes

    .. autofunction:: estimagic.optimization.pygmo_optimizers.pygmo_cmaes

.. dropdown::  pygmo_simulated_annealing

    .. autofunction:: estimagic.optimization.pygmo_optimizers.pygmo_simulated_annealing

.. dropdown::  pygmo_pso

    .. autofunction:: estimagic.optimization.pygmo_optimizers.pygmo_pso

.. dropdown::  pygmo_pso_gen

    .. autofunction:: estimagic.optimization.pygmo_optimizers.pygmo_pso_gen

.. dropdown::  pygmo_mbh

    .. autofunction:: estimagic.optimization.pygmo_optimizers.pygmo_mbh

.. dropdown::  pygmo_xnes

    .. autofunction:: estimagic.optimization.pygmo_optimizers.pygmo_xnes

.. dropdown::  pygmo_gwo

    .. autofunction:: estimagic.optimization.pygmo_optimizers.pygmo_gwo

.. dropdown::  pygmo_compass_search

    .. autofunction:: estimagic.optimization.pygmo_optimizers.pygmo_compass_search

.. dropdown::  pygmo_ihs

    .. autofunction:: estimagic.optimization.pygmo_optimizers.pygmo_ihs

.. dropdown::  pygmo_de1220

    .. autofunction:: estimagic.optimization.pygmo_optimizers.pygmo_de1220


.. _ipopt_algorithm:

The Interior Point Optimizer (ipopt)
------------------------------------

estimagic's support for the Interior Point Optimizer (:cite:`Waechter2005`,
:cite:`Waechter2005a`, :cite:`Waechter2005b`, :cite:`Nocedal2009`) is built on
`cyipopt <https://cyipopt.readthedocs.io/en/latest/index.html>`_, a Python wrapper
for the `Ipopt optimization package <https://coin-or.github.io/Ipopt/index.html>`_.

To use ipopt, you need to have `cyipopt installed
<https://cyipopt.readthedocs.io/en/latest/index.html>`_ (``conda install
cyipopt``).


.. dropdown:: ipopt

    .. autofunction:: estimagic.optimization.cyipopt_optimizers.ipopt


The NLOPT Optimizers (nlopt)
-----------------------------

estimagic supports the following `NLOPT <https://nlopt.readthedocs.io/en/latest/>`_
algorithms. Please add the `appropriate citations
<https://nlopt.readthedocs.io/en/latest/Citing_NLopt/>`_ in addition to estimagic when
using an NLOPT algorithm. To install nlopt run ``conda install nlopt``.

.. dropdown:: nlopt_bobyqa

    Minimize a scalar function using the BOBYQA algorithm.

    The implementation is derived from the BOBYQA subroutine of M. J. D. Powell.

    The algorithm performs derivative free bound-constrained optimization using
    an iteratively constructed quadratic approximation for the objective function.
    Due to its use of quadratic appoximation, the algorithm may perform poorly
    for objective functions that are not twice-differentiable.

    For details see:
    M. J. D. Powell, "The BOBYQA algorithm for bound constrained optimization
    without derivatives," Department of Applied Mathematics and Theoretical
    Physics, Cambridge England, technical report NA2009/06 (2009).

    ``nlopt_bobyqa`` supports the following ``algo_options``:

    - convergence.relative_params_tolerance (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - convergence.absolute_params_tolerance (float): Stop when the absolute movement
      between parameter vectors is smaller than this.
    - convergence.relative_criterion_tolerance (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - convergence.absolute_criterion_tolerance (float): Stop when the change of the
      criterion function between two iterations is smaller than this.
    - stopping.max_criterion_evaluations (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.


.. dropdown:: nlopt_neldermead

    .. autofunction:: estimagic.optimization.nlopt_optimizers.nlopt_neldermead

.. dropdown:: nlopt_praxis

    .. autofunction:: estimagic.optimization.nlopt_optimizers.nlopt_praxis

.. dropdown:: nlopt_cobyla

    .. autofunction:: estimagic.optimization.nlopt_optimizers.nlopt_cobyla

.. dropdown:: nlopt_sbplx

    .. autofunction:: estimagic.optimization.nlopt_optimizers.nlopt_sbplx

.. dropdown:: nlopt_newuoa

    .. autofunction:: estimagic.optimization.nlopt_optimizers.nlopt_newuoa

.. dropdown:: nlopt_tnewton

    .. autofunction:: estimagic.optimization.nlopt_optimizers.nlopt_tnewton

.. dropdown:: nlopt_lbfgs

    .. autofunction:: estimagic.optimization.nlopt_optimizers.nlopt_lbfgs

.. dropdown:: nlopt_ccsaq

    .. autofunction:: estimagic.optimization.nlopt_optimizers.nlopt_ccsaq

.. dropdown:: nlopt_mma

    .. autofunction:: estimagic.optimization.nlopt_optimizers.nlopt_mma

.. dropdown:: nlopt_var

    .. autofunction:: estimagic.optimization.nlopt_optimizers.nlopt_var

.. dropdown:: nlopt_slsqp

    .. autofunction:: estimagic.optimization.nlopt_optimizers.nlopt_slsqp

.. dropdown:: nlopt_direct

    .. autofunction:: estimagic.optimization.nlopt_optimizers.nlopt_direct

.. dropdown:: nlopt_esch

    .. autofunction:: estimagic.optimization.nlopt_optimizers.nlopt_esch

.. dropdown:: nlopt_isres

    .. autofunction:: estimagic.optimization.nlopt_optimizers.nlopt_isres

.. dropdown:: nlopt_crs2_lm

    .. autofunction:: estimagic.optimization.nlopt_optimizers.nlopt_crs2_lm


**References**

.. bibliography:: ../../refs.bib
    :labelprefix: algo_
    :filter: docname in docnames
    :style: unsrt
