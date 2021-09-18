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


How to read the algorithms documentation
========================================


Below we document the supported algorithms. The documentation refers to the internal
optimizer interface (see :ref:`internal_optimizer_interface`). However, those functions
are not intended to be called by the user. Instead users should call them by calling
``maximize`` or ``minimize`` with the corresponding algorithm argument.

The arguments ``criterion_and_derivative``, ``x``, ``lower_bound`` and ``upper_bound``
of the signatures below should simply be ignored.

The other arguments can be set as ``algo_options`` when calling ``maximize`` or
``minimize``.


.. _list_of_algorithms:

Available Optimizers
====================


Optimizers from scipy
---------------------


.. _scipy_algorithms:


estimagic supports most ``scipy`` algorithms. You do not need to install additional
dependencies to use them:

.. dropdown::  scipy_lbfgsb

    .. autofunction:: estimagic.optimization.scipy_optimizers.scipy_lbfgsb


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

To use DF-OLS you need to have
`the dfols package <https://tinyurl.com/y5ztv4yc>`_ installed.
BOBYQA requires `the pybobyqa package <https://tinyurl.com/y67foub7>`_ .

.. dropdown::  nag_dfols

    .. autofunction:: estimagic.optimization.nag_optimizers.nag_dfols

.. dropdown::  nag_pybobyqa

    .. autofunction:: estimagic.optimization.nag_optimizers.nag_pybobyqa

The Interior Point Optimizer (ipopt)
------------------------------------

estimagic's support for the Interior Point Optimizer (:cite:`Waechter2005`,
:cite:`Waechter2005a`, :cite:`Waechter2005b`, :cite:`Nocedal2009`) is built on
`cyipopt <https://cyipopt.readthedocs.io/en/latest/index.html>`_, a Python wrapper
for the `Ipopt optimization package <https://coin-or.github.io/Ipopt/index.html>`_.

.. autofunction:: estimagic.optimization.cyipopt_optimizers.ipopt



**References**

.. bibliography:: ../../refs.bib
    :filter: docname in docnames
    :style: unsrt



.. With ``nlopt`` installed
.. With ``cyipopt`` installed
.. With ``pygmo`` installed
