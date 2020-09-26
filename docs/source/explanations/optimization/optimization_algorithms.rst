.. _algorithms:

====================================
Optimization Algorithms in Estimagic
====================================

The *algorithm* Argument
========================

The ``algorithm`` argument can either be string with the name of a algorithm that is
implemented in estimagic, or a function that fulfills the
:ref:`internal_optimizer_interface`.


Which algorithms are available in estimagic depends on the packages a user has
installed.



The *algo_options* Argument
===========================

``algo_options`` is a dictionary with optional keyword arguments that are passed to the
optimizer.

We align the names of all ``algo_options`` across algorithms. However, not all
algorithms support all options. Which options are supported and very specific details of
what the options mean are documented for each algorithm below.

To make it easier to switch between algorithms, we simply ignore non-supported options
and issue a warning that explains which options have been ignored.



How to Read the Algorithms Documentation
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


Supported Algorithms
====================


Without Optional Dependencies
-----------------------------

*********************
The scipy optimizers
*********************

estimagic supports all of scipy's algorithms except for the following algorithms that
require the specification of the Hessian:

- dogleg
- trust-ncg
- trust-exact
- trust-krylov

The following arguments are not supported as part of ``algo_options``:

- ``disp``
    If set to True would print a convergence message.
    In estimagic it's always set to its default False.
    Refer to estimagic's result dictionary's "success" entry for the convergence
    message.
- ``return_all``
    If set to True, a list of the best solution at each iteration is returned.
    In estimagic it's always set to its default False.
    Use estimagic's database and dashboard instead to explore your criterion and
    algorithm.
- ``tol``
    This argument of minimize (not an options key) is passed as different types of
    tolerance (gradient, parameter or criterion, as well as relative or absolute)
    depending on the selected algorithm. We require the user to explicitely input
    the tolerance criteria or use our defaults instead.
- ``args``
    This argument of minimize (not an options key) is partialed into the function
    for the user. Specify ``criterion_kwargs`` in ``maximize`` or ``minimize`` to
    achieve the same behavior.
- ``callback``
    This argument would be called after each iteration and the algorithm would terminate
    if it returned True.

.. note::
    Scipy's COBYLA, SLSQP and trust-constr support general non linear constraints in
    principle. However, for the moment they are not supported.


Algorithms:
*********************


.. automodule:: estimagic.optimization.scipy_optimizers
    :members:


With ``petsc4py`` installed
----------------------------

.. automodule:: estimagic.optimization.tao_optimizers
   :members:


With ``pybobyqa`` installed
----------------------------

`pybobyqa <https://numericalalgorithmsgroup.github.io/pybobyqa/>`_ is provided by
the `Numerical Algorithms Group <https://www.nag.com/>`_.

Remember to cite :cite:`Powell2009` and :cite:`Cartis2018` when using pybobyqa in
addition to estimagic. If you take advantage of the ``seek_global_optimum`` option,
cite :cite:`Cartis2018a` additionally.

The following arguments are not supported as part of ``algo_options``:

- ``scaling_within_bounds``
- ``init.run_in_parallel``
- ``do_logging``, ``print_progress`` and all their advanced options.
  Use estimagic's database and dashboard instead to explore your criterion and algorithm.

.. autofunction:: estimagic.optimization.nag_optimizers.nag_pybobyqa



With ``nlopt`` installed
-------------------------


With ``cyipopt`` installed
---------------------------


With ``pygmo`` installed
--------------------------



References:
==============

.. bibliography:: ../../refs.bib
    :filter: docname in docnames
    :style: unsrt
