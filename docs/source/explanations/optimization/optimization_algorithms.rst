
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



How to Read The Algorithms Documentation
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

.. automodule:: estimagic.optimization.scipy_optimizers
    :members:


With `nlopt` installed
----------------------


With `cyipopt` installed
------------------------


With `pygmo` installed
----------------------


With `petsc4py` installed
-------------------------
