.. _nag_dfols:

Derivative-Free Optimizer for Least-Squares Minimization
=========================================================

`DF-OLS <https://numericalalgorithmsgroup.github.io/dfols/>`_ is provided by
the `Numerical Algorithms Group <https://www.nag.com/>`_.

Remember to cite :cite:`Cartis2018b` when using DF-OLS in addition to estimagic.

The following arguments are not supported as part of ``algo_options``:

- ``scaling_within_bounds``
- ``init.run_in_parallel``
- ``do_logging``, ``print_progress`` and all their advanced options.
  Use estimagic's database and dashboard instead to explore your criterion
  and algorithm.

.. autofunction:: estimagic.optimization.nag_optimizers.nag_dfols


**References**

.. bibliography:: ../../refs.bib
    :labelprefix: dfols
    :filter: docname in docnames
    :style: unsrt
