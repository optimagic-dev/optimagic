.. _nag_bobyqa:

Py-BOBYQA
===========

`pybobyqa <https://numericalalgorithmsgroup.github.io/pybobyqa/>`_ is provided by
the `Numerical Algorithms Group <https://www.nag.com/>`_.

Remember to cite :cite:`Powell2009` and :cite:`Cartis2018` when using pybobyqa in
addition to estimagic. If you take advantage of the ``seek_global_optimum`` option,
cite :cite:`Cartis2018a` additionally.

The following arguments are not supported as part of ``algo_options``:

- ``scaling_within_bounds``
- ``init.run_in_parallel``
- ``do_logging``, ``print_progress`` and all their advanced options.
  Use estimagic's database and dashboard instead to explore your criterion and
  algorithm.

.. autofunction:: estimagic.optimization.nag_optimizers.nag_pybobyqa



**References**

.. bibliography:: ../../refs.bib
    :labelprefix: bobyqa
    :filter: docname in docnames
    :style: unsrt
