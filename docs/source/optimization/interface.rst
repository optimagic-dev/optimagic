The *minimize* and *maximize* Functions
============================================

.. currentmodule:: estimagic.optimization.optimize

Both optimization functions have the exactly same interface. In fact, ``maximize`` just
switches the sign of your criterion function and calls ``minimize``.

.. autofunction:: minimize
    :noindex:

.. autofunction:: maximize
    :noindex:

The most noteworthy point is that only the ``algo_options`` depend on the optimizer
used. All the rest, including all types of constraints work with all optimizers! The
next pages will describe the more complex arguments of ``maximize`` and ``minimize`` in
more detail.
