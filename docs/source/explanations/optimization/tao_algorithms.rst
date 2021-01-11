.. _tao_algorithms:


Toolkit for Advanced Optimization (TAO) Algorithms
===================================================

At the moment, estimagic only supports
`TAO's <https://www.anl.gov/mcs/tao-toolkit-for-advanced-optimization>_`
POUNDERs algorithm.


POUNDERs
---------

The `POUNDERs algorithm <https://www.mcs.anl.gov/papers/P5120-0414.pdf>`_
by Stefan Wild is tailored to minimize a non-linear sum of squares
objective function. Remember to cite :cite:`Wild2015` when using POUNDERs in
addition to estimagic.

To use POUNDERs you need to have
`petsc4py <https://pypi.org/project/petsc4py/>`_ installed.

.. autofunction:: estimagic.optimization.tao_optimizers.tao_pounders


**References**

.. bibliography:: ../../refs.bib
    :labelprefix: Tao
    :filter: docname in docnames
    :style: unsrt
