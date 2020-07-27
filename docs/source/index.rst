=========
Estimagic
=========

Estimagic is a Python package that helps to build high-quality and user friendly
implementations of (structural) econometric models. It is designed with large structural
models in mind, but also "scales down" to simpler use cases.

Estimagic provides tools for nonlinear optimization, numerical differentiation and
statistical inference.

Optimization
------------

- Unified interface to a large number of local and global optimization algorithms. Of
  course we have all algorithms from `scipy.optimize` but many more become available
  when you install optional dependencies.
- Efficient reparametrizations make it possible to many types of constraints with any
  algorithm that supports simple box constraints.
- The constraints are specified with a very intuitive interface and users can completely
  abstract from how they are implemented under the hood.
- Parameters are specified as pandas DataFrames that can have any kind of single or
  MultiIndex
- The complete history of parameters and function evaluations is saved in a database.
- An interactive Dashboard allows to monitor the optimization in real time.


.. image:: images/dashboard.gif
  :scale: 21 %


Differentiation
---------------

- Calculate precise numerical derivatives using `Richardson extrapolations
  <https://en.wikipedia.org/wiki/Richardson_extrapolation>`_.
- All function evaluations needed for numerical derivatives can be done in parallel with
  pre-implemented or user provided batch evaluators.


Statistical Inference
---------------------

- Asymptotic standard errors for maximum likelihood an method of simulated moments
- Bootstrap confidence intervals and standard errors for nonlinear estimators. Of course
  the bootstrap procedures are parallelized.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


   installation
   tutorials/index
   how_to_guides/index
   explanations/index
   reference_guides/index

.. toctree::
   :maxdepth: 2
   :caption: Additional Topics

   faq
   api/index
   contributing/index
   changes
   credits


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
