=========
estimagic
=========

.. image:: https://anaconda.org/OpenSourceEconomics/estimagic/badges/version.svg
   :target: https://anaconda.org/OpenSourceEconomics/estimagic

.. image:: https://anaconda.org/OpenSourceEconomics/estimagic/badges/platforms.svg
   :target: https://anaconda.org/OpenSourceEconomics/estimagic

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-orange.svg
    :target: https://opensource.org/licenses/BSD-3-Clause
    :alt: License

.. image:: https://readthedocs.org/projects/estimagic/badge/?version=latest
    :target: https://estimagic.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/OpenSourceEconomics/estimagic/workflows/Continuous%20Integration%20Workflow/badge.svg?branch=master
    :target: https://github.com/OpenSourceEconomics/estimagic/actions?query=branch%3Amaster

.. image:: https://codecov.io/gh/OpenSourceEconomics/estimagic/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/OpenSourceEconomics/estimagic

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

Introduction
============

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


.. image:: docs/source/_static/images/dashboard.gif
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


Installation
============

The package can be installed via conda. To do so, type the following commands in a
terminal:

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda install -c opensourceeconomics estimagic

The first line adds conda-forge to your conda channels. This is necessary for conda to
find all dependencies of estimagic. The second line installs estimagic and its
dependencies.

Documentation
=============

The documentation is hosted (`on rtd <https://estimagic.readthedocs.io/en/latest/#>`_)

Citation
========

If you use Estimagic for your research, please do not forget to cite it. Many people
worked on this software and you should recognize their effort.

.. code-block::

    @Unpublished{Gabler2020,
      Title  = {A Python Tool for the Estimation of (Structural) Econometric Models.},
      Author = {Janos Gabler},
      Year   = {2020},
      Url    = {https://github.com/OpenSourceEconomics/estimagic}
    }


Warning
=======

Estimagic is still in alpha status and the API might still change. We will try to keep
the API more stable When we switch versions `0.0.x` to `0.x.x`. Until then we want to
achieve the following things:
