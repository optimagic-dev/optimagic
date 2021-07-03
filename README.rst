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

estimagic is a Python package that provides high-quality and user-friendly tools
to fit large scale empirical models to data and make inferences about the estimated
model parameters. It is especially suited to solve difficult constrained optimization
problems.

estimagic provides several advantages over similar packages, including a unified
interface that supports a large number of local and global optimization algorithms
and the possibility of monitoring the optimization procedure via a beautiful
interactive dashboard.

estimagic provides tools for nonlinear optimization, numerical differentiation
and statistical inference.


Optimization
------------

- estimagic wraps all algorithms from *scipy.optimize* and many more become
  available when installing optional dependencies.
- estimagic can automatically implement many types of constraints via
  reparametrization, with any optmizer that supports simple box constraints.
- estimagic encourages name-based parameters handling. Parameters are specified
  as pandas DataFrames that can have any kind of single or MultiIndex. This is
  especially useful when specifying constraints.
- The complete history of parameters and function evaluations are saved in a
  database for maximum reproducibility and displayed in real time via an
  interactive dashboard.


.. image:: docs/source/_static/images/dashboard.gif
  :scale: 21 %


Numerical differentiation
-------------------------

- estimagic can calculate precise numerical derivatives using `Richardson extrapolations
  <https://en.wikipedia.org/wiki/Richardson_extrapolation>`_.
- Function evaluations needed for numerical derivatives can be done in parallel
  with pre-implemented or user provided batch evaluators.


Statistical Inference
---------------------

- estimagic provides asymptotic standard errors for maximum likelihood and method
  of simulated moments.
- estimagic also provides bootstrap confidence intervals and standard errors.
  Of course the bootstrap procedures are parallelized.


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

Estimagic is still in beta status and the API might still change. With version `0.2.0`
we start to mark experimental functionality with warnings. Everything that does not
throw such warnings will not change without notice. With version `0.3.0` we will
try to keep the API for full estimation workflows (optimization, inference and
sensitivity analysis) more stable and use deprecation cycles for breaking changes.


Roadmap
=======

Version `0.2.0`
---------------

- Raise warnings for all things that are still considered experimental. Conversely,
  everything that does not throw an experimental warning is from now on stable and
  breaking changes require deprecations.
- `estimagic.optimization.utilities` becomes `estimagic.utilities` (breaking change)
- Return of `first_derivative` becomes a dict (breaking change)
- Logging disabled by default (potentially breaking change)
- Support bootstrap (experimental)
- Support basic sensitivity analysis for moments based estimation (experimental)
- Support scaling of optimization problems (experimental)

Version `0.2.1`
---------------

- Support multi start optimizations as in TikTak (experimental)

Version `0.2.2`
---------------

- Improve packaging and upload on conda-forge
- Split up into several packages to keep runtimes for the test suites manageable
    - estimagic-optimization
    - estimagic-differentiation
    - estimagic-inference
- Internal refactoring of constraints code

Version `0.2.x`
---------------

Improve interfaces of inference functions but still keep them experimental

Version `0.3.0` (getting rid of experimental warnings)
------------------------------------------------------

- Find a good example model that can be estimated with ML, MSM, GMM and II for test
  cases and documentation
- Have stable interfaces for all inference and sensitivity analysis functions
- Make tutorials that show full workflows for each estimation principle
- Promote estimagic via blogposts

Additionally planned Features
-----------------------------

- Wrap nlopt and ipopt
- Implement a flexible toolkit for optimization of noisy functions
- Wrap pygmo
- Add code for numerical second derivatives
- Make dashboard options configurable in GUI
- Improve the appearance of the current Dashboard
- Add a dashboard tab for numerical derivatives
- Add a dashboard tab for bootstrapping
