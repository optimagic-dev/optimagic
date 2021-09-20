=========
estimagic
=========

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

Tools
=====

Optimization
------------

- estimagic wraps all algorithms from *scipy.optimize* and many more become
  available when installing optional dependencies. See :ref:`list_of_algorithms`
- estimagic can automatically implement many types of constraints via
  reparametrization, with any optimizer that supports bounds. See :ref:`constraints`
- estimagic encourages name-based parameters handling. Parameters are specified
  as pandas DataFrames with any kind of single or MultiIndex. See :ref:`params`.
- The complete history of parameters and function evaluations can be saved in a
  database for maximum reproducibility. See `How to use logging`_
- The progress of the optimization is displayed in real time via an
  interactive dashboard. See :ref:`dashboard`.


.. _How to use logging: how_to_guides/optimization/how_to_use_logging.ipynb


  .. image:: _static/images/dashboard.gif
    :scale: 80 %
    :align: center

Estimation and Inference
------------------------

- You can estimate a model using method of simulated moments (MSM), calculate standard
  errors and do sensitivity analysis with just one function call.
  See `MSM Tutorial`_
- Asymptotic standard errors for maximum likelihood estimation.
- estimagic also provides bootstrap confidence intervals and standard errors.
  Of course the bootstrap procedures are parallelized.

.. _MSM Tutorial: getting_started/first_msm_estimation_with_estimagic.ipynb


Numerical differentiation
-------------------------
- estimagic can calculate precise numerical derivatives using `Richardson extrapolations
  <https://en.wikipedia.org/wiki/Richardson_extrapolation>`_.
- Function evaluations needed for numerical derivatives can be done in parallel
  with pre-implemented or user provided batch evaluators.


Structure of the documentation
==============================

.. raw:: html

    <div class="container" id="index-container">
        <div class="row">
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="getting_started/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/light-bulb.svg" class="card-img-top"
                             alt="getting_started-icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">Getting Started</h5>
                            <p class="card-text">
                                New users of estimagic should read this first.
                            </p>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="how_to_guides/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/book.svg" class="card-img-top"
                             alt="how-to guides icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">How-to Guides</h5>
                            <p class="card-text">
                                Detailed instructions for specific and advanced tasks.
                            </p>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="explanations/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/books.svg" class="card-img-top"
                             alt="explanations icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">Explanations</h5>
                            <p class="card-text">
                                Background information to key topics
                                underlying the package.
                            </p>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="reference_guides/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/coding.svg" class="card-img-top"
                             alt="reference guides icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">API Reference</h5>
                            <p class="card-text">
                                Overview of functions and modules as well as
                                implementation details.
                            </p>
                        </div>
                    </div>
                 </a>
            </div>
        </div>
    </div>

.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started/index
   how_to_guides/index
   explanations/index
   reference_guides/index

**Useful links for search:** :ref:`genindex` | :ref:`modindex` | :ref:`search`

.. toctree::
  :maxdepth: 1
  :hidden:

  development/index
  changes
  credits
