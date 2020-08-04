=========
Estimagic
=========

Estimagic is a Python package that helps to build high-quality and user friendly
implementations of (structural) econometric models. It is designed with large structural
models in mind, but also "scales down" to simple cases.

Estimagic provides tools for nonlinear optimization, numerical differentiation and
statistical inference.


Structure of the Documentation
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
                                New users of estimagic should read this first
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
                            <h5 class="card-title">Reference Guides</h5>
                            <p class="card-text">
                                Overview of functions and modules as well as
                                implementation details
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



Highlights
==========

Optimization
------------

- Unified interface to a large number of local and global optimization algorithms. Of
  course we have all algorithms from `scipy.optimize` but many more become available
  when you install optional dependencies.
- Parameters are specified as pandas DataFrames that can have any kind of single or
  MultiIndex
- Many types of constraints can be used with any optimizer that supports simple box
  constraints. Constraints are specified using parameter names, not positions.
- The complete history of parameters and function evaluations is saved in a database for
  maximum reproducibility. They are also displayed in real time in an interactive
  dashboard.


.. image:: _static/images/dashboard.gif
  :scale: 60 %
  :align: center


Numerical Differentiation
-------------------------

- Calculate precise numerical derivatives using `Richardson extrapolations
  <https://en.wikipedia.org/wiki/Richardson_extrapolation>`_.
- All function evaluations needed for numerical derivatives can be done in parallel with
  pre-implemented or user provided batch evaluators.


Statistical Inference
---------------------

- Asymptotic standard errors for maximum likelihood an method of simulated moments
- Bootstrap confidence intervals and standard errors for nonlinear estimators. Of course
  the bootstrap procedures are parallelized.


Additional Topics
=================

.. toctree::
   :maxdepth: 1

   contributing/index
   changes
   credits


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
