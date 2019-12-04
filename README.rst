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

.. image:: https://dev.azure.com/OpenSourceEconomics/estimagic/_apis/build/status/OpenSourceEconomics.estimagic?branchName=master
    :target: https://dev.azure.com/OpenSourceEconomics/estimagic/_build/latest?definitionId=1&branchName=master


Introduction
============

Estimagic is a Python package that helps to build high-quality and user friendly
implementations of (structural) econometric models.

===============   =============
|left_text|       |db_gif|
===============   =============

.. |db_gif| image:: docs/source/images/dashboard.gif
  :scale: 50 %

.. |left_text| replace::
  It is designed with large structural models in mind. However, it is also useful for any
  other estimator that numerically minimizes or maximizes a criterion function (Extremum
  Estimator). Examples are maximum likelihood estimation, generalized method of moments,
  method of simulated moments and indirect inference.

Estimagic is in a very early stage and should not be used for major projects yet.
However, we do encourage interested users to try it out, report bugs and provide
feedback.


Credits
=======

Estimagic is designed and maintained by Janos Gabler (`janosg
<https://github.com/janosg>`_).

However, it has been a collaborative project right from the start.

In particular we would like to thank:

- Klara Röhrl (`roecla <https://github.com/roecla>`_) for writing most of the dashboard
  code, contributing plots and doing code reviews.
- Tobias Raabe (`tobiasraabe <https://github.com/tobiasraabe>`_) for setting up the
  continuous integration and contributing several bug-fixes and improvements.
- Max Blesch (`MaxBlesch <https://github.com/MaxBlesch>`_) for contributing code for
  numerical integration and calculation of standard errors for GMM and MSM estimators
- Linda Maokomatanda (`lindamaok899 <https://github.com/lindamaok899>`_) for
  contributing code for the calculation of likelihood standard errors
- Moritz Mendel (`mo2561057 <https://github.com/mo2561057>`_) for wrapping the pounders
  algorithm from TAO

If you want to find your name here as well, please contact us or browse through our
Issues and submit a Pull Request.


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

    @Unpublished{Gabler2019,
      Title  = {A Python Tool for the Estimation of (Structural) Econometric Models.},
      Author = {Janos Gabler},
      Year   = {2019},
      Url    = {https://github.com/OpenSourceEconomics/estimagic}
    }


Roadmap
=======

Estimagic is still in alpha status and the API might still change. We will try
to keep the API more stable When we switch versions `0.0.x` to `0.x.x`. Until
then we want to achieve the following things:

- Fully wrapping scipy.optimize and pygmo, including pygmo's capability for
  parallelized optimizers
- Convenient and robust functions to calculate covariance matrices of
  Maximum Likelihood, GMM, MSM and Indirect Inference estimates.

Afterwards we want to also implement the following extensions:

- Support for general nonlinear constraints for some optimizers
- Confidence intervals based on the profile-likelihood method

