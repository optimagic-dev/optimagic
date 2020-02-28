Bootstrap Inference
========================

.. currentmodule:: estimagic.inference.bootstrap

In this section we document how to use the bootstrap to approximate the distribution
of statistics of interest on a given sample.

The main idea of the bootstrap is to combine three steps: Firstly, generate bootstrap
samples by drawing from the original data set with replacement. Secondly, calculating
the statistic of interest from these resampled data, and finally, using the various
estimates to draw inference on the distribution of the true statistic of interest.

The main function of this module takes care of all of these three steps and returns a
table containing means, standard errors and confidence intervals for estimated
parameters.

.. autofunction:: bootstrap

However, the steps can also be called separately by the user, as described in what
follows.

.. currentmodule:: estimagic.inference.bootstrap_samples

The module implements the first step of drawing  B bootstrap samples in two separate
ways. The first, and default, method, is to simply draw observations from the original
dataset with replacement. The second method is a cluster robust bootstrap, often called
pairs cluster bootstrap, that draws clusters, defined by observations with the same
value of some specified stratum variable, from the original dataset with replacement.
Sample drawing is implemented by drawing seeds. Seeds can be re-used to calculate
multiple statistics on the same data points.

.. autofunction:: get_seeds

To be as memory-efficient as possible, the actual drawing of data points is by default
deferred to the calculation of statistics. However, it is possible to get a list of
drawn resampled datasets using

.. autofunction:: get_bootstrap_samples

.. currentmodule:: estimagic.inference.bootstrap_estimates

The calculation of statistics is handled by the following function:

.. autofunction:: get_bootstrap_estimates

.. currentmodule:: estimagic.inference.bootstrap

To calculate mean, standard deviation, and a confidence interval of the statistic of
interest, we can use

.. autofunction:: get_results_table

.. currentmodule:: estimagic.inference.bootstrap_ci

There is also a standalone function for calculating bootstrap intervals.

.. autofunction:: compute_ci

A tutorial on how to use the bootstrap in estimagic can be found in the following
jupyter notebook:

.. toctree::
    :maxdepth: 1

    bootstrap_tutorial.ipynb

Here you can learn more about what confidence interval methods are supported:

.. toctree::
    :maxdepth: 1

    bootstrap_ci.rst
