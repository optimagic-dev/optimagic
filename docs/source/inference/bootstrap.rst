Bootstrap Inference
========================

Function Descriptions
************************

.. currentmodule:: estimagic.inference.bootstrap

In this section we document how to use the bootstrap to approximate the distribution
of statistics of interest on a given sample.

The main idea of the bootstrap is to combine three steps: Firstly, generate bootstrap
samples by drawing from the original data set with replacement. Secondly, calculating
the statistic of interest from these resampled data, and finally, using the various
estimates to draw inference on the distribution of the true statistic of interest.

The main function of this module takes care of all of these three steps and returns a
table containing means, standard errors and confidence intervals for the estimated
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

There is also a function to calculate only confidence intervals, given the estimates.
The code of this function is an adjusted version of the code from Daniel Saxton's
resample library (https://github.com/dsaxton/resample/). It has been adjusted to allow
for multi-valued statistics as well as the additional confidence interval types "bc",
"basic" and "normal".

.. currentmodule:: estimagic.inference.bootstrap

.. autofunction:: compute_ci

Here is a jupyter notebook with an example on how to use the bootstrap library:

.. toctree::
    :maxdepth: 1

    bootstrap_tutorial.ipynb

Here is an overview and theoretical background on the supported types of confidence
intervals:

.. toctree::
    :maxdepth: 1

    bootstrap_ci.rst
