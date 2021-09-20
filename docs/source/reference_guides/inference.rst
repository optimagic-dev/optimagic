Bootstrap Inference
========================

.. currentmodule:: estimagic.inference.bootstrap

In this section we document how to use the bootstrap to approximate the distribution
of statistics of interest on a given sample.

The main idea of the bootstrap is to combine three steps: Firstly, generate bootstrap
samples by drawing from the original data set with replacement. Secondly, calculating
the statistic of interest from these resampled data, and finally, using the various
estimates to draw inference on the distribution of the true statistic of interest.

Parts of the bootstrap functionality is based on  Daniel Saxton's resample library
:cite:`Saxton2018`. It has been adjusted to allow for the additional confidence interval
types "bc", "basic" and "normal". Moreover, it is faster.

The main bootstrap function takes care of all three steps and returns a
table containing means, standard errors and confidence intervals for the estimated
parameters.

.. autofunction:: bootstrap

However, the steps can also be called separately by the user, as described in what
follows.

.. currentmodule:: estimagic.inference.bootstrap_samples

The module implements the first step of drawing  B bootstrap samples in two separate
ways. The first, and default, method, is to simply draw observations from the original
dataset with replacement. The second method is a cluster robust bootstrap, often called
pairs cluster bootstrap, that draws clusters, defined by observations with the same value
of some specified stratum variable, from the original dataset with replacement. Sample
drawing is implemented by drawing seeds. To be as memory-efficient as possible, the
actual drawing of data points is by default deferred to the calculation of statistics.
However, it is possible to get a list of drawn resampled datasets using

.. autofunction:: get_bootstrap_samples

.. currentmodule:: estimagic.inference.bootstrap_estimates

The drawing of samples and calculation of statistics is handled by the following
function:

.. autofunction:: estimagic.inference.bootstrap_outcomes.get_bootstrap_outcomes

.. currentmodule:: estimagic.inference.bootstrap

To calculate mean, standard deviation, and a confidence interval of the statistic of
interest from the bootstrap outcomes, we can use

.. autofunction:: bootstrap_from_outcomes


The supported types of confidence are explained in more detail here:
:ref:`bootstrap_cis`.


.. bibliography:: ../refs.bib
    :filter: docname in docnames
