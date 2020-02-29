Bootstrap Inference
========================

.. currentmodule:: estimagic.inference.bootstrap

In this section we document how to use the bootstrap to approximate the distribution
of statistics of interest on a given sample.
We use the notation and formulations provided in chapter 10 of :cite:`Hansen2019`

Bruce E. Hansen - Econometrics,  (https://www.ssc.wisc.edu/~bhansen/econometrics)

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

.. autofunction:: get_bootstrap_sample_seeds

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

The details of confidence interval calculation are discussed below.

This module supports the calculation of different types of bootstrap confidence
intervals.

The first supported type is the "percentile" confidence interval.
Let :math: '\{ \hat{\theta}_1^*, ..., \hat{\theta}_B^*\}' denote the estimates of
estimator :math: '\hat{\theta}' for the B bootstrap samples. The idea of the percentile
confidence interval is to simply take the empirical quantiles :math: 'q_{p}^*' of
this distributions, so we have

.. math:: CI^{percentile} = [q_{\alpha/2}^*, q_{1-\alpha/2}].

The second supported confidence interval "normal" is based on a normal approximation.
Let :math: 's_{\hat{\theta}^{boot}}' be the sample standard error of the distribution
of bootstrap estimators, :math: 'z_q' the q-quantile of a standard normal
distribution and :math: '\hat{\theta}' be the full sample estimate of :math: '\theta'.
Then, the asymptotic normal confidence interval is given by

.. math:: CI^{normal} = [\hat{theta} - z_{1- \alpha/2} s_{\hat{\theta}^{boot}},  \hat{theta} + z_{1- \alpha/2} s_{\hat{\theta}^{boot}}].

The bias corrected "bc" bootstrap confidence interval addresses the issue of biased
estimators. Econometric details are discussed in section 10.17 of :cite:`Hansen2019`.
Let

.. math:: p^* = \frac{1}{B} \sum_{b=1}^B 1(\hat{\theta}_b^* \leq \hat{\theta})

and define :math: `z_0^* = \Phi^{-1} (p^*)` where :math: `\Phi` is the normal cdf. The
correction works via correcting significance level. Define

.. math:: x(\alpha) = \Phi(z_\alpha + 2 z_0^*).

Then, the bc confidence interval is given by

.. math:: CI^{bc} = [q_{x(\alpha/2)}^*, q_{x(1-\alpha/2)}^*].







.. math:: V = n^{-1}(Q^{T}WQ)^{-1} (Q^{T}W\Omega WQ) (Q^{T}WQ)^{-1}


.. bibliography:: ../refs.bib
    :filter: docname in docnames
