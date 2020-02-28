Bootstrap Inference
========================

.. currentmodule:: estimagic.inference.bootstrap

In this section we document how to use the bootstrap to approximate the distribution
of statistics of interest on a given sample.
We use the notation and formulations provided in chapter 10 of :cite:`Hansen2019`.

Bruce E. Hansen - Econometrics,  (https://www.ssc.wisc.edu/~bhansen/econometrics)

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

Confidence Intervals
********************************

This module supports the calculation of different types of bootstrap confidence
intervals.

The first supported type is the **"percentile"** confidence interval, as discussed in
section 10.10 of the Hansen textbook.
Let :math:`\{ \hat{\theta}_1^*, ..., \hat{\theta}_B^*\}` denote the estimates of
estimator :math:`\hat{\theta}` for the B bootstrap samples. The idea of the percentile
confidence interval is to simply take the empirical quantiles :math:`q_{p}^*` of
this distributions, so we have

.. math:: CI^{percentile} = [q_{\alpha/2}^*, q_{1-\alpha/2}^*].

The second supported confidence interval **"normal"** is based on a normal approximation
and discussed in Hansen's section 10.9.
Let :math:`s_{\hat{\theta}^{boot}}` be the sample standard error of the distribution
of bootstrap estimators, :math:`z_q` the q-quantile of a standard normal
distribution and :math:`\hat{\theta}` be the full sample estimate of :math:`\theta`.
Then, the asymptotic normal confidence interval is given by

.. math:: CI^{normal} = [\hat{\theta} - z_{1- \alpha/2} s_{\hat{\theta}^{boot}},  \hat{\theta} + z_{1- \alpha/2} s_{\hat{\theta}^{boot}}].

The bias-corrected **"bc"** bootstrap confidence interval addresses the issue of biased
estimators. This problem is often present when estimating nonlinear models. Econometric
details are discussed in section 10.17 of Hansen. Let

.. math:: p^* = \frac{1}{B} \sum_{b=1}^B 1(\hat{\theta}_b^* \leq \hat{\theta})

and define :math:`z_0^* = \Phi^{-1} (p^*)`, where :math:`\Phi` is the standard normal
cdf. The bias correction works via correcting the significance level.
Define :math:`x(\alpha) = \Phi(z_\alpha + 2 z_0^*)` as the corrected significance level
for a target significant level of :math:`\alpha`. Then, the bias-corrected confidence
interval is given by

.. math:: CI^{bc} = [q_{x(\alpha/2)}^*, q_{x(1-\alpha/2)}^*].


A further refined version of the bias-corrected confidence interval is the
bias-corrected and accelerated interval, short **"bca"**, as discussed in section 10.20
of Hansen. The general idea is to correct for skewness sampling distribution.
Downsides of this confidence interval are that it takes quite a lot of time to compute,
since it features calculating leave-one-out estimates of the original sample.
Formally, again, the significance levels are adjusted. Define

.. math:: \hat{a}=\frac{\sum_{i=1}^{n}\left(\bar{\theta}-\hat{\theta}_{(-i)}\right)^{3}}{6\left(\sum_{i=1}^{n}\left(\bar{\theta}-\hat{\theta}_{(-i)}\right)^{2}\right)^{3 / 2}},

where :math:`\bar{\theta}=\frac{1}{n} \sum_{i=1}^{n} \widehat{\theta}_{(-i)}`.
This is an estimator for the skewness of :math:`\hat{\theta}`. Then, the corrected
significance level is given by

.. math:: x(\alpha)=\Phi(z_{0}+\frac{z_{\alpha}+z_{0}}{1-a(z_{\alpha}+z_{0})})

and the bias-corrected and accelerated confidence interval is given by

.. math:: CI^{bca} = [q_{x(\alpha/2)}^*, q_{x(1-\alpha/2)}^*].

The studentized confidence interval, here called **"t"** type confidence interval first
studentizes the bootstrap parameter distribution, i.e. applies the transformation
:math:`\frac{\hat{\theta}_b-\hat{\theta}}{s_{\hat{\theta}^{boot}}}`, and then builds
the confidence interval based on the estimated quantile function of the studentized
data :math:`\hat{G}`:

.. math:: CI^{t} = \left[\hat{\theta}+\hat{\sigma} \hat{G}^{-1}(\alpha / 2), \hat{\theta}+\hat{\sigma} \hat{G}^{-1}(1-\alpha / 2)\right]

The final supported confidence interval method is the **"basic"** bootstrap confidence
interval, which is derived in section 3.4 of :cite:`Wassermann2006`, where it is called
the pivotal confidence interval. It is given by

.. math:: CI^{basic} = \left[\hat{\theta}+\left(\hat{\theta}-\hat{\theta}_{u}^{\star}\right), \hat{\theta}+\left(\hat{\theta}-\hat{\theta}_{l}^{\star}\right)\right],

where :math:`\hat{\theta}_{u}^{\star}` denotes the :math:`1-\alpha/2` empirical quantile
of the bootstrap estimate distribution for parameter :math:`\theta` and
:math:`\hat{\theta}_{l}^{\star}` denotes the :math:`\alpha/2` quantile.


.. bibliography:: ../refs.bib
    :filter: docname in docnames
