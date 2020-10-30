Moments-based estimation
========================

.. currentmodule:: estimagic.inference.moment_covs

In this section we document how to calculate standard errors of a GMM or MSM estimator.
We use the notation and formulations provided in section 13.25 of :cite:`Hansen2019`

The distribution of the estimator is shaped by the moment conditions, which are
functions of the estimated parameter :math:`\beta` to the real numbers. We denote the
vector of all moment condition values for observation :math:`i` by :math:`\textbf{g}_i`.
With this on hand the calculation of the covariance matrix can be formalized:

.. math:: V = n^{-1}(Q^{T}WQ)^{-1} (Q^{T}W\Omega WQ) (Q^{T}WQ)^{-1}

where :math:`W` is some weighting matrix chosen by the econometrican, :math:`\Omega` the
covariance matrix of the moment conditions, given by

.. math::

    \Omega = n^{-1} \sum_{i=1}^n (\mathbf{g}_i(\beta) - \mathbf{\bar{g}})
    (\mathbf{g}_i(\beta) - \mathbf{\bar{g}})^T

where :math:`\mathbf{\bar{g}}` is the mean of all :math:`\mathbf{g}_i(\beta)`. The
auxiliary matrix :math:`Q` is calculated by

.. math:: Q = n^{-1} \sum_{i=1}^n \frac{\partial}{\partial\beta^T} \mathbf{g}_i(\beta)

The corresponding function to this calculations is

.. autofunction:: gmm_cov

After the estimation of :math:`\beta`, the moment condition functions have to be
evaluated at :math:`\beta` for each observation to get the first input of the function
above. The second input can be obtained by calling for each observation the jacobian
function in the differentiation module of estimagic. This function provides the partial
derivatives at the estimated parameter value. The third input is the weighting matrix,
which was chosen before the estimation.


.. bibliography:: ../../refs.bib
    :filter: docname in docnames
