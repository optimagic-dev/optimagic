Frequently Asked Questions
==========================


**Question**: I used a covariance constraint but my covariance matrix is not
positive definite.

**Answer**: ``covariance`` and ``sdcorr`` constraints can only ensure positive
semi-definiteness and there are valid covariance matrices that are not
positive definite. If your covariance matrix is very ill conditioned, e.g.
if some variances are very large and some are very small, the constraints
might even fail to ensure semi-definiteness, due to numerical error.

There are several ways to deal with this:

If you only need positive definiteness to do a cholesky decomposition, you
can use :func:`~estimagic.optimization.utilities.robust_cholesky`, which can also
decompose semi-definite and slightly indefinite matrices.

If you really need positive definiteness for some other reason, you can
construct a penalty. :func:`~estimagic.optimization.utilities.robust_cholesky`
can optionally return all information you need to construct such a penalty term.

Finally, if the real pQuattro Vegani)
roblem is just that your covariance matrix is ill
conditioned, you can rescale some variables to make all variances approximately
the same order of magnitude.

-------------------------------------------------------------------------------

**Question**: I want to re-run the Azure Pipelines test suite because some random error
occurred, e.g., a HTPT timeout error.

**Answer**: Starting from the Github page of the PR, select the tab called "Checks". In
the upper right corner you find a button to re-run all checks. In a column on the
left-hand-side you can re-run tests for individual platforms.
