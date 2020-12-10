.. _optimization_error_handling:

========================================
How to Handle Errors During Optimization
========================================


Try to Avoid Errors
===================

Often, optimizers try quite extreme parameter vectors, which then can raise errors in
your criterion function or derivative. Even though estimagic makes it simple to restart
your optimization from the last parameter value, this is annoying. Below is a very short
list of things you can do to avoid this behavior:

- Set bounds for your parameters, that prevent extreme parameter constellations.
- Use the ``bounds_distance`` option with a not too small value for ``covariance`` and
  ``sdcorr`` constraints.
- Use :func:`~estimagic.optimization.utilities.robust_cholesky` instead of normal
  cholesky decompositions or try to avoid cholesky decompositions by restructuring
  your algorithm.
- Avoid to take ``np.exp`` without further safeguards. With 64 bit floating point
  numbers, the exponential function is only well defined roughly between -700 and 700.
  Below it is -inf, above it is inf. Sometimes you can use ``scipy.special.logsumexp``
  to avoid unsafe evaluations of the exponential function. Otherwise you can avoid
  problems by setting bounds. In the worst case, use clipping. Note, however, that
  clipping leads to flat regions in your criterion function which can lead to erroneous
  convergence.


The Two Levels of Error Handling
================================

Despite all efforts, some errors cannot be avoided. Therefore, you have a lot of control
over error handling during the optimizations. The three levels on which you can
configure it are:

The ``error_handling`` and ``eror_penalty`` arguments of ``maximize`` and ``minimize``
--------------------------------------------------------------------------------------

These two arguments determine if the optimization algorithms sees an
error that might occur during the criterion or gradient evaluation.

``error_handling`` takes the values ``"raise"`` and ``"continue"``. If ``"raise"``,
the error is not caught and the optimizer will stop or handle it. If ``"continue"``,
we replace the criterion function by a penalty term that can be fix or parameter
dependent and the optimizer will never know that an error occurred. Note that you will
still be warned about all errors.

The default error handling is ``"raise"``.

``error_penalty`` is a dict with the entries "constant" (float) and "slope" (float)
which determine the value of the penalty. The penalty function is then calculated as
``constant + slope * norm(params - start_params)`` where ``norm`` is the euclidean
distance.

Making the penalty parameter dependent via the slope is meant to avoid flat spots in the
penalized region. For minimization problems, a positive slope guides the optimizer back
to the start parameters until it reaches a valid region again. The same holds for a
negative slope in maximization problems.

Of course you can deactivate this and set the slope to 0.

The default constant is f0 + abs(f0) + 100 for minimizations and f0 - abs(f0) - 100 for
maximizations, where f0 is the criterion value at start parameters.
The default slope is 0.1 for minimization problems and -0.1 for maximization problems.


The ``error_handling`` entry in the ``batch_evaluator_options``
---------------------------------------------------------------

This argument determines when you get notified about a failed optimization.

It is mainly relevant when using estimagic's ability to run several optimizations in
parallel.

It can take the values ``"raise"`` and ``"continue"``. If ``"raise"``, you will get an
error as soon as any optimization fails. Otherwise, estimagic runs all optimizations
even if some of them fail. The result of the failing optimizations will contain the
traceback of the error.

The default value is ``"continue"`` if more than one optimization is run and ``"raise"``
if you run only one optimization.
