
.. _internal_optimizer_interface:

=================================
Internal Optimizers for Estimagic
=================================


Introduction
============

Estimagic provides a large collection of optimization algorithm that can be
used by passing the algorithm name as ``algorithm`` into ``maximize`` or ``minimize``.
However, advanced users can also use estimagic with their own algorithm, as long as it
conforms with the internal optimizer interface.

The advantages of using the algorithm with estimagic over using it directly are:

- Estimagic turns an unconstrained optimizer into one that can deal efficiently with a
  wide range of constraints
  (see .. _link: how_to_guides/how_to_use_constranits.ipynb).
- You can use estimagic's logging capabilities.
- You get a beautiful real time dashboard to monitor your optimization.
- You get great error handling for exceptions in the criterion function or gradient.
- You get a parallelized and customizable numerical gradient if the user did not provide
  a closed form gradient.
- You can compare your optimizer with all the other estimagic optimizers by changing
  only one line of code.

All of this functionality is achieved by transforming a more complicated user provided
problem into a simpler problem and then calling "internal optimizers" to solve the
transformed problem.



The Internal Optimizer Interface
================================

An internal optimizer is a function that minimizes an objective function it has two
mandatory arguments:

1. ``criterion_and_derivative`` (described below)
2. ``x``: One dimensional numpy array with start parameters.

Followed by two strongly encouraged but not absolutely mandatory arguments:

1. lower_bounds: 1d numpy array with the same length as x with lower
    bounds for the parameters. Can be -np.inf for unbound parameters.
2. upper_bounds: 1d numpy array with the same length as x with upper
    bounds for the parameters. Can be -np.inf for unbound parameters.

Moreover, it can accept any number of additional arguments that are used by the
optimizer. Neither the mandatory arguments, nor the bounds should have a default value.

The only non-trivial argument is ``criterion_and_derivative``. This is a callable that
accepts three arguments and returns the output of the user provided criterion function
(float or dict), the output of a user provided or numerical derivative (np.ndarray) or
both. For more details on valid outputs of the criterion function see
:ref:`maximize_and_minimize`.

The arguments  of criterion_and_derivative are:

- x (np.ndarray): A one dimensional vector with free parameters
- task (str): One of "criterion", "derivative" and "criterion_and_derivative"
- algorithm_info (dict): Dict with the following entries:
    - "primary_criterion_entry": For optimizers that minimize a scalar function this has
      to be "value". For optimizers that optimize a sum it has to be "contributions".
      For optimizers that optimize a sum of squares it has to be "root_contributions".
    - "parallelizes": Bool that indicates if the algorithm calls the internal
      criterion function in parallel. If so, caching is disabled.
    - "needs_scaling": Bool. True if the algorithm is not scale invariant. In that case
      we can issue a warning if the user did not scale his problem with estimagic.
    - "name": The name of the algorithm that can be displayed in error messages

``criterion_and_derivative`` is the centerpiece of how estimagic achieves its magic.
It does logging, converts the internal parameters into a DataFrame for the user provided
criterion function, handles errors and more.

.. _internal_optimizer_output:

Output of Internal Optimizers
=============================


After convergence or when another stopping criterion is achieved the internal optimizer
should return a dictionary with the following entries:

- solution_x: The best parameter achieved so far
- solution_criterion: The value of the criterion at solution_x. This can be a scalar
  or dictionary.
- solution_derivative: The derivative evaluated at solution_x
- solution_hessian: The (approximate) hessian evaluated at solution_x
- n_criterion_evaluations: The number of criterion evaluations.
- n_derivative_evaluations: The number of derivative evaluations.
- n_iterations: The number of iterations
- success: True if convergence was achieved
- reached_convergence_criterion: The name of the reached convergence criterion.
- message: A string with additional information.

If some of the entries are missing, they will automatically be filled with ``None`` and
no errors are raised. Nevertheless, you should try to return as much information as
possible.


.. _naming_conventions:

Naming Conventions for Optional Arguments
=========================================

Many optimizers have similar but slightly different names for arguments that configure
the convergence criteria, other stopping conditions, and so on. We try to harmonize
those names and their default values where possible.

The preferred default values can be imported from
``estimagic.optimization.algo_options`` which are documented in :ref:`algo_options`.
If you add a new optimizer to estimagic you should only deviate from them if you have
good reasons.

Note that a complete harmonization is not possible nor desirable, because often
convergence criteria that clearly are the same are implemented slightly different for
different optimizers. However, complete transparency is possible and we try to document
the exact meaning of all options for all optimizers.


.. _convergence_criteria:

Convergence Criteria
--------------------

Declare convergence when any of the convergence criteria is satisfied:

In general, the defaults are inspired by scipy because Nlopt deactivates all
convergence criteria by default. The names are inspired by Nlopt, but more expressive.


- **relative_criterion_tolerance** *(float)*: Stop when the relative improvement between
  two iterations is smaller than this. The exact definition of relative improvement
  depends on the optimizer and should be documented  there. To disable it, set it to 0.

- **absolute_criterion_tolerance** *(float)*: Stop when the absolute improvement between
  two iterations is smaller than this. To disable it, set it to 0.

- **absolute_gradient_tolerance** *(float)*: Stop when all entries (or for some
  algorithms the norm) of the gradient are smaller than this. For bound constrained
  optimizers this typically refers to a projected gradient. The exact definition should
  be documented for each optimizer. To disable it, set it to zero.

- **relative_gradient_tolerance** *(float)*: Stop when all entries (or for some
  algorithms the norm) of the gradient, divided by the absolute value of the criterion
  function is smaller than this. For bound constrained optimizers this typically refers
  to a projected gradient. The exact definition should be documented for each
  optimizer. To disable it, set it to zero.


- **scaled_gradient_tolerance** *(float)*: Stop when all entries (or for some
  algorithms the norm) of the gradient, divided by the norm of the gradient at start
  parameters is smaller than this. For bound constrained optimizers this typically
  refers to a projected gradient. The exact definition should be documented for each
  optimizer. To disable it, set it to zero.

- **relative_params_tolerance** *(float)*:
  Stop when the relative change in parameters between two iterations is smaller than
  this. The exact definition of relative change and whether this refers to the maximum
  change or the average change depends on the algorithm and should be documented there.
  To disable it, set it to zero.

- **absolute_params_tolerance** *(float)*: Stop when the absolute change in parameters
  between two iterations is smaller than this. Whether this refers to the maximum
  change or the average change depends on the algorithm and should be documented there.
  To disable it, set it to zero.


Other stopping criteria
-----------------------


- **max_criterion_evaluations** *(int)*: If the maximum number of function evaluation is
  reached, the optimization stops but we do not count this as successful convergence.
  The function evaluations used to evaluate a numerical gradient do not count for this.

- **max_iterations** *(int)*: If the maximum number of iterations is reached, the
  optimization stops, but we do not count this as successful convergence. The difference
  to ``max_criterion_evaluations`` is that one iteration might need several criterion
  evaluations, for example in a line search or to determine if the trust region radius
  has to be decreased.


Optimizer configurations
------------------------

- **max_line_search_steps** *(int)*: Maximum number of criterion (and possibly
  derivative) evaluations per line search.

- **trustregion_initial_radius** *(float)*: Note this is very problem specific and
  you can typically do better than using the default. Intuitively, this can be
  described as a reasonable change in parameters the optimizer can take from the
  start values. The default is to take 10% of the largest start parameter, but at
  least 0.1.

- **limited_memory_storage_length** *(int)*: Maximum number of stored objects
  in a limited memory algorithm. Typically this refers to the number of gradient
  evaluations used to approximate the Hessian in Quasi-Newton algorithms.


Other Conventions
=================

- Internal optimizer are functions and should thus adhere to python naming conventions,
  for functions (i.e. only consist of lowercase letters and individual words should be
  separated by underscores). For optimizers that are implemented in many packages
  (e.g. Nelder Mead or BFGS), the name of the original package in which it was
  implemented has to be part of the name.
- All arguments except ``criterion_and_derivative`` and ``x`` should be keyword only
  and have default values that are set to the preferred defaults documented above
  unless there is a good reason to deviate.
- There should only be arguments used by the optimizer, i.e. only the
  convergence criteria that are actually supported by an optimizer should be part of
  its interface. The signature should also not contain ``*args`` or ``**kwargs``.
- In particular, if an optimizer does not support bounds, it should not have the bounds
  as arguments.
