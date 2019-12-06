How to add an optimizer
=======================

Adding optimizers to estimagic is a well-defined task which requires almost no knowledge
about the rest of the package. The basic idea is to write a slightly adjusted interface
which works with a generic objective function that maps a NumPy array to a criterion
value.

In the following, the interface is discussed using the wrapper for the scipy package as
an example. Without further ado, here is the complete wrapper!

.. literalinclude:: ../../../estimagic/optimization/scipy.py
    :language: python
    :lines: 5-34

The function receives the following arguments

- ``func`` is the objective function which works on NumPy arrays.
- ``x0`` are the starting values of the parameters.
- ``bounds`` is a tuple of two NumPy arrays which have the same length as ``x0``. The
  first array represents the lower and the second the upper bound. Infinite values
  represent unbounded parameters.
- ``algo_name`` is optional for choosing the optimizer if the wrapper is for a package
  with multiple options.
- ``algo_options`` is a dictionary of options for the specific ``algo_name``. It could
  also be the case that the keys in ``algo_options`` become arguments of the interface
  which can be seen in the interface for POUNDERs
  (:func:`~estimagic.optimization.pounders.minimize_pounders_np`).

The function proceeds by preparing the arguments for the package. In this case, the
bounds are converted.

Then, the optimization is started which returns an result object.

In the last step, the result object is converted to a dictionary. estimagic tries to
harmonize the output of different optimizers as much as possible, but in general all
information should be passed to the user. The harmonized fields are the following.

- ``"fitness"`` contains the fitness value of the optimization. Other names are the
  sample log-likelihood or the sum of squared weighted errors.
- ``"x"`` contains the parameter vector of the solution.
- ``"n_evaluations"`` is the number of function evaluations, not counting evaluations
  for gradients, etc..

The rest of the fields should be named appropriately. See the other functions,
:func:`~estimagic.optimization.scipy._process_scipy_results`,
:func:`~estimagic.optimization.pygmo._process_pygmo_results`, and
:func:`~estimagic.optimization.pounders._process_pounders_results`, for examples.
