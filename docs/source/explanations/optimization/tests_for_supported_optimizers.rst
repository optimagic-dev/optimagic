How all optimization algorithms supported in estimagic are tested
==================================================================

estimagic provides a unified interface for running optimizations
with algorithms from different optimization libraries and
additionally, allows setting different types of constraints to the optimization problem.

To test the external interface of all supported algorithms, we consider four criterion
or benchmark functions:

* Sum of squares/Sphere

:math:`f({x}) = \Sigma^{D}_{i=1} ix_{i}^2`

:math:`D=3 \rightarrow f({x}) = x_1^2 + 2x_2^2 + 3x_3^2`

Global minima: :math:`x* = (0, 0, 0), \quad f(x*) = 0`

* Trid

:math:`f({x}) = \Sigma^{D}_{i=1}(x_{i} - 1)^2 - \Sigma^{D}_{i=2}(x_i x_{i-1})`

:math:`D=3 \rightarrow f({x}) = (x_1-1)^2 + (x_2-1)^2 + (x_3-1)^2 - x_2 x_1 - x_3 x_2`

* Rotated hyper ellipsoid

:math:`f({x}) = \Sigma^{D}_{i=1} \Sigma^{i}_{j=1}x_j^2`

:math:`D=3 \rightarrow f({x}) = x^2_1 + (x^2_1 + x^2_2) + (x^2_1 + x^2_2 + x^2_3)`

* Rosenbrock

:math:`\Sigma^{D-1}_{i=1}(100(x_i+1 - x_i^2)^2 + (x_i - 1)^2)`

:math:`D=3 \rightarrow f({x}) = 100(x_2 - x_1^2) + (x_1 - 1)^2`

Global minimum: :math:`x* = (1, 1, 1)`


We implement each function and its gradient in different ways, taking
into account the types of objective functions that estimagic's
``minimize`` and ``maximize`` accepts  for optimization. All algorithms accept
criterion functions specified in a dictionary, while a subset also accept the criterion
specified in scalar form. Likewise, if specified, the gradient of a criterion can be
an nd.array or a pandas object. We test for all possible cases.
For instance, for rotated hyper ellipsoid, we implement the following functions:

* rotated_hyper_ellipsoid_scalar_criterion
* rotated_hyper_ellipsoid_dict_criterion: This provides a dictionary wherein the
  ``contributions`` and ``root_contributions`` keys present the criterion as a least
  squares problem, relevant when we are testing a least squares algorithm.
* rotated_hyper_ellipsoid_gradient
* rotated_hyper_ellipsoid_pandas_gradient: Computes the gradient of the rotated hyper
  ellipsoid function, as a pandas object.
* rotated_hyper_ellipsoid_criterion_and_gradient

These criterion functions are specified in the ``examples`` directory. See docstrings
for all relevant details relating to each implementation.


Constraints
---------------------------
Here we show the calculations behind the true values, for each testcase (one criterion
and one constraint). The test functions compare these values with the solutions returned
by the algorithms, for each corresponding testcase.

Tests
-----------------------------
For testing the external interface, we write several test functions, each considers the
case of one constraint. Given the constraint, the test function considers all possible
combinations of - algorithm, to maximize or to minimize, criterion function
implementation, gradient function implementation for that criterion (if provided),
and whether ``criterion_and_derivative`` has been provided or not.
To illustrate, an example of a 'complete' testcase testing the ``scipy_lbfgsb``
algorithm for the fixed_constraint case would be:

``
test_with_fixed_constraint[scipy_lbfgsb-minimize-rosenbrock_dict_criterion-
rosenbrock_gradient-rosenbrock_criterion_and_gradient]
``

Testcase with the gradient not specified and ``criterion_and_derivative`` not specified:

``
test_with_fixed_constraint[scipy_lbfgsb-minimize-rosenbrock_scalar_criterion-None-None]>
``
