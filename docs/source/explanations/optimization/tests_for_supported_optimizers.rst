How all optimization algorithms supported in estimagic are tested
==================================================================

estimagic provides a unified interface for running optimizations
with algorithms from different optimization libraries and
additionally, allows setting different types of constraints to the optimization problem.

To test the external interface of all supported algorithms, we consider four criterion
or benchmark functions:

* Sum of squares/Sphere

  :math:`f({x}) = \Sigma^{D}_{i=1} ix_{i}^2`

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


Tests
-----------------------------
For testing the external interface, we write several test functions, each considers the
case of one constraint. Given the constraint, the test function considers all possible
combinations of - algorithm, to maximize or to minimize, criterion function
implementation, gradient function implementation for that criterion (if provided),
and whether ``criterion_and_derivative`` has been provided or not.
To illustrate, an example of a 'complete' testcase testing the ``scipy_lbfgsb``
algorithm for the fixed_constraint case would be:

test_with_fixed_constraint[scipy_lbfgsb-minimize-rosenbrock_dict_criterion-
rosenbrock_gradient-rosenbrock_criterion_and_gradient]


Same testcase with the gradient not specified and ``criterion_and_derivative`` not
specified:

test_with_fixed_constraint[scipy_lbfgsb-minimize-rosenbrock_scalar_criterion-None-None]

Constraint-cases
---------------------------
Here we show the calculations behind the true values, for each testcase (one criterion
and one constraint). The test functions compare these values with the solutions returned
by the algorithms, for each corresponding testcase.

**Trid Function**

  :math:`D=3 \rightarrow f({x}) = x_1^2 + 2x_2^2 + 3x_3^2`
  
.. raw:: html

    <div class="container">
    <div id="accordion" class="shadow tutorial-accordion">

        <div class="card tutorial-card">
            <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseOne">
                <div class="d-flex flex-row tutorial-card-header-1">
                    <div class="d-flex flex-row tutorial-card-header-2">
                        <button class="btn btn-dark btn-sm"></button>
                        No constraint case
                    </div>
                    <span class="badge gs-badge-link">

.. raw:: html

                    </span>
                </div>
            </div>
            <div id="collapseOne" class="collapse" data-parent="#accordion">
                <div class="card-body">

                1. No constraints case: ``[]``

                    :math:`x* = (3, 4, 3)`


                2. Fixed constraint: ``[{"loc": "x_1", "type": "fixed", "value": 1}]``

                    :math:`x_{1} = 1 \rightarrow f(x) = (x_2 - 1)^2 + (x_3 - 1)^2 - x_2 - x_3 x_2 \\
                    \Rightarrow \frac{\delta f({x})}{\delta x_2} = 2x_2 - 3 - x_3 = 0
                    \Rightarrow x_3 = 2x_2 - 3\\
                    \Rightarrow \frac{\delta f({x})}{\delta x_3} = 2x_3 - 2 - x_2 = 0
                    \Rightarrow x_2 = 2x_3 - 2\\
                    \Rightarrow x_2 = \frac{8}{3} , \quad x_3 = \frac{7}{3}\\
                    \rightarrow x* = (1,\frac{8}{3}, \frac{7}{3})`


                3. Probability constraint: ``[{"loc": ["x_1", "x_2"], "type": "probability"}]``

                    :math:`x_{1} + x_{2} = 1, \quad 0 \leq x_1 \leq 1, \quad 0 \leq x_2 \leq 1 \\
                    \rightarrow f({x}) = 3x_1^2 - 3x_1 - 3x_3 + x_3^2 + x_1 x_3 + 2 \\
                    \Rightarrow \frac{\delta f({x})}{\delta x_1} = 6x_1 - 3 + x_3 = 0
                    \Rightarrow x_3 = 3 - 6x_1\\
                    \Rightarrow \frac{\delta f({x})}{\delta x_3} = 2x_3 - 3 + x_1 = 0
                    \Rightarrow x_1 = 3 - 2x_3\\
                    \Rightarrow x_1 = \frac{3}{11}, \quad x_3 = \frac{15}{11}\\
                    \rightarrow x* = (\frac{3}{11}, \frac{8}{11}, \frac{15}{11})`


                4. Increasing constraint: ``[{"loc": ["x_2", "x_3"], "type": "increasing"}]``

                     :math:`\mathcal{L}({x_i}) = (x_1 - 1)^2 + (x_2 - 1)^2 + (x_3 - 1)^2 - x_1 x_2 - x_3 x_2 - \lambda(x_3 - x_2)\\
                     \Rightarrow \frac{\delta \mathcal{L}}{\delta x_1} = 2(x_1 - 1) - x_2 = 0\\
                     \Rightarrow \frac{\delta \mathcal{L}}{\delta x_2} = 2(x_2 - 1) - x_1 - x_3 + \lambda = 0\\
                     \Rightarrow \frac{\delta \mathcal{L}}{\delta x_3} = 2(x_3 - 1) - x_2 - \lambda = 0\\
                     \Rightarrow \frac{\delta \mathcal{L}}{\delta \lambda} = - x_3 + x_2 = 0\\
                     \Rightarrow x_2 = 2(x_1 - 1) = x_3 = \frac{10}{3}\\
                     \Rightarrow 2(x_2 - 1) - x_1 - 2 = 0\\
                     \Rightarrow 4(x_1 - 1) - 2 - x_1 - 2 = 0\\
                     \Rightarrow 3x_1 - 8 = 0 \Rightarrow x_1 = \frac{8}{3}\\
                     \rightarrow x* = (\frac{8}{3}, \frac{10}{3}, \frac{10}{3})`


                5. Decreasing constraint: ``[{"loc": ["x_1", "x_2"], "type": "decreasing"}]``

                    As of 8.03.20, we don't know.


                6. Equality constraint: ``[{"loc": ["x_1", "x_2", "x_3"], "type": "equality"}]``

                    :math:`x_{1} = x_{2} = x_{3} = x \\
                    \rightarrow f({x}) = x^2 - 6x + 3\\
                    \Rightarrow \frac{\delta f({x})}{\delta x} = 2x - 6 = 0\\
                    \Rightarrow x = 3\\
                    \rightarrow x* = (3,3,3)`


                7. Pairwise equality constraint: ``[{"locs": ["x_1", "x_2"], "type": "pairwise_equality"}]``


                    :math:`x_{1} = x_{2} \\
                    \rightarrow f({x}) = 2(x_1 - 1)^2 + (x_3 - 1)^2 - x_1^2 - x_3 x_1\\
                    \Rightarrow \frac{\delta f({x})}{\delta x_1} = 2x_1 - x_3 - 4 = 0 \Rightarrow x_3 = 2x_1 - 4\\
                    \Rightarrow \frac{\delta f({x})}{\delta x_3} = 2x_3 - x_1 - 2 = 0 \Rightarrow x_1 = 2x_3 - 2\\
                    \Rightarrow x_1 = \frac{10}{3}, x_3 = \frac{8}{3}\\
                    \rightarrow x* = (\frac{10}{3},\frac{10}{3},\frac{8}{3})`


                8. Covariance constraint: ``[{"loc": ["x_1", "x_2", "x_3"], "type": "covariance"}]``

                    As of 8.03.20, we don't know.


                9. Sdcorr constraint: ``[{"loc": ["x_1", "x_2", "x_3"], "type": "sdcorr"}]``

                    As of 8.03.20, we don't know.


                10. Linear constraint:``[{"loc": ["x_1", "x_2"], "type": "linear", "weights": [1, 2], "value": 4}]``
                     :math:`x_1 + 2x_2 = 4\\
                     \mathcal{L}({x_i}) = (x_1 - 1)^2 + (x_2 - 1)^2 + (x_3 - 1)^2 - x_1 x_2 - x_3 x_2 - \lambda(x_1 +2x_2-4)\\
                     \Rightarrow \frac{\delta \mathcal{L}}{\delta x_1} = 2(x_1 - 1) - x_2 - \lambda = 0\\
                     \Rightarrow \frac{\delta \mathcal{L}}{\delta x_2} = 2(x_2 - 1) - x_1 - x_3 - 2\lambda = 0\\
                     \Rightarrow \frac{\delta \mathcal{L}}{\delta x_3} = 2(x_3 - 1) - x_2 = 0 \\
                     \Rightarrow \frac{\delta \mathcal{L}}{\delta \lambda} = - x_1 - 2x_2 + 4 = 0\\
                     \Rightarrow x_2 = 2(x_3 - 1), \quad x_1 = 4 - 2x_2\\
                     \Rightarrow 2(4 - 2x_2 - 1) - x_2 = x_2 - 1 - 2 + x_2 - \frac{x_2}{4} - \frac{1}{2}\\
                     \rightarrow x* = (\frac{32}{27}, \frac{38}{27}, \frac{46}{27})`
