.. _scaling:

==================================
How to scale optimization problems
==================================


Real world optimization problems often comprise Parameters of vastly different orders of
magnitudes. This is typically not a problem for gradient based optimization algorithms
but can considerably slow down derivative free optimizers. Below we describe three
simple heuristics to improve the scaling of optimization problems and discuss the pros
and cons of each approach.

What does well scaled mean
==========================

In short, an optimization problem is well scaled if a fixed step in any direction yields
 a roughly similar sized change in the criterion function.

In practice, this can never be achieved perfectly (at least for nonlinear problems).
However, one can easily improve over simply ignoring the problem altogether.


Heuristics to improve scaling
=============================




Divide by absolute value of start parameters
--------------------------------------------

In many applications, parameters with very large start values will vary over a wide
range and a change in that parameter will only lead to a relatively small change in
the criterion function. If this is the case, the scaling of the optimization problem
can be improved by simply dividing all parameter vectors by the start parameters.

Advantages

- very simply
- works with any type of constraints

Disadvantages

- Makes scaling dependent on start values
- Parameters with zero start value need special treatment



.. code-block:: python

    def sphere(params):
        return (params["value"] ** 2).sum()


    start_params = pd.DataFrame(data=np.arange(5), columns=["value"])
    start_params["lower_bound"] = 0
    start_params["upper_bound"] = 2 * np.arange(5) + 1

    minimize(
        criterion=sphere_with_noise,
        params=start_params,
        algorithm="scipy_lbfgsb",
        scaling=True,
        scaling_options={"method": "start_values", "clipping_value": 0.1},
    )


Divide by bounds
----------------

In many optimization problems one has additional information on bounds of the parameter
space. Some of these bounds are hard (e.g. probabilities or variances are non negative),
others are soft and derived from simple considerations (e.g. if a time discount factor
were smaller than 0.7, we would not observe anyone to pursue a university degree in a
structural model of educational choices or if an infection probability was higher
than 20 % for distant contacts the covid pandemic would have been over after a
month). For parameters than strongly influence the criterion function, the bounds
stemming from these considerations are typically tighter as for parameters that have
a small effect on the criterion function.

Thus a natural approach to improve the scaling of the optimization problem is to re-map
all parameters such that the bounds [0, 1] for all parameters. This has the additional
advantage that absolute and relative convergence criteria on parameter changes become
the same.

Advantages

- very simple
- works well in many practical applications
- scaling is independent of start value
- No problems with division by zero

Disadvantages

- Only works if all parameters have bounds
- This prohibits some kinds of other constraints in estimagic


.. code-block:: python

    def sphere(params):
        return (params["value"] ** 2).sum()


    start_params = pd.DataFrame(data=np.arange(5), columns=["value"])
    start_params["lower_bound"] = 0
    start_params["upper_bound"] = 2 * np.arange(5) + 1

    minimize(
        criterion=sphere_with_noise,
        params=start_params,
        algorithm="scipy_lbfgsb",
        scaling=True,
        scaling_options={"method": "bounds", clipping_value: 0.0},
    )


Divide by gradient
------------------

Dividing all parameters by the gradient of the criterion function at the start values
means that around the start values the problem is scaled optimally. In practice, we do
not take the exact gradient, but a numerical gradient calculated with a very large step
size. This is more robust for noisy or wiggly functions.


Advantages

- Optimal scaling near start values
- Less arbitrary than other methods

Disadvantages

- Computationally expensive
- Not robust for very noisy or very wiggly functions
- Depends on start values
- Parameters with zero gradient need special treatment


.. code-block:: python

    def sphere(params):
        return (params["value"] ** 2).sum()


    start_params = pd.DataFrame(data=np.arange(5), columns=["value"])
    start_params["lower_bound"] = 0
    start_params["upper_bound"] = 2 * np.arange(5) + 1

    minimize(
        criterion=sphere_with_noise,
        params=start_params,
        algorithm="scipy_lbfgsb",
        scaling=True,
        scaling_options={"method": "gradient", "clipping_value": 0.1},
    )


Notes on the Syntax
-------------------

Scaling is disabled by default. If enabled, but no ``scaling_options`` are provided,
we use the ``"start_values"`` method with a ``"clipping_value"`` of 0.1. This is the
default method because it can be used for all optimization problems and low
computational cost. We strongly recommend you read the above guidelines and choose the
method that is most suitable for your problem.
