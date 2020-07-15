
.. _first_derivative:


Gradient, Jacobian and Hessian
==============================

Estimagic wraps `numdifftools <https://pypi.org/project/numdifftools/>`_ to provide functions to calculate very precise gradients, jacobians and hessians of functions. The precision is achieved by evaluating numerical derivatives at different step sizes and using Richardson extrapolations. While this increases the computational cost, it works for any function, whereas other approaches that would yield a similar precision, like complex step derivatives, have stronger requirements.


.. automodule:: estimagic.differentiation.differentiation
    :members:



.. _step_options:

Influencing the Step Size
=========================

As mentioned before, we use numdifftools_ for all numerical differentiations with
Richardson extrapolations. numdifftools_ offers many ways of influencing the step
size of the extrapolation:


- base_step (float, array-like, optional):
    Defines the maximum step, if None, the value is set to ``EPS**(1/scale)````

- step_ratio (float , optional):
    Ratio between sequential steps generated. Must be > 1. If None
    then ``step_ratio`` is 2 for first order derivatives, otherwise it is 1.6

- num_steps (scalar integer, optional):
    default ``min_num_steps + num_extrap``. Defines number of steps generated.
    It should be larger than ``min_num_steps = (n + order - 1) / fact`` where
    ``fact`` is 1, 2 or 4 depending on differentiation method used.


- offset (float, optional):
    offset to the base step.

- num_extrap (int):
    number of points used for extrapolation.  Numdifftools Documentation says
    that the default is 0 but that would be surprising.


- scale (float, array like):
    scale used in base step. If not None it will override the default
    computed with the default_scale function.
