===============
Differentiation
===============

.. currentmodule:: estimagic.differentiation.differentiation

Overview
--------

Here we will document functions and the underlying methods for first and second order
differentiation. We provide differentiation based on finite differences, namely forward,
backward or central differences for the following problems:

- Gradient of real valued multivariate functions
- Jacobian matrix of real vector-valued multivariate functions
- Hessian matrix of real valued multivariate functions


Functions
---------

Here we will give a short documentation of the functions. The numerical functions based
on the given finite difference method are not implemented by us and instead
called from the numdifftools_ library.

.. _numdifftools: https://pypi.org/project/numdifftools/

The variable method "method" allows the user to call a specific finite difference
method. We only allow for one of the methods and therefore exclude complex step and
other possibilities:

- forward differences (method="forward")
- backward differences (method="backward")
- central differences (method="central")

As the central difference method is the most accurate, it is the default for all three
functions. The numdifftools functions use Richardson extrapolation as their standard
tool to increase accuracy. The concept of this extrapolation method can be found in the
documentation_ of numdifftools.

.. _documentation: https://numdifftools.readthedocs.io/en/latest/

To deactivate the exptrapolation method and just use standard finite differences,
just set the "extrapolation" variable to False. Then we will use the stepsize
introduced below.

Therefore our functions have the following interface:

.. autofunction:: gradient

The second order, which is the hessian matrix is calculated by:

.. autofunction:: hessian

For vector valued multivariate functions the following yields the numerical calculation
of the jacobian matrix:

.. autofunction:: jacobian


.. _step_options:

Influencing the Step Size
-------------------------

As mentioned before, we use numdifftools_ for all numerical differentiations with
Richardson extrapolations. numdifftools_ offers many ways of influencing the step
size of the extrapolation:


- base_step (float, array-like, optional):
    Defines the minimum step, if None, the value is set to ``EPS**(1/scale)````

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



Methods
-------

In this section we will explain the mathematical background of forward, backward and
central differences. The main ideas in this chapter are taken from this [Dennis]_ and
Schneidler. Our toolkit is developed for multivariate functions but can of
course also be used for single variable functions. Notation wise x is used for the
pandas series params_sr. I index the entries of x as a n-dimensional vector, where
n is the number of variables in params_sr. The forward difference for the gradient
is given by:

.. [Dennis] Numerical Methods for Unconstrained Optimization and Nonlinear Equations
      J. E. Dennis, Jr. and Robert B. Schnabel, 1996

.. math::

    \nabla f(x) = \begin{pmatrix}\frac{f(x + e_0 * h_0) - f(x)}{h_0}\\
    \frac{f(x + e_1 * h_1) - f(x)}{h_1}\\.\\.\\.\\ \frac{f(x + e_n * h_n)
    - f(x)}{h_n} \end{pmatrix}


The backward difference for the gradient is given by:

.. math::

    \nabla f(x) = \begin{pmatrix}\frac{f(x) - f(x - e_0 * h_0)}{h_0}\\ \frac{f(x) -
    f(x - e_1 * h_1)}{h_1}\\.\\.\\.\\ \frac{f(x) - f(x - e_n * h_n)}{h_n}
    \end{pmatrix}


The central difference for the gradient is given by:

.. math::

    \nabla f(x) =
    \begin{pmatrix}\frac{f(x + e_0 * h_0) - f(x - e_0 * h_0)}{h_0}\\
    \frac{f(x + e_1 * h_1) - f(x - e_1 * h_1)}{h_1}\\.\\.\\.\\ \frac{f(x + e_n * h_n)
    - f(x - e_n * h_n)}{h_n} \end{pmatrix}

For the optimal stepsize h the following rule of thumb applies:

.. math::

    h_i = (1 + |x[i]|) * \sqrt\epsilon

With the above in mind it is easy to calculate the Jacobian matrix. The calculation of
the finite difference w.r.t. each variable of params_sr yields a vector, which is the
corresponding column of the Jacobian matrix. The optimal stepsize remains the same.


For the Hessian matrix, we repeatetly call the finite differences functions. As we
allow for central finite differences in the second order derivative only, the
deductions for forward and backward, are left to the interested reader:

.. math::

    f_{i,j}(x)
        = &\frac{f_i(x + e_j * h_j) - f_i(x - e_j * h_j)}{h_j} \\
        = &\frac{\frac{f(x + e_j * h_j + e_i * h_i) - f(x + e_j * h_j - e_i * h_i)}{h_i}
           - \frac{
                 f(x - e_j * h_j + e_i * h_i) - f(x - e_j * h_j - e_i * h_i)
             }{h_i}}{h_j} \\
        = &\frac{
               f(x + e_j * h_j + e_i * h_i) - f(x + e_j * h_j - e_i * h_i)
           }{h_j * h_i} \\
          &+ \frac{
                 - f(x - e_j * h_j + e_i * h_i) + f(x - e_j * h_j - e_i * h_i)
             }{h_j * h_i}

For the optimal stepsize a different rule applies:

.. math::

    h_i = (1 + |x[i]|) * \sqrt[3]\epsilon

Similar deviations lead to the elements of the hessian matrix calculated by backward and
central differences. As this is very straightforward, it will not be mentioned here.
