===============
Differentiation
===============

Overview
--------

Here we will document functions and the underlying methods for first and second order
differentiation. We provide differentiation based on finite differences, namely forward,
backward or central differences for the following problems:

- Gradient of real valued multivariate functions
- Jacobian matrix of real vector-valued multivariate functions
- Hessian matrix of real valued multivariate functions

This will give us a easy tool to calculate standard errors etc.

Functions
---------

Here we will give a short documentation of the functions. The different calculation
methods, defined by the two variables method and extrapolation are explained in the next
chapter. The first two functions give the first and second order derivative for a real
valued multivariate function. The first order is calculated by:

.. automodule:: estimagic.differentiation.gradient :members:

The second order, which is the hessian matrix is calculated by:

.. automodule:: estimagic.differentiation.jacobian
    :members:

For vector valued multivariate functions the following yields the numerical calculation
of the jacobian matrix:

.. automodule:: estimagic.differentiation.hessian
    :members:

Methods
-------

In this section we will explain the mathematical background of forward, backward and
central differences. Our toolkit is developed for multivariate functions but can of
course also be used for single variable functions. Notation wise x is used for the
pandas series params_sr. I index the entries of x as a n-dimensional vector, where
n is the number of variables in params_sr. The forward difference for the gradient
is given by:

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


For the Hessian matrix, we repeatetly call the finite differences functions. This yields
to the following scheme for the hessian matrix, calculated by forward difference:

.. math::
      f_{i,j}(x) = \frac{f_i(x + e_j * h_j) - f_i(x)}{h_j} =
      \frac{\frac{f(x + e_j * h_j + e_i * h_i) - f(x + e_j * h_j)}{h_i} -
      \frac{f(x + e_i * h_i) - f(x)}{h_i}}{h_j} = \\
      = \frac{f(x + e_j * h_j + e_i * h_i) - f(x + e_j * h_j) -
      f(x + e_i * h_i) + f(x)}{h_j * h_i}

For the optimal stepsize a different rule applies:

.. math::
        h_i = (1 + |x[i]|) * \sqrt[3]\epsilon

Similar deviations lead to the elements of the hessian matrix calculated by backward and
central differences. As this is very straightforward, it will not be mentioned here.

Extrapolation
-------------

The variable "extrapolant" in the three functions above, allows with the value
"richardson" to use the Richardson extrapolation method. This leads to higher accuracy
in the case of forward or backward differences. How this extrapolation works, will be
outlined in the following. First we will recall the Taylor expansion, which leads to the
finite differences:

.. math::
        f(x + e_i * h_i) \approx f(x) + h_i * f_i(x) + \frac{h_i^2 }{2} f_{i, i}(x) +
         \frac{h_i^3}{6} f_{i,i,i}(x) + ....

If we cut this term after the second order differentiation and evaluate the term at 2h
and 4h, we get the following linear equation system:

.. math::
        \begin{pmatrix} f(x + e_i * h_i) - f(x) \\ f(x + e_i * 2h_i) - f(x) \\
        f(x + e_i * 4h_i) - f(x) \end{pmatrix} =
        \begin{bmatrix} 1 & 1/2 & 1/6 \\ 2 & 2 & 8/6 \\ 4 & 8 & 64/6 \end{bmatrix}
        \begin{pmatrix} h_i * f_i(x) \\ h_i^2 f_{i, i}(x) \\ h_i^3 f_{i,i,i}(x)
        \end{pmatrix}

The equation system is solved by:

.. math::
        \begin{pmatrix} h_i * f_i(x) \\ h_i^2 f_{i, i}(x) \\ h_i^3 f_{i,i,i}(x)
        \end{pmatrix} =
        \frac{1}{12}\begin{bmatrix} 32 & 12 & 1 \\ -48 & 30 & -3 \\ 24 & -18 & 3
        \end{bmatrix}
        \begin{pmatrix} f(x + e_i * h_i) - f(x) \\ f(x + e_i * 2h_i) - f(x) \\
        f(x + e_i * 4h_i) - f(x) \end{pmatrix}

And therefore we get:

.. math::
        f_i(x) = \frac{ 32 (f(x + e_i * h_i) - f(x))  - 12  (f(x + e_i * 2h_i) - f(x)) +
        f(x + e_i * 4h_i) - f(x)}{12h_i}

Again this calculation can be easily adapted to central and backward differences and
will therefore not be notated here. For the hessian matrix, this method is repeatedly
called and the formulas are therefore straightforward.
