Background and Methods
======================

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
