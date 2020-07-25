
.. _first_derivative:


Calculate Gradients and Jacobians
=================================

Estimagic wraps `numdifftools <https://pypi.org/project/numdifftools/>`_ to provide functions to calculate very precise gradients, jacobians and hessians of functions. The precision is achieved by evaluating numerical derivatives at different step sizes and using Richardson extrapolations. While this increases the computational cost, it works for any function, whereas other approaches that would yield a similar precision, like complex step derivatives, have stronger requirements.


.. automodule:: estimagic.differentiation.derivatives
    :members:
