.. _estimation_general_options:

General options
===============

You can pass a dictionary with general options to the
:func:`~estimagic.optimization.optimize.minimize` or
:func:`~estimagic.optimization.optimize.maximize` function. The following options are
available.

**Scaling of the parameter vector**

``estimagic`` is able to automatically scale the parameter vector such that the
optimizer receives parameters closer to 1. There are two available methods.

1. ``{"scaling": "start_values"}`` divides the parameter vector by the starting
   values for all starting values not in ``[-1, 1]``.

2. ``{"scaling": "gradient"}`` divides the parameter vector by the inverse of the
   gradient for each parameter not in ``[-1e-2, 1e-2]``. By default,
   ``{"scaling_gradient_method": "central", "scaling_gradient_extrapolation": False}``
   are also set. ``"forward"`` and ``"backward"`` can also be used as a method and
   extrapolation can be turned on by setting the value to ``True``. Note that without
   extrapolation, the gradient is computed faster.
