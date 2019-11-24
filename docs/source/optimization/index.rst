============
Optimization
============


Optimization is usually the most difficult and time consuming part
of a research project in structural econometrics. Different problems require
different optimizers and it is not known ex-ante which optimizer will perform
best. Moreover, most optimizers are black boxes that only return when they are
done and don't give you any information on the progress in the meantime.

Estimagic makes it very easy to try out a large number of optimizers, either
sequentially or in parallel by just switching a few arguments of the
``maximize`` or ``minimize`` functions. And the estimagic Dashboard informs
you in real time how the optimization is going.

In the following we first explain the basic interface of estimagic's
optimization functions. Then we explain the arguments in detail. This section
is also relevant for you if you call the ``minimize`` or ``maximize`` functions
indirectly, e.g. via the a ``fit`` method of a package that uses estimagic.
While those packages might implement different defaults that are tailored to the
problem at hand, they will usually let you pass your own arguments through to
estimagic if you desire to do so.

.. toctree::
   :maxdepth: 1

   interface
   params
   constraints/index
   dashboard
   algorithms
   general_options
