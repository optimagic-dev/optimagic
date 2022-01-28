.. _eep-01:

===============
EEP-01: Pytrees
===============

+------------+------------------------------------------------------------------+
| Author     | `Janos Gabler <https://github.com/janosg>`_                      |
+------------+------------------------------------------------------------------+
| Status     | Draft                                                            |
+------------+------------------------------------------------------------------+
| Type       | Standards Track                                                  |
+------------+------------------------------------------------------------------+
| Created    | 2022-01-28                                                       |
+------------+------------------------------------------------------------------+
| Resolution |                                                                  |
+------------+------------------------------------------------------------------+


Abstract
========

This EEP explains how we will use pytrees to allow for more flexible specification
of parameters for optimization or differentiation, more convenient ways of writing
moment functions for msm estimation and more. The actual code to work with pytrees
is implemented in `Pybaum`_, developed by :ghuser:`janosg` and :ghuser:`tobiasraabe`.


.. _Pybaum: https://github.com/OpenSourceEconomics/pybaum


Motivation
==========

Estimagic has many functions that require user written functions as inputs. Examples
are:
- criterion functions and their derivatives for optimization
- functions of which numerical derivatives are taken
- functions that calculate simulated moments
- functions that calculate bootstrap statistics

In all cases, there are some restrictions on possible inputs and outputs of the
user written functions. For example, parameters for numerical optimization need to
be provided as pandas.DataFrame with a value column. Simulated moments and
bootstrap statistics need to be returned as a pandas.Series, etc.

Pytrees allow to relax many of those restrictions on interfaces of user provided
functions. This is not only more convenient for users, but sometimes also allows to
reduce overhead because the user can choose optimal data structures for their problem.


Background: What is a pytree
============================

Pytree is a term used in TensorFlow and JAX to refer to a tree-like structure built out
of container-like Python objects with arbitrary levels of nesting.

What is a container can be re-defined for each application. By default, lists, tuples
and dicts are considered containers and everything else is a leaf. Then the following
are examples of pytrees:


.. code-block:: python

    [1, "a", np.arange(2)]  # 3 leaves

    [1, {"k1": 2, "k2": (3, 4)}, 5]  # 5 leaves

    np.arange(5)  # 1 leaf


What makes pytrees so powerful are the operations defined for them. The most important
ones are:

- ``tree_flatten``: Convert any pytree into a flat list of leaves + metadata
- ``tree_unflatten``: The inverse of ``tree_flatten``
- ``tree_map``: Apply a function to all leaves in a pytree
- ``tree_names``: Generate a list of names for all leaves in a pytree

The above examples of pytrees would look as follows when flattened (with a default
definition of containers):

.. code-block:: python

    [1, "a", np.arange(2)]

    [1, 2, 3, 4, 5]

    [np.arange(5)]

By adding numpy arrays to the registry of container like objects the flattened versions
of the examples would look as follows:

.. code-block:: python

    [1, "a", 0, 1]

    [1, 2, 3, 4, 5]

    [0, 1, 2, 3, 4]

Needless to say, it is possible to register anything as container. For example, we would
add pandas.Series and pandas.DataFrame (with varying definitions, depending on the
application).


Backwards compatibility
=======================

All changes are fully backwards compatible.



Use of pytrees in estimagic
===========================


``params`` for optimization and differentiation
-----------------------------------------------




How to specify constraints when ``params`` are pytrees
------------------------------------------------------

Currently, parameters to which a constraint is applied are selected via a "loc" or
"query" entry in the constraint dictionary.

This keeps working as long as params are specified as one DataFrame. If a more general
pytree is used we need a "selector" entry instead. The value of that entry is a
callable that takes the pytree and returns selected parameters.

The selected parameters can be returned as pytrees (same container definition as in
params, i.e. only "value" column of DataFrames is considered, unless the user
overrides container definition). For constraints where order plays a role
(e.g. increasing), the order defined by ``tree_flatten`` is used.



The output of criterion functions and functions to be differentiated
--------------------------------------------------------------------

TBD



Empirical moments and output of ``simulate_moments``
----------------------------------------------------

TBD



Output of ``outcome`` function in bootstrap
-------------------------------------------

TBD




Higher dimensional extensions of pytrees
========================================

Intuition for the problem
-------------------------

Pytrees usually replace function inputs or outputs that are represented as vectors in
math and as 1d numpy arrays in code. This is the case for optimization, differentiation
estimation and bootstrapping in estimagic.

In those applications, higher dimensional objects might arise. For example, the
first derivative of a function that takes a vector and returns a vector (the Jacobian)
is a matrix. The second derivative of such a function (the Hessians) would usually be
defined as a 3d array. Another example of higher dimensional objects are covariance
matrices of parameter vectors that arise during estimation.


How does JAX do it
------------------

JAX's solution to this problem entails two things:

1. Functions that deal with higher dimensional extensions of pytrees only allow pytrees
where all leaves have a natural higher dimensional extension (e.g. numbers become
1d arrays, 1d arrays become 2d arrays, ...
2. These function return deeply nested pytrees of arrays to accomodate all results.

Let's look at an example. We first define a function in terms of 1d arrays and then
in terms of pytrees and look at a JAX calculated jacobian in both cases:


.. code-block:: python

    def square(x):
        return x ** 2


    x = jnp.array([1, 2, 3, 4, 5.0])

    jacobian(square)(x)

.. code-block:: bash

    DeviceArray([[ 2.,  0.,  0.,  0.,  0.],
                 [ 0.,  4.,  0.,  0.,  0.],
                 [ 0.,  0.,  6.,  0.,  0.],
                 [ 0.,  0.,  0.,  8.,  0.],
                 [ 0.,  0.,  0.,  0., 10.]], dtype=float32)


.. code-block:: python

    def tree_square(x):
        out = {
            "c": x["a"] ** 2,
            "d": x["b"] ** 2,
        }
        return out


    tree_x = {"a": jnp.array([1.0, 2]), "b": jnp.array([3.0, 4, 5])}

    jacobian(tree_square)(tree_x)

.. code-block:: python

    {
        'c': 
        {
            'a': DeviceArray([[2., 0.],
                              [0., 4.]], dtype=float32),

            'b': DeviceArray([[0., 0., 0.],
                              [0., 0., 0.]], dtype=float32)
        },
        'd':
        {
            'a': DeviceArray([[0., 0.],
                              [0., 0.],
                              [0., 0.]], dtype=float32),

            'b': DeviceArray([[ 6.,  0.,  0.],
                              [ 0.,  8.,  0.],
                              [ 0.,  0., 10.]], dtype=float32)
        }
    }

The outputs for hessians have even deeper nesting and three dimensional arrays inside
the nested dictionary.


Can we do the same?
-------------------




Compatibility with plotting and estimation tables
=================================================





Advanced options for functions that work with pytrees
=====================================================

There are two argument to ``tree_flatten`` and other pytree functions that determine
which entries in a pytree are considered a leaf and which a container as well as how
containers are flattened. 1. ``registry`` and 2. ``is_leaf``. See the documentation
of ``pybaum`` for details.

To allow for absolute flexibility, each function that works with pytrees needs to
allow a user to pass in a ``registry`` and an ``is_leaf`` argument. If a function
works with multiple pytrees (e.g. in ``estimate_msm`` the ``params`` are a
pytree and ``emprirical_moments`` are a pytree) it needs to allow users to pass in
multiple registries and is_leaf functions (e.g. ``params_registry``,
``params_is_leaf`` and ``moments_registry``, ``moments_is_leaf``.


However, we need only as many registries as there are different pytrees. For example
since ``simulated_moments`` and ``empirical_moments`` always need to be pytrees with
the same structure, they do not need separate registries and is_leaf functions.



Compatibility with JAX autodiff
===============================


While we allow for pytrees of arrays, numbers and DataFrames, JAX only allows pytrees
of arrays and numbers for automatic differentiation.

If you want to use automatic differentiation with estimagic you will thus have to
restrict yourself in the way you specify parameters.

We will try to find a way of extending JAX but it probably won't happen very soon.


Need for documentation
======================

New documentation
-----------------

- New best practices for params
- Examples of optimizing over a custom params class
- Examples of simulated moments with pytree


Adjustments
-----------
