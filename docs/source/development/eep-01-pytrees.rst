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


Optimization by Example
=======================

In this example we use a hypothetical criterion function with pytree inputs and outputs
to describe how how a user can optimize it.  We also give a rough intuition what happens
behind the scenes and with which registries the pytree functions are called.


Inputs
------

Consider a criterion function that takes parameters in the following format:

.. code-block:: python

    params = {
        "delta": 0.95,
        "utility": pd.DataFrame(
            [[0.5, 0]] * 3, index=["a", "b", "c"], columns=["value", "lower_bound"]
        ),
        "probs": np.array([[0.8, 0.2], [0.3, 0.7]]),
    }

Outputs
-------

The criterion function returns a dictionary of the form:

.. code-block:: python

    {
        "value": 1.1,
        "contributions": {"a": np.array([0.36, 0.25]), "b": 0.49},
        "root_contributions": {"a": np.array([0.6, 0.5]), "b": 0.7},
    }


Run an optimization
-------------------

.. code-block:: python

    from estimagic import minimize

    minimize(
        criterion=crit,
        params=params,
        algorithm="scipy_lbfgsb",
    )

The internal optimizer (in this case the lbfgsb algorithm from scipy) will only see
a modified version of ``crit`` that takes a 1d numpy array as only argument and
returns a scalar float (the "value" entry of the result of crit). Numerical derivatives
are also taken on that function.

If instead a derivative based least squares optimizer like ``"scipy_ls_dogbox"`` had
been used, the internal optimizer would see a modified version of ``crit`` that takes
a 1d numpy array and returns a 1d numpy array (the flattened version of the
``"root_contributions"`` entry of the result of crit).


To do the conversion between the pytrees and the flat arrays, we would use
``tree_flatten`` and ``tree_unflatten`` with the following container types:
- dict
- list
- tuple
- numpy.ndarray
- pd.Series
- pd.DataFrame (when flattening params only the value column would be considered.
when flattening the output of criterion, all numerical values of the DataFrame would
be considered)


The optimization output
-----------------------

The following entries of the output of minimize are affected by the change:
- ``"solution_params"``: A pytree with the same structure as ``params``
- ``"solution_criterion"``: The output dictionary of crit evaluated solution params
- ``solution_derivative``: Maybe we should not even have this entry. In its current
form it is meaningless because it is a derivative with respect to internal
parameters.


Add a bound on "delta"
----------------------

Bounds on parameters that are inside a DataFrame with "value" column can simply be
specified as before. For all others, there are separate ``lower_bounds`` and
``upper_bounds`` arguments in ``maximize`` and ``minimize``.


``lower_bounds`` and ``upper_bounds`` are pytrees of the same structure as ``params``
or a sub-tree that preserves enough structure to match all bounds. For example:


.. code-block:: python

    minimize(
        criterion=crit,
        params=params,
        algorithm="scipy_lbfgsb",
        lower_bounds={"delta": 0},
        upper_bounds={"delta": 1},
    )

This would add bounds for delta, keep the bounds on all ``"utility"`` parameters
and have no bounds on the ``"probs"`` parameters.


Add a constraint
----------------

Currently, parameters to which a constraint is applied are selected via a "loc" or
"query" entry in the constraint dictionary.

This keeps working as long as params are specified as one DataFrame. If a more general
pytree is used we need a "selector" entry instead. The value of that entry is a
callable that takes the pytree and returns selected parameters.

The selected parameters can be returned as pytrees (same container definition as in
params, i.e. only "value" column of DataFrames is considered, unless the user
overrides container definition). For constraints where order plays a role
(e.g. increasing), the order defined by ``tree_flatten`` is used.

As an example, let's add probability constraints for each row of ``"probs"``:


.. code-block:: python

    constraints = [
        {"selector": lambda params: params["probs"][0], "type": "probability"},
        {"selector": lambda params: params["probs"][1], "type": "probability"},
    ]

    minimize(
        criterion=crit,
        params=params,
        algorithm="scipy_lbfgsb",
        constraints=constraints,
    )


Internally, constraints are already applied on a 1 dimensional numpy array and the
parameter selections specified by "loc" and "query" are translated into positions in
that array. The only thing that changes is that we now also have to translate the
parameter selections from "selector" functions into positions. This is trivial by
calling ``tree_unflatten`` on an ``np.arange()`` of suitable length, calling
the selector functions on the resulting pytree and recording all numbers that are there.


Numerical derivatives during optimization
-----------------------------------------

Derivatives are taken on modified functions that map from 1d numpy array to scalars
or 1d numpy arrays.


Closed form derivatives
-----------------------

.. danger:: It is not clear yet what closed form derivatives need to look like.
    Since most of them will be calculated by JAX (at least in our applications)
    it would be good to be JAX compatible in all cases that are supported by JAX.
    In all cases, they should be aligned with the results one would get from when using
    our numerical derivative functions directly on the criterion function, even though
    this would not happen during optimization.


Numerical derivatives by example
================================






Likelihood Estimation by example
================================














Backwards compatibility
=======================

All changes are fully backwards compatible.






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
        "c": {
            "a": DeviceArray([[2.0, 0.0], [0.0, 4.0]], dtype=float32),
            "b": DeviceArray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float32),
        },
        "d": {
            "a": DeviceArray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float32),
            "b": DeviceArray(
                [[6.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 10.0]], dtype=float32
            ),
        },
    }

The outputs for hessians have even deeper nesting and three dimensional arrays inside
the nested dictionary.

The JAX solution represents an extreme approach in the sense that it never tries to
flatten anything in order to avoid high dimensional or nested outputs. This is the
only possible choice, considering the goals of JAX:
1. It is essentially a library that implements n-dimensional arrays
2. Everything is composable, i.e. there are never things that are just results and
not inputs for further calculations.


The other extreme would be to flatten all pytrees into pandas.Series or DataFrames with
"value" column. This would bring us back to the state before pytrees. However, it is
not a desirable solution because the outputs are hard to work with and it would even be
hard to ensure backwards compatibility for the case where parameters are just one
DataFrame with value column.


Can we do the same as JAX
-------------------------

Unfortunately, we cannot do exactly the same. The main reasons are:

- We have to allow for pytrees containing DataFrames for backward compatibility and
  those do not have a natural extension in arbitrary dimensions.
- For estimation results (at least for summaries from which tables can be produced) we
  need a way to "add columns" to a pytree. This is a form of higher dimensional
  extension of pytrees that does not have a counterpart in JAX
- A covariance matrix that is represented similar to the jacobian above is not useful
  for most users of estimagic



Design goals
------------

1. If a derivative is taken, that could also be taken with JAX, it should produce
the same output.
2. Our solution needs to naturally nest the current behavior when ``params`` are just
one DataFrame with value column.


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
