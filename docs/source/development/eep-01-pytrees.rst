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


Backwards compatibility
=======================

All changes are fully backwards compatible.


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


Optimization with pytrees
=========================

In this section we look at optimizations that become possible with the proposed changes.
As an example we use a hypothetical criterion function with pytree inputs and outputs
to describe how a user can optimize it. We also give a rough intuition what happens
behind the scenes and with which registries the pytree functions are called.


The criterion function
----------------------

Consider a criterion function that takes parameters in the following format:

.. code-block:: python

    params = {
        "delta": 0.95,
        "utility": pd.DataFrame(
            [[0.5, 0]] * 3, index=["a", "b", "c"], columns=["value", "lower_bound"]
        ),
        "probs": np.array([[0.8, 0.2], [0.3, 0.7]]),
    }


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
- ``solution_derivative``: Maybe we should not even have this entry.

.. danger:: We need to discuss if an in which form we want to have a solution
    derivative entry. In it's current form it is useless if constraints are used.
    This gets worse when we allow for pytrees and translating this into a meaningful
    shape might be very difficult.


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

Closed form derivatives need to take the exact same format as one would obtain
when applying our numerical derivatives to the criterion function (see below). This is
also compatible with JAX (in all cases that are supported by JAX) and thus a natural
requirement since in most cases closed form derivatives will be calculated via JAX.

Numerical derivatives with pytrees
==================================

Problem: Higher dimensional extensions of pytrees
-------------------------------------------------

The derivative of a function that maps from a 1d array to a 1d array (usually called
Jacobian) is a 2d matrix. If the 1d arrays are replaced by pytrees, we need a
two dimensional extension of the pytrees. Below we well look at how JAX does this
and why we cannot simply copy that solution, even though we want to stay as compatible
with it as possible.


The JAX interface
-----------------

Let's look at an example. We first define a function in terms of 1d arrays and then
in terms of pytrees and look at a JAX calculated jacobian in both cases:


.. code-block:: python

    def square(x):
        return x ** 2


    x = jnp.array([1, 2, 3, 4, 5, 6.0])

    jacobian(square)(x)

.. code-block:: bash

    DeviceArray([[ 2.,  0.,  0.,  0.,  0.,  0],
                 [ 0.,  4.,  0.,  0.,  0.,  0],
                 [ 0.,  0.,  6.,  0.,  0.,  0],
                 [ 0.,  0.,  0.,  8.,  0.,  0],
                 [ 0.,  0.,  0.,  0., 10.,  0],
                 [ 0.,  0.,  0.,  0.,  0., 12]], dtype=float32)


.. code-block:: python

    def tree_square(x):
        out = {
            "c": x["a"] ** 2,
            "d": x["b"].flatten() ** 2,
        }

        return out


    tree_x = {"a": jnp.array([1, 2.0]), "b": jnp.array([[3, 4], [5, 6.0]])}


    jacobian(tree_square)(tree_x)

Instead of showing the entire results, let's just look at the resulting tree structure
and array shapes:


.. code-block:: python

    {
        "c": {
            "a": (2, 2),
            "b": (2, 2, 2),
        },
        "d": {
            "a": (4, 2),
            "b": (4, 2, 2),
        },
    }

The outputs for hessians have even deeper nesting and three dimensional arrays inside
the nested dictionary. Similarly, we would get higher dimensional arrays if one of
the original pytrees had already contained a 2d array.


Limitations of the JAX interface
--------------------------------

Most JAX functions `only work with Pytrees of arrays
<https://jax.readthedocs.io/en/latest/pytrees.html#pytrees-and-jax-functions>`_, whereas
estimagic allows pytrees containing pandas.Series and pandas.DataFrames with value
column. Unfortunately, this poses non-trivial challenges for numerical derivatives
because those data types have no natural extension in arbtirary dimensions.


Proposed solution
-----------------

Our solution needs to fulfill two requirements:

1. Compatible with JAX in the sense than whenever a derivative can be calculated with
JAX it can also be calculated with estimagic and the result has the same structure.
2. Compatible with the rest of estimagic in the sense that any function that can be
optimized can also be differentiated. In the special case of differentiating with
respect to a DataFrame it also needs to be backwards compatible.

A solution that achieves this is to treat Series and DataFrames with value columns as
1d arrays and other DataFrames as 2d arrays, then proceed as in JAX and finally try
to preserve as much index and column information as possible.

This leads to very natural results in the typical usecases with flat dicts of Series
or params DataFrames both as inputs and outputs and is backwards compatible with
everything that is supported already.

Howeverer, similar to JAX, not everything that is supported will also be a good idea.
Predicting where a pandas Object is preserved and where it will be replaced by an array
might be hard for very nested pytrees. However, these rules are mainly defined to avoid
hard limitations that have to be checked and documented. Users will learn to avoid too
much complexity by avoiding complex pytrees as inputs and outputs at the same time.


Examples of pytrees with DataFrames
-----------------------------------

We repeat the example from the JAX interface above with the following changes:

1. The 1d numpy array in x["a"] is replaced by a DataFrame with value column
2. The "d" entry in the output becomes a Series instead of a 1d numpy array.


.. code-block:: python

    def pd_tree_square(x):
        out = {
            "c": x["a"]["value"] ** 2,
            "d": pd.Series(x["b"].flatten() ** 2, index=list("jklm")),
        }

        return out


    pd_tree_x = {
        "a": pd.DataFrame(data=[[1], [2]], index=["alpha", "beta"], columns=["value"]),
        "b": np.array([[3, 4], [5, 6]]),
    }

    pd_tree_square(pd_tree_x)


::

    {
        'c':
            "alpha"    1
            "beta"     4
            "dtype": int64,
        'd':
            "j"        9
            "k"       16
            "l"       25
            "m"       36
            dtype: int64,
    }

The resulting shapes of the jacobian will be the same as before. For all arrays
with only two dimensions we can preserve some information from the Series and DataFrame
indices. On the higher dimensional ones, this will be lost.

.. code-block:: python

    {
        "c": {
            "a": (2, 2),  # df with columns ["alpha", "beta"], index ["alpha", "beta"]
            "b": (2, 2, 2),  # numpy array without label information
        },
        "d": {
            "a": (4, 2),  # columns ["alpha", "beta"], index [0, 1, 2, 3]
            "b": (4, 2, 2),  # numpy array without label information
        },
    }



To get more intuition for the structure of the result, let's add a few labels to the
very first jacobian:


+--------+----------+----------+----------+----------+----------+----------+----------+
|        |          | a        |          | b        |          |          |          |
+--------+----------+----------+----------+----------+----------+----------+----------+
|        |          | alpha    | beta     | j        | k        | l        | m        |
+--------+----------+----------+----------+----------+----------+----------+----------+
| c      | alpha    | 2        | 0        | 0        | 0        | 0        | 0        |
+        +----------+----------+----------+----------+----------+----------+----------+
|        | beta     | 0        | 4        | 0        | 0        | 0        | 0        |
+--------+----------+----------+----------+----------+----------+----------+----------+
| d      | 0        | 0        | 0        | 6        | 0        | 0        | 0        |
+        +----------+----------+----------+----------+----------+----------+----------+
|        | 1        | 0        | 0        | 0        | 8        | 0        | 0        |
+        +----------+----------+----------+----------+----------+----------+----------+
|        | 2        | 0        | 0        | 0        | 0        | 10       | 0        |
+        +----------+----------+----------+----------+----------+----------+----------+
|        | 3        | 0        | 0        | 0        | 0        | 0        | 12       |
+--------+----------+----------+----------+----------+----------+----------+----------+


The indices ["j", "k", "l", "m"] unfortunately never made it into the result because
they were only applied to elements that already came from a 2d array and thus always
have a 3d Jacobian, i.e. the result entry ``["c"][b"]`` is a reshaped version of the
upper right 2 by 4 array and the result entry ``["d"]["b"]`` is a reshaped version of
the lower right 4 by 4 array.


Implementation
--------------

.. danger:: This is the only place in the EEP where I have now clue what the
    implementation will look like. ``pybaum`` does not yet support the generation of
    higher dimensional extensions of pytrees even for simple pytrees of arrays.

    My guess is that internally we would always flatten inputs and outputs as much as
    possible, calculate numerical derivatives and then parse the resulting numerical
    derivatives to give them the same structure as in JAX. Ideas are welcome!






Estimation summaries with pytrees
=================================


Covariance matrices with pytrees
================================


Moments in MSM estimation as pytrees
====================================


Sensitivity measures as pytrees
===============================




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
