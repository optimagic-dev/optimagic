===========================
When (not) to use estimagic
===========================

estimagic provides tools for the estimation of empirical models, not for the implementation of the model themselves.
Consider the following workflow:

- | **The user sets up a model**, either programming it from scratches or using an already-existing implementation. As an example, the model can be a set of equations, possibly non-linear, together with distributional assumptions on model components.

- | **The user choose an estimation method** to fit the model's parameters to the observed data, for instance Maximum Likelihood or Method of Simulated Moments.

- | Given the chosen estimation method, **the user constructs the criterion function**. To continue our example, the criterion function will be the (negative) log-likelihood if Maximum Likelihood is chosen as an estimation method, or the (wighted) distance between simulated and observed moments for the Method of Simulated Moments.

This criterion function is then passed to estimagic for optimization and inference on the estimated parameters.
estimagic is flexible enough to handle derivative-based and derivative-free optimization, optimization errors and many type of constraints, but it may not be the tool you are looking for if your optimization problem has certain features. We provide a list of these features below.


Too many parameters
===================

If you have more than a few hundreds free parameters, you should look into the machine learning literature to find a suitable optimization algorithm.
estimagic is not a good fit because of two reasons:

* | The optimization algorithms in estimagic are either derivative-free or make some use of second-order information, for instance using the Hessian or the approximated Hessian.
  | Derivative-free optimization algorithms are suitable whenever the criterion function has many local optima, it is discontinuous or non-differentiable, but do not scale well to high-dimensional problems (because of the computational cost and/or the low convergence rate in high-dimensional space).
  | Computing the Hessian when the dimensionality of the problem is too high becomes unfeasible, because the memory requirement increases quadratically with the number of parameters.

* In estimagic, constraints are expressed based on the parameters' labels, rather than the parameters' positions. This introduces an additional memory overhead and slows
  down the optimization procedures when the parameter vector is extremely large. Moreover, the dashboard is not performant enough to display so many parameters.


A very fast criterion function
==============================
In each iteration of the numerical optimization, besides calculating the parameter vector for the next iteration, estimagic additionally:

* Logs the parameter vector and function evaluation in a database;
* Transforms an internal parameter vector (a numpy array) to the external one (a pandas DataFrame);
* Checks if errors occurred and handles them.

This introduces an overhead that can be relevant if the criterion function is very fast.
Exact timings depend on several factors. A rough guideline, measured on a laptop, is that the overhead is between 5 and 10 milliseconds for a large optimization problem with several constraints.

If you have a fast criterion function that is easy to optimize, consider ``scipy.optimize`` and ``nlopt`` instead. If you need diagnostic tools, constraints and complex parameter handling, it might be worth to use estimagic anyways.


A well-behaved or specialized problem
======================================
If you have a very well behaved problem (for instance, you know your criterion is convex or you have a linear programming problem), you should consider using a specialized library,
such as ``cvxopt`` or ``PuLP``.  While estimagic would work for these kind of problems, using libraries that exploit the additional knowledge will be faster.


Discrete optimization
=====================
estimagic can only optimize over continuous parameter spaces.
