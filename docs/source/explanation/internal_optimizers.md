(internal_optimizer_interface)=

# Internal optimizers for optimagic

optimagic provides a large collection of optimization algorithm that can be used by
passing the algorithm name as `algorithm` into `maximize` or `minimize`. Advanced users
can also use optimagic with their own algorithm, as long as it conforms with the
internal optimizer interface.

The advantages of using the algorithm with optimagic over using it directly are:

- optimagic turns an unconstrained optimizer into constrained ones.
- You can use logging.
- You get great error handling for exceptions in the criterion function or gradient.
- You get a parallelized and customizable numerical gradient if the user did not provide
  a closed form gradient.
- You can compare your optimizer with all the other optimagic optimizers by changing
  only one line of code.

All of this functionality is achieved by transforming a more complicated user provided
problem into a simpler problem and then calling "internal optimizers" to solve the
transformed problem.

## The internal optimizer interface

(to be written)

## Output of internal optimizers

(naming-conventions)=

## Naming conventions for algorithm specific arguments

Many optimizers have similar but slightly different names for arguments that configure
the convergence criteria, other stopping conditions, and so on. We try to harmonize
those names and their default values where possible.

Since some optimizers support many tuning parameters we group some of them by the first
part of their name (e.g. all convergence criteria names start with `convergence`). See
{ref}`list_of_algorithms` for the signatures of the provided internal optimizers.

The preferred default values can be imported from `optimagic.optimization.algo_options`
which are documented in {ref}`algo_options`. If you add a new optimizer to optimagic you
should only deviate from them if you have good reasons.

Note that a complete harmonization is not possible nor desirable, because often
convergence criteria that clearly are the same are implemented slightly different for
different optimizers. However, complete transparency is possible and we try to document
the exact meaning of all options for all optimizers.

## Algorithms that parallelize

(to be written)

## Nonlinear constraints

(to be written)
