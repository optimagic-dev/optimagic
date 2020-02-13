"""Finite difference formulae for jacobians and hessians."""


def jacobian_forward(evals, steps, f0):
    n_steps, dim_f, dim_x = evals.pos.shape
    diffs = evals.pos - f0.reshape(1, dim_f, 1)
    return diffs / steps.pos.reshape(n_steps, 1, dim_x)


def jacobian_backward(evals, steps, f0):
    n_steps, dim_f, dim_x = evals.neg.shape
    diffs = evals.neg - f0.reshape(1, dim_f, 1)
    return diffs / steps.neg.reshape(n_steps, 1, dim_x)


def jacobian_central(evals, steps, f0):
    n_steps, dim_f, dim_x = evals.pos.shape
    diffs = evals.pos - evals.neg
    deltas = steps.pos - steps.neg
    return diffs / deltas.reshape(n_steps, 1, dim_x)
