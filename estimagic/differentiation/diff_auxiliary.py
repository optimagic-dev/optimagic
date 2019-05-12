def central(f, f_x0, params_sr, var, h, *func_args, **func_kwargs):
    params_r = params_sr.copy()
    params_r[var] = params_sr[var] + h
    params_l = params_sr.copy()
    params_l[var] = params_sr[var] - h
    central_diff = f(params_r, *func_args, **func_kwargs) - \
                   f(params_l, *func_args, **func_kwargs)
    return central_diff / (2.0 * h)


def forward(f, f_x0, params_sr, var, h, *func_args, **func_kwargs):
    params = params_sr.copy()
    params[var] = params_sr[var] + h
    return (f(params, *func_args, **func_kwargs) - f_x0) / h


def backward(f, f_x0, params_sr, var, h, *func_args, **func_kwargs):
    params = params_sr.copy()
    params[var] = params_sr[var] - h
    return (f_x0 - f(params, *func_args, **func_kwargs)) / h
