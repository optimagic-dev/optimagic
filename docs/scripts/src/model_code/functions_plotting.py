import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from scipy.optimize import minimize
import functools
import imageio
from matplotlib.widgets import TextBox
from src.model_code.example_function import example_criterion
from src.model_code.example_function import _unpack_x
from src.model_code.example_function import example_hessian
from src.model_code.example_function import example_gradient

from estimagic.visualization.colors import get_colors
colors = get_colors(palette = "categorical",number = 12)


def plot_function():
    x_grid = np.linspace(0, 20, 100)
    y_grid = [example_criterion(x) for x in x_grid]
    fig, ax = plt.subplots()
    sns.lineplot(x=x_grid, y=y_grid, ax=ax, color=colors[0])
    sns.despine()
    return fig, ax
    

def plot_history(evaluated_points, argmin):
    """Plot the function and all evaluated points."""
    fig, ax = plot_function()
    sns.rugplot(evaluated_points, ax=ax, color=colors[1])
    ax.plot([argmin], [example_criterion(argmin)], marker="*", color=colors_red[0])
    return fig, ax


def minimize_with_history(fun, x0, method, jac=None, hess=None):
    """Dumbed down scipy minimize that returns full history.
    
    This is really only meant for illustration in this notebook. In particular,
    the following restrictions apply:
    
    - Only works for 1 dimensional problems
    - does not support all arguments
    
    """
    history = []
    def wrapped_fun(x, history=history):
        history.append(_unpack_x(x))
        return fun(x)
    
    res = minimize(wrapped_fun, x0, method=method, jac=jac, hess=hess)
    res.history = history
    return res
    
def taylor_expansion(x, x0, radius):
    """Evaluate taylor expansion around x0 at x."""
    x = _unpack_x(x)
    x0 = _unpack_x(x0)
    f = example_criterion(x0)
    f_prime = example_gradient(x0)
    f_double_prime = example_hessian(x0)
    
    diff = x - x0
    res = f + f_prime * diff + f_double_prime * 0.5 * diff ** 2
    return res


def regression_surrogate(x, x0, radius):
    """Evaluate a regression based surrogate model at x.
    
    x0 and radius define the trust region in which evaluation points are sampled. 
   
   """
    x = _unpack_x(x)
    x0 = _unpack_x(x0)
    deviations = [-radius, 0, radius]
    
    evaluations = [example_criterion(x0 + deviation) for deviation in deviations]
    df = pd.DataFrame()
    df["x"] = deviations
    df["y"] = evaluations
    params = sm.ols(formula="y ~ x + I(x**2)", data=df).fit().params
    vec = np.array([1, (x -x0), (x - x0)  ** 2])
    return params @ vec
    
    
def plot_trust_region_algo(x0, radius, surrogate_func):
    fig, ax = plot_function()
    x0 = _unpack_x(x0)
    trust_x_grid = np.linspace(x0 - radius, x0 + radius, 50)
    partialed = functools.partial(surrogate_func, x0=x0, radius=radius)
    trust_y_grid = [partialed(x) for x in trust_x_grid]
    argmin_index = np.argmin(trust_y_grid)
    argmin = trust_x_grid[argmin_index]
    
    ax.plot([argmin], [partialed(np.array([argmin]))], marker="*", color=colors[5])
    ax.plot([argmin], [example_criterion(np.array([argmin]))], marker="*", color=colors[7])
    
    sns.lineplot(x=trust_x_grid, y=trust_y_grid, ax=ax, color = colors[1])
    
    new_x = x0 if example_criterion([argmin]) >= example_criterion(x0) else np.array([argmin])
    
    if surrogate_func == taylor_expansion:
        x_values = [x0]
    else:
        x_values = [x0 - radius, x0, x0 + radius]
        
    sns.rugplot(x_values, ax=ax, color = colors[7])
    return fig, ax, new_x


def plot_direct_search(x0, other):
    fig, ax = plot_function()
    x0 = _unpack_x(x0)
    other = _unpack_x(other)
    
    x_values = [x0, other]
    evaluations = [example_criterion(x) for x in x_values]
    
    argmin_index = np.argmin(evaluations)
    argmin = x_values[argmin_index]
    
    ax.plot([argmin], [example_criterion(argmin)], marker="*", color=colors[1])
    sns.rugplot(x_values, ax=ax, color=colors[1] )
    
    return fig, ax, argmin


def plot_line_search(x0):
    fig, ax = plot_function()
    x0 = _unpack_x(x0)
    
    function_value = example_criterion(x0)
    gradient_value = example_gradient(x0)
    approx_hessian_value = np.clip(example_hessian(x0), 0.1, np.inf)
    base_step = - 1 / approx_hessian_value * gradient_value
    
    gradient_grid = [x0 - 2, x0, x0 + 2]
    gradient_evals = [function_value - 2 * gradient_value, function_value, function_value + 2 * gradient_value]
    sns.lineplot(x=gradient_grid, y=gradient_evals, ax=ax, color=colors[7])
    
    new_value = np.inf
    x_values = [x0]
    evaluations = [function_value]
    alpha = 1
    while new_value >= function_value:
        new_x = x0 + alpha * base_step
        new_value = example_criterion(new_x)
        x_values.append(new_x)
        evaluations.append(new_value)
        
    sns.rugplot(x_values, ax=ax, color=colors[7])
    ax.plot([new_x], [new_value], marker="*", color=colors[1])
    return fig, ax, new_x
    
def plot_real_history(evaluated_points,i):
    fig, ax = plot_function()
    sns.rugplot(evaluated_points[0:i+1], ax=ax)
    plt.plot(evaluated_points[i], example_criterion(evaluated_points[i]), marker="*", color=colors[7])
    sns.despine()
    return fig, ax
