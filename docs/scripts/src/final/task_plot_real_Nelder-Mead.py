import numpy as np
import pandas as pd
import pytask
from src.config import SRC
from src.config import BLD
import matplotlib.pyplot as plt
import src.model_code.functions_plotting  as tools
import pytask
import os
import sys
import imageio
import statsmodels.formula.api as sm
from scipy.optimize import minimize
import seaborn as sns
from src.model_code.example_function import example_criterion


def plot_real_Nelder_Mead(save_path):
    path = 'docs/source/_static/images/intermediate_result/Nelder_Mead'
    if not os.path.isdir(path): os.makedirs(path)
    start_x = np.array([2])
    res = tools.minimize_with_history(example_criterion, start_x, method="Nelder-Mead")
    evaluated_points=res.history
    filenames = []
    for i,value in enumerate(evaluated_points):
        fig, ax = tools.plot_real_history(evaluated_points,i)
        filename = os.path.join(path, f'Nelder_Mead{i}.png')
        filenames.append(filename)
        plt.savefig(filename, dpi=300)
        plt.close()
    images = list(map(lambda filename: imageio.imread(filename), filenames))
    imageio.mimsave(os.path.join(save_path), images, duration = 0.5)

@pytask.mark.produces(BLD/"images"/"final_result"/"Nelder_Mead.gif")
def task_plot_real_Nelder_Mead(depends_on, produces):
    plot_real_Nelder_Mead(produces)


    