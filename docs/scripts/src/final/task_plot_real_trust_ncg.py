import pytask
import os
import sys
from src.config import SRC
from src.config import BLD
import numpy as np
import matplotlib.pyplot as plt
import src.model_code.functions_plotting as tools
import imageio
from src.model_code.example_function import example_criterion
from src.model_code.example_function import example_gradient
from src.model_code.example_function import example_hessian

def plot_real_trust_ncg(save_path):
    path = 'docs/source/_static/images/intermediate_result/trust_ncg'
    if not os.path.isdir(path): os.makedirs(path)
    filenames = []
    start_x = np.array([2])
    res = tools.minimize_with_history(example_criterion, start_x, method="trust-ncg", jac=example_gradient, hess=example_hessian)
    evaluated_points=res.history
    
    for i,value in enumerate(evaluated_points):
        fig, ax = tools.plot_real_history(evaluated_points,i)
        filename = os.path.join(path, f'trust_ncg{i}.png')
        filenames.append(filename)
        plt.savefig(filename, dpi=300)
        plt.close()
    images = list(map(lambda filename: imageio.imread(filename), filenames))
    imageio.mimsave(os.path.join(save_path), images, duration = 0.5)

@pytask.mark.produces(BLD/"images"/"final_result"/"Trust_NCG.gif")
def task_plot_locations(depends_on, produces):
    plot_real_trust_ncg(produces)