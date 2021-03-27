import os
import sys

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pytask
import src.model_code.functions_plotting as tools
from src.config import BLD
from src.config import SRC
from src.model_code.example_function import example_criterion
from src.model_code.example_function import example_gradient


def plot_real_L_BFGS_B(save_path):
    path = "docs/source/_static/images/intermediate_result/L_BFGS_B"
    if not os.path.isdir(path):
        os.makedirs(path)
    start_x = np.array([2])
    res = tools.minimize_with_history(
        example_criterion, start_x, method="L-BFGS-B", jac=example_gradient
    )
    evaluated_points = res.history
    filenames = []
    for i, value in enumerate(evaluated_points):
        fig, ax = tools.plot_real_history(evaluated_points, i)
        filename = os.path.join(path, f"L_BFGS{i}.png")
        filenames.append(filename)
        plt.savefig(filename, dpi=300)
        plt.close()
    images = list(map(lambda filename: imageio.imread(filename), filenames))
    imageio.mimsave(os.path.join(save_path), images, duration=0.5)


@pytask.mark.produces(BLD / "images" / "final_result" / "L_BFGS_B.gif")
def task_plot_real_L_BFGS_B(depends_on, produces):
    plot_real_L_BFGS_B(produces)
