import os
import sys

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import src.model_code.functions_plotting as tools
from src.config import BLD
from src.config import SRC


def plot_derivative_based_trust_region(textstr, save_path):
    path = (
        "docs/source/_static/images/intermediate_result/derivative_based_trust_region"
    )
    if not os.path.isdir(path):
        os.makedirs(path)
    filenames = []
    start_x = np.array([2])
    x = start_x
    start_radius = 2
    radius = 1
    for i in range(5):
        if i < 3:
            radius += 1
        else:
            radius = start_radius
            start_radius -= 1
        fig, ax, x = tools.plot_trust_region_algo(
            x, radius, surrogate_func=tools.taylor_expansion
        )
        fig.set_size_inches(8, 6)
        plt.subplots_adjust(bottom=0.2)
        plt.figtext(
            0.5,
            -0.01,
            textstr[i],
            multialignment="center",
            ha="center",
            fontsize=18,
            bbox={"facecolor": "white", "alpha": 0.05, "pad": 5},
        )
        x = x
        filename = os.path.join(path, f"derivative_based_trust_region{i}.png")
        filenames.append(filename)
        # save frame
        plt.savefig(filename, dpi=300)
        plt.close()

    # build gif
    images = list(map(lambda filename: imageio.imread(filename), filenames))
    imageio.mimsave(os.path.join(save_path), images, duration=3.5)


@pytask.mark.depends_on(SRC / "final" / "derivative_based_trust_region_algorithm.txt")
@pytask.mark.produces(
    BLD / "images" / "final_result" / "derivative_based_trust_region_algorithm.gif"
)
def task_plot_derivative_based_trust_region_algorithm(depends_on, produces):
    # Load locations after each round
    with open(depends_on, "r") as f:
        lines = f.readlines()
        textstr = [x + y for x, y in zip(lines[0::2], lines[1::2])]
    plot_derivative_based_trust_region(textstr, produces)
