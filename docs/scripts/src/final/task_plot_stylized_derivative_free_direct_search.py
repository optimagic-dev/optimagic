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


def plot_derivative_free_direct_search(textstr, save_path):
    path = (
        "docs/source/_static/images/intermediate_result/derivative_free_direct_search"
    )
    if not os.path.isdir(path):
        os.makedirs(path)
    filenames = []
    start_x = np.array([2])
    x = start_x
    for i in range(5):
        if i == 0:
            fig, ax, x = tools.plot_direct_search(x, x - 2)
        elif i < 4:
            fig, ax, x = tools.plot_direct_search(x, x + i + 1)
        else:
            fig, ax, x = tools.plot_direct_search(x, x + 2)
        x = x
        fig.set_size_inches(8, 6)
        plt.subplots_adjust(bottom=0.25)
        # create file name and append it to a list
        filename = os.path.join(path, f"derivative_free_direct_search{i}.png")
        filenames.append(filename)
        # save frame
        plt.figtext(
            0.5,
            -0.01,
            textstr[i],
            multialignment="center",
            ha="center",
            wrap=True,
            fontsize=18,
            bbox={"facecolor": "white", "alpha": 0.5, "pad": 5},
        )
        plt.savefig(filename, dpi=300)
        plt.savefig(filename, dpi=300)
        plt.close()

    # build gif
    images = list(map(lambda filename: imageio.imread(filename), filenames))
    imageio.mimsave(os.path.join(save_path), images, duration=3.5)


@pytask.mark.depends_on(SRC / "final" / "derivative_free_direct_search_algorithm.txt")
@pytask.mark.produces(
    BLD / "images" / "final_result" / "derivative_free_direct_search_algorithm.gif"
)
def task_plot_derivative_free_direct_search(depends_on, produces):
    # Load locations after each round
    with open(depends_on, "r") as f:
        lines = f.readlines()
        textstr = [x + y for x, y in zip(lines[0::2], lines[1::2])]
    plot_derivative_free_direct_search(textstr, produces)
