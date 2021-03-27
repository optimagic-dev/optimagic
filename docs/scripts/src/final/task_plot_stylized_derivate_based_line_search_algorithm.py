import os
import sys

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pytask
import src.model_code.functions_plotting as tools
from src.config import BLD
from src.config import SRC


def plot_derivate_based_line_search_algorithm(textstr, save_path):
    path = "docs/source/_static/images/intermediate_result/derivate_based_line_search_algorithm"
    if not os.path.isdir(path):
        os.makedirs(path)
    filenames = []
    start_x = np.array([2])
    x = start_x
    n_frames = 10

    for i in range(7):
        fig, ax, x = tools.plot_line_search(x)
        fig.set_size_inches(8, 6)
        plt.subplots_adjust(bottom=0.2)
        x = x
        filename = os.path.join(path, f"derivate_based_line_search_algorithm{i}.png")
        filenames.append(filename)
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
        plt.close()

    images = list(map(lambda filename: imageio.imread(filename), filenames))
    imageio.mimsave(os.path.join(save_path), images, duration=1.5)


@pytask.mark.depends_on(SRC / "final" / "derivate_based_line_search_algorithm.txt")
@pytask.mark.produces(
    BLD / "images" / "final_result" / "derivate_based_line_search_algorithm.gif"
)
def task_plot_locations(depends_on, produces):
    # Load locations after each round
    with open(depends_on, "r") as f:
        lines = f.readlines()
        textstr = [x + y for x, y in zip(lines[0::2], lines[1::2])]
    plot_derivate_based_line_search_algorithm(textstr, produces)
