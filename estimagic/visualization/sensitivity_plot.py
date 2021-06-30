"""
Draw lollipop plots of moment sensitivity measures.
"""
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def moment_sensitivity_plot(sensitivity):
    """
    Args:
        sensitivity (pd.DataFrame or list of pd.DataFrame)
    """

    number_sens = len(sensitivity)

    measures = list(sensitivity.keys())
    df_columns = sensitivity[measures[0]].index.get_level_values("name").to_list()
    number_params = len(df_columns)

    figures = []

    for name, df in sensitivity.items():
        df = df.copy(deep=True)
        df.iloc[:, -number_sens:] = df.iloc[:, -number_sens:].abs()

        df = df.T[-number_sens:]
        df.columns = df_columns
        df.reset_index(inplace=True)

        # Make the PairGrid
        g = sns.PairGrid(
            data=df,
            x_vars=df.columns[-number_params:],
            y_vars=["index"],
            height=7,
            aspect=0.25,
        )

        # Draw a dot plot using the stripplot function
        g.map(
            sns.stripplot,
            jitter=False,
            size=12,
            orient="h",
            palette="ch:s=1,r=-.1,h=1_r",
            linewidth=1,
            edgecolor="w",  # color of dot edge
        )

        # Use the same x axis limits on all columns and add better labels
        g.set(
            xlabel="",
            ylabel="",
        )

        # Use semantically meaningful titles for the columns
        titles = ["intersection", "beta1", "beta2"]

        for ax, title in zip(g.axes.flat, titles):

            # Set a different title for each axes
            ax.set(title=title)
            ax.set_xlim(left=0)

            # Make the grid horizontal instead of vertical
            ax.xaxis.grid(False)
            ax.yaxis.grid(True)

        sns.despine(bottom=True)

        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(name)

        figures.append(g)

    return figures
