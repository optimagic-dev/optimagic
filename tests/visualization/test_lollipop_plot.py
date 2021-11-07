import numpy as np
import pandas as pd
from estimagic.visualization.lollipop_plot import lollipop_plot


def test_lollipop_plot_runs():
    df = pd.DataFrame(
        np.arange(12).reshape(4, 3),
        index=pd.MultiIndex.from_tuples([(0, "a"), ("b", 1), ("a", "b"), (2, 3)]),
        columns=["a", "b", "c"],
    )

    lollipop_plot(df)
