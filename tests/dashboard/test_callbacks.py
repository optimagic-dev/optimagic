import numpy as np
from bokeh.models import ColumnDataSource
from estimagic.dashboard.callbacks import _create_params_data_for_update
from estimagic.dashboard.callbacks import _reset_column_data_sources


PARAM_IDS = ["a", "b", "c", "d", "e"]


def test_create_params_data_for_update():
    param_ids = PARAM_IDS

    data = {
        "rowid": [1, 2, 3, 4, 5],
        "params": [
            np.array([0.47, 0.22, -0.46, 0.0, 2.0]),
            np.array([0.56, 0.26, 0.48, -0.30, 1.69]),
            np.array([0.50, 0.24, -0.15, -0.10, 1.89]),
            np.array([0.51, 0.24, -0.12, -0.10, 1.89]),
            np.array([0.52, 0.23, -0.12, -0.10, 1.90]),
        ],
    }

    expected = {
        "iteration": [1, 2, 3, 4, 5],
        "a": [0.47, 0.56, 0.50, 0.51, 0.52],
        "b": [0.22, 0.26, 0.24, 0.24, 0.23],
        "c": [-0.46, 0.48, -0.15, -0.12, -0.12],  # this wouldn't be plotted.
        "d": [0.0, -0.30, -0.10, -0.10, -0.10],
        "e": [2.0, 1.69, 1.89, 1.89, 1.90],
    }

    res = _create_params_data_for_update(
        data=data, param_ids=param_ids, clip_bound=1e100,
    )
    assert res == expected


def test_reset_column_data_sources():
    data = {"x": [0, 1, 2], "y": [2, 3, 4]}
    cds = ColumnDataSource(data)
    _reset_column_data_sources([cds])
    assert cds.data == {"x": [], "y": []}
