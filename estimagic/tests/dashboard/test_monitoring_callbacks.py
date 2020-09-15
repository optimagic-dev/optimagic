from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource
from bokeh.models import Toggle
from bokeh.plotting import figure

from estimagic.dashboard.monitoring_callbacks import _create_params_data_for_update
from estimagic.dashboard.monitoring_callbacks import _reset_column_data_sources
from estimagic.dashboard.monitoring_callbacks import _switch_to_linear_scale
from estimagic.dashboard.monitoring_callbacks import _switch_to_log_scale
from estimagic.dashboard.monitoring_callbacks import _update_monitoring_tab
from estimagic.logging.database_utilities import load_database

PARAM_IDS = ["a", "b", "c", "d", "e"]


def test_switch_to_log_scale():
    button = Toggle(active=False)
    linear_plot = figure(name="linear_plot")
    log_plot = figure(name="log_plot")

    _switch_to_log_scale(
        button=button, linear_criterion_plot=linear_plot, log_criterion_plot=log_plot
    )

    assert linear_plot.visible is False
    assert log_plot.visible is True
    assert button.button_type == "primary"


def test_switch_to_linear_scale():
    button = Toggle(active=False)
    linear_plot = figure(name="linear_plot")
    log_plot = figure(name="log_plot")

    _switch_to_linear_scale(
        button=button, linear_criterion_plot=linear_plot, log_criterion_plot=log_plot
    )

    assert linear_plot.visible is True
    assert log_plot.visible is False
    assert button.button_type == "default"


def test_update_monitoring_tab():
    # note: this test database does not include None in the value column.
    # it has only 7 entries.
    db_path = Path(__file__).parent / "db1.db"
    database = load_database(metadata=None, path=db_path)

    crit_data = {"iteration": [3, 5], "criterion": [-10, -10]}
    criterion_cds = ColumnDataSource(crit_data)

    param_data = {f"p{i}": [i, i, i] for i in range(6)}
    param_data["iteration"] = [3, 4, 5]
    plotted_param_data = {
        k: v for k, v in param_data.items() if k in ["p0", "p2", "p4", "iteration"]
    }
    param_cds = ColumnDataSource(plotted_param_data)

    start_params = pd.DataFrame()
    start_params["group"] = ["g1", "g1", None, None, "g2", "g2"]
    start_params["id"] = [f"p{i}" for i in range(6)]

    session_data = {"last_retrieved": 5}
    tables = []  # not used
    rollover = 500
    update_chunk = 5

    expected_crit_data = {
        "iteration": [3, 5, 6, 7],
        "criterion": [-10, -10] + [3.371916994681647e-18, 3.3306686770405823e-18],
    }

    expected_param_data = plotted_param_data.copy()
    expected_param_data["iteration"] += [6, 7]
    expected_param_data["p0"] += [
        -7.82732387e-10,
        -7.45058016e-10,
    ]
    expected_param_data["p2"] += [
        -7.50570405e-10,
        -7.45058015e-10,
    ]
    expected_param_data["p4"] += [
        -7.44958198e-10,
        -7.45058015e-10,
    ]

    _update_monitoring_tab(
        database=database,
        criterion_cds=criterion_cds,
        param_cds=param_cds,
        session_data=session_data,
        tables=tables,
        rollover=rollover,
        start_params=start_params,
        update_chunk=update_chunk,
    )

    assert session_data["last_retrieved"] == 7
    assert criterion_cds.data == expected_crit_data
    assert param_cds.data == expected_param_data


def test_create_params_data_for_update():
    param_ids = PARAM_IDS

    data = {
        "rowid": [1, 2, 3, 4, 5],
        "external_params": [
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
        data=data, param_ids=param_ids, clip_bound=1e100
    )
    assert res == expected


def test_reset_column_data_sources():
    data = {"x": [0, 1, 2], "y": [2, 3, 4]}
    cds = ColumnDataSource(data)
    _reset_column_data_sources([cds])
    assert cds.data == {"x": [], "y": []}
